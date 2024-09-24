import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple
from utils.tensor_functions import compute_in_batches

from nets.graph_encoder import GraphAttentionEncoder
from torch.nn import DataParallel
from utils.beam_search import CachedLookup
from utils.functions import sample_many


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return AttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key],
                glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
                glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
                logit_key=self.logit_key[key]
            )
        # return super(AttentionModelFixed, self).__getitem__(key)
        return tuple.__getitem__(self, key)


def _get_attention_node_data(fixed):
    return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key


class AttentionModel(nn.Module):

    def __init__(self,
                 p,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None,
                 dy=True):
        super(AttentionModel, self).__init__()

        self.p = p
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0

        self.is_PM = problem.NAME == 'PM'
        self.is_PC = problem.NAME == 'PC'
        self.is_MCLP = problem.NAME == 'MCLP'
        self.is_LSCP = problem.NAME == 'LSCP'
        self.is_MultiPM = problem.NAME == 'MultiPM'
        self.is_dy = dy
        self.tanh_clipping = tanh_clipping
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size

        step_context_dim = 2 * embedding_dim
        node_dim = 3  # x, y
        # Learned input symbols for first action
        self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))
        self.W_placeholder.data.uniform_(-1, 1)  # Placeholder should be in range of activations
        self.first_placeholder = nn.Parameter(torch.Tensor(embedding_dim))
        self.first_placeholder.data.uniform_(-1, 1)

        self.init_embed = nn.Linear(node_dim, embedding_dim)
        # self.init_dynamic = nn.Linear(1, embedding_dim//4)
        self.init_dynamic = nn.Linear(2, embedding_dim)
        self.l2_dynamic = nn.Linear(embedding_dim//4, embedding_dim//2)
        self.l3_dynamic = nn.Linear(embedding_dim//2, embedding_dim)

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )
        self.gru = nn.GRU(embedding_dim, embedding_dim, 1,
                          batch_first=True)
        self.gru1 = nn.GRU(embedding_dim, embedding_dim, 1,
                          batch_first=True)
        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input, return_pi=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """
        
        if self.checkpoint_encoder and self.training:  # Only checkpoint if we need gradients
            embeddings, _ = checkpoint(self.embedder, self._init_embed(input))
        else:
            embeddings, _ = self.embedder(self._init_embed(input))

        _log_p, pi, cc = self._inner(input, embeddings)
        
        if self.is_PM or self.is_PC or self.is_MultiPM:
            cost = self.problem.get_total_dis(input, pi)
        elif self.is_MCLP:
            cover_num = self.problem.get_total_num(input, pi)
            cost = -cover_num
        elif self.is_LSCP:
            # cost = self.problem.get_facility_num(input, pi)
            cost = cc.flatten()
        else:
            assert False, "Unknown problem"
        
        ll = _log_p.sum(-1)

        for i in range(ll.size(0)):
            ll[i] = ll[i] / torch.nonzero(_log_p[i]).size(0) * 50

        if return_pi:
            return cost, ll, pi

        return cost, ll

    def precompute_fixed(self, input):
        embeddings, _ = self.embedder(self._init_embed(input))
        # Use a CachedLookup such that if we repeatedly index this object with the same index we only need to do
        # the lookup once... this is the case if all elements in the batch have maximum batch size
        return CachedLookup(self._precompute(embeddings))

    def propose_expansions(self, beam, fixed, expand_size=None, normalize=False, max_calc_batch_size=4096):
        # First dim = batch_size * cur_beam_size
        log_p_topk, ind_topk = compute_in_batches(
            lambda b: self._get_log_p_topk(fixed[b.ids], b.state, k=expand_size, normalize=normalize),
            max_calc_batch_size, beam, n=beam.size()
        )

        assert log_p_topk.size(1) == 1, "Can only have single step"
        # This will broadcast, calculate log_p (score) of expansions
        score_expand = beam.score[:, None] + log_p_topk[:, 0, :]

        # We flatten the action as we need to filter and this cannot be done in 2d
        flat_action = ind_topk.view(-1)
        flat_score = score_expand.view(-1)
        flat_feas = flat_score > -1e10  # != -math.inf triggers

        # Parent is row idx of ind_topk, can be found by enumerating elements and dividing by number of columns
        flat_parent = torch.arange(flat_action.size(-1), out=flat_action.new()) // ind_topk.size(-1)

        # Filter infeasible
        feas_ind_2d = torch.nonzero(flat_feas)

        if len(feas_ind_2d) == 0:
            # Too bad, no feasible expansions at all :(
            return None, None, None

        feas_ind = feas_ind_2d[:, 0]

        return flat_parent[feas_ind], flat_action[feas_ind], flat_score[feas_ind]

    def _calc_log_likelihood(self, _log_p, a, mask):

        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)

    def _init_embed(self, input):
        loc = input['combined']
        return self.init_embed(loc)

    def _inner(self, input, embeddings):

        outputs = []
        sequences = []

        state = self.problem.make_state(input)


        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = self._precompute(embeddings)

        last_hh = None
        last_hh1 = None

        # Perform decoding steps
        while not (state.all_finished()):
            log_p, mask, last_hh, last_hh1 = self._get_log_p_gru(fixed, state, normalize=True, last_hh=last_hh, last_hh1=last_hh1)

            # Select the indices of the next nodes in the sequences, result (batch_size) long
            logp_selected, selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])

            # if self.is_LSCP:
            #     for instance in range(len(mask)):
            #         if state.instance_finished(instance):

            # Squeeze out steps dimension

            state = state.update(selected)

            if self.is_LSCP:
                state.instance_finished()


            # Collect output of step
            outputs.append(logp_selected)
            sequences.append(selected)
        # print(state.sol_length)
        # Collected lists, return Tensor
        if self.is_LSCP:
            return torch.stack(outputs, 1), torch.stack(sequences, 1), state.sol_length
        else:
            return torch.stack(outputs, 1), torch.stack(sequences, 1), 0

    def sample_many(self, input, batch_rep=1, iter_rep=1, isLSCP=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :return:
        """
        # Bit ugly but we need to pass the embeddings as well.
        # Making a tuple will not work with the problem.get_cost function
        return sample_many(
            lambda input: self._inner(*input),  # Need to unpack tuple into arguments
            lambda input, pi: self.problem.get_total_dis(input[0], pi),  # Don't need embeddings as input to get_costs
            # lambda input, pi: self.problem.get_total_num(input[0], pi),  # Don't need embeddings as input to get_costs
            (input, self.embedder(self._init_embed(input))[0]),  # Pack input with embeddings (additional input)
            batch_rep, iter_rep, isLSCP
        )

    def _select_node(self, probs, mask):
        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            logp, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)
            logp = probs.gather(-1, selected.unsqueeze(1)).squeeze(1)
            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)
                logp = probs.gather(-1, selected.unsqueeze(1)).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return logp.log(), selected

    def _precompute(self, embeddings, num_steps=1):

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _get_log_p_topk(self, fixed, state, k=None, normalize=True):
        log_p, _ = self._get_log_p(fixed, state, normalize=normalize)

        # Return topk
        if k is not None and k < log_p.size(-1):
            return log_p.topk(k, -1)

        # Return all, note different from torch.topk this does not give error if less than k elements along dim
        return (
            log_p,
            torch.arange(log_p.size(-1), device=log_p.device, dtype=torch.int64).repeat(log_p.size(0), 1)[:, None, :]
        )

    def _get_log_p_gru(self, fixed, state, normalize=True, last_hh=None, last_hh1=None):

        current_node = state.get_current_node()
        batch_size, num_steps = current_node.size()
        if state.i.item() == 0:
            # decoder_hidden = self.x0.expand(batch_size, -1, -1)
            decoder_input = self.first_placeholder[None, None, :].expand(batch_size, 1, -1)
        else:
            decoder_input = fixed.node_embeddings.gather(
                1,
                current_node[:, :, None].expand(batch_size, 1, fixed.node_embeddings.size(-1))
            )

        # decoder_hidden = self.project_step_context(decoder_hidden)

        if last_hh is None:
            last_hh = fixed.context_node_projected.transpose(1, 0)

        # query = last_hh

        self.gru.flatten_parameters()
        rnn_out, last_hh = self.gru(decoder_input, last_hh)

        # query = self.project_query(torch.cat((rnn_out,fixed.context_node_projected), dim=-1))
        query = rnn_out
        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = _get_attention_node_data(fixed)

        # ====================================
        # add dynamic
        a = state.c[:,state.pp,:].transpose(1, 2).clone()
        b = state.get_dynamic().transpose(1, 2).clone()
        dynamic_input = torch.cat((a,b),2)
        # print(state.c[:,state.i,:].transpose(1, 2).size())
        # print(state.get_dynamic().transpose(1, 2).size())
        dynamic_hidden = self.init_dynamic(dynamic_input)
        # print(dynamic_hidden)
        # print(dynamic_hidden.size())
        #TODO:GRU
        # print(dynamic_hidden.size())
        self.gru1.flatten_parameters()
        rnn_out1, last_hh1 = self.gru1(dynamic_hidden, last_hh1)
        # logit_K = logit_K + rnn_out1.unsqueeze(1)
        logit_K = logit_K.mul(1 + rnn_out1.unsqueeze(1))
        # logit_K = state.get_dynamic().transpose(1, 2).expand_as(logit_K.squeeze(1)).mul(logit_K.squeeze(1)).unsqueeze(1)
        # ====================================
        # dynamic_hidden = self.init_dynamic(state.get_dynamic().transpose(1, 2))
        # # logit_K = logit_K + dynamic_hidden.unsqueeze(1)
        # logit_K = logit_K.mul(1 + dynamic_hidden.unsqueeze(1))
        # print(logit_K)

        # Compute the mask
        mask = state.get_mask()

        # Compute logits (unnormalized log_p)
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)

        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p, mask, last_hh, last_hh1

    def _get_parallel_step_context(self, embeddings, state, from_depot=False):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)
        
        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """

        current_node = state.get_current_node()
        batch_size, num_steps = current_node.size()

        if num_steps == 1:  # We need to special case if we have only 1 step, may be the first or not
            if state.i.item() == 0:
                # First and only step, ignore prev_a (this is a placeholder)
                return self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1))
            else:
                return embeddings.gather(
                    1,
                    torch.cat((state.first_a, current_node), 1)[:, :, None].expand(batch_size, 2, embeddings.size(-1))
                ).view(batch_size, 1, -1)
        # More than one step, assume always starting with first
        embeddings_per_step = embeddings.gather(
            1,
            current_node[:, 1:, None].expand(batch_size, num_steps - 1, embeddings.size(-1))
        )
        return torch.cat((
            # First step placeholder, cat in dim 1 (time steps)
            self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1)),
            # Second step, concatenate embedding of first with embedding of current/previous (in dim 2, context dim)
            torch.cat((
                embeddings_per_step[:, 0:1, :].expand(batch_size, num_steps - 1, embeddings.size(-1)),
                embeddings_per_step
            ), 2)
        ), 1)

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )
