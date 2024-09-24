import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter


class StateMultiPM(NamedTuple):
    # Fixed input
    loc: torch.Tensor
    pk: torch.Tensor
    radius: torch.Tensor
    dist: torch.Tensor
    c: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the loc and dist tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    first_a: torch.Tensor
    prev_a: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    mask_cover: torch.Tensor
    dynamic: torch.Tensor
    facility: torch.Tensor  # B x p
    length: torch.Tensor  # obj val of current solution
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step
    pp: torch.Tensor

    @property
    def visited(self):
        if self.visited_.dtype == torch.bool:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.loc.size(-2))

    @staticmethod
    def initialize(data, visited_dtype=torch.bool):
        loc = data['loc']
        pk = data['pk'][0]
        radius = data['radius'][0]
        c = data['c']
        period = len(pk)
        batch_size, n_loc, _ = loc.size()
        dist = (loc[:, :, None, :] - loc[:, None, :, :]).norm(p=2, dim=-1)

        # facility_list = [[] for i in range(batch_size)]
        facility = torch.tensor([], dtype=torch.int64, device=loc.device)
        length = torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device)
        prev_a = torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device)
        
        return StateMultiPM(
            loc=loc,
            pk=pk,
            radius=radius,
            dist=dist,
            c=c,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(
                    batch_size, 1, n_loc,
                    dtype=torch.bool, device=loc.device
                )
                if visited_dtype == torch.bool
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            mask_cover=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(
                    batch_size, 1, n_loc,
                    dtype=torch.bool, device=loc.device
                )
                if visited_dtype == torch.bool
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            dynamic=torch.ones(batch_size, 1, n_loc, dtype=torch.float, device=loc.device),
            facility=facility,
            length=length,
            prev_a=prev_a,
            first_a=prev_a,
            cur_coord=None,
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),  # Vector with length num_steps
            pp=torch.zeros(1, dtype=torch.int64, device=loc.device)
        )

    def get_final_cost(self):

        assert self.all_finished()
        # assert self.visited_.

        return self.length

    def get_length(self, facility):
        """
        :param facility: list, a list of facility index list,  if None, generate randomly
        :return: obj val of given facility_list
        """

        batch_size, n_loc, _ = self.loc.size()
        _, p = facility.size()
        facility_tensor = facility.unsqueeze(-1).expand_as(torch.Tensor(batch_size, p, n_loc))
        f_u_dist_tensor = self.dist.gather(1, facility_tensor)
        lengths = torch.sum(torch.min(f_u_dist_tensor, dim=1)[0], dim=-1)
        return lengths

    def update(self, selected):
        # Update the state
        cur_selected = selected.unsqueeze(-1)
        prev_a = cur_selected  # Add dimension for step
        cur_coord = self.loc[self.ids, prev_a]

        cur_facility = self.facility
        new_facility = torch.cat((cur_facility, cur_selected), dim=1)
        new_length = self.get_length(new_facility)

        first_a = prev_a if self.i.item() == 0 else self.first_a

        if self.visited_.dtype == torch.bool:
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            visited_ = mask_long_scatter(self.visited_, prev_a)

        # mask covered cities
        batch_size, sequence_size, _ = self.loc.size()
        dists = (self.loc[self.ids.squeeze(-1)] - cur_coord).norm(p=2, dim=-1).unsqueeze(1)
        dynamic = self.dynamic.clone()
        mask_cover = dists <= self.radius

        mask = dists > self.radius
        dynamic_update = dists.masked_fill(mask, value=1)
        dynamic = dynamic.mul(dynamic_update)
        pp = self.pp

        while self.i + 1 > self.pk[pp]:
            pp = pp + 1

        return self._replace(first_a=first_a, prev_a=prev_a, visited_=visited_, mask_cover=mask_cover,
                             dynamic=dynamic, length=new_length, cur_coord=cur_coord, i=self.i + 1, pp=pp)

    def all_finished(self):
        return self.i == self.pk[-1]

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        return self.visited_  # Hacky way to return bool or uint8 depending on pytorch version

    def get_dynamic(self):
        return self.dynamic
