'''
Date: 2024-03-11 18:00:06
LastEditors: jackeymiao
LastEditTime: 2024-04-03 12:08:38
FilePath: /ADNet/problems/MultiPM/problem_MultiPM.py
'''
from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.MultiPM.state_MultiPM import StateMultiPM


class MultiPM(object):
    NAME = 'MultiPM'

    @staticmethod
    def get_total_dis(dataset, pi):
        loc = dataset['loc']
        pk = dataset['pk'][0]
        c = dataset['c']
        period = len(pk)
        batch_size, n_loc, _ = loc.size()

        dist = (loc[:, :, None, :] - loc[:, None, :, :]).norm(p=2, dim=-1)
        total_length = 0
        total_cost = 0
        for i in range(period):
            pi_temp = pi[:,:pk[i]]
            assert pk[i] == len(pi_temp[0])
            p = pk[i]
            facility_tensor = pi_temp.unsqueeze(-1).expand_as(torch.Tensor(batch_size, p, n_loc))
            dist_p = dist.gather(1, facility_tensor)
            length = torch.min(dist_p, 1)
            lengths = length[0].sum(-1)
            total_length = total_length +lengths

            pi_part = pi[:,:pk[i]] if i == 0 else pi[:,pk[i-1]:pk[i]]
            cost_part = torch.zeros_like(pi_part,dtype=torch.float32)
            c_part = c[:, i]
            jj, kk = pi_part.size()
            # for j in range(jj):
            #     for k in range(kk):
            #         cost_part[j][k] = c_part[pi_part[j][k]]
            jj_indices = torch.arange(jj).view(-1, 1)  # 创建 jj 长度的列向量
            cost_part = c_part[jj_indices, pi_part]
            cost = cost_part.sum(-1)

            total_cost = total_cost + cost
        objective = total_cost + total_length    

        return objective


    @staticmethod
    def make_dataset(*args, **kwargs):
        return MultiPMDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateMultiPM.initialize(*args, **kwargs)


class MultiPMDataset(Dataset):
    def __init__(self, filename=None, size=50, num_samples=5000, offset=0, pk=[1, 2, 2], r=0.32, distribution=None):
        super(MultiPMDataset, self).__init__()

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [row for row in (data[offset:offset + num_samples])]
        else:
            # Sample points randomly in [0, 1] square

            self.data = []
            for _ in range(num_samples):
                c = torch.FloatTensor(1, size).uniform_(2, 4)
                for i in range(len(pk) - 1):
                    c_temp = c[-1]
                    factor = torch.FloatTensor(1, size).uniform_(0.8, 0.88)
                    c = torch.cat((c, c_temp * factor))

                loc = torch.FloatTensor(size, 2).uniform_(0, 1)
                combined = torch.cat((loc,c[0].reshape(size, 1)),axis=1)
                self.data.append(dict(loc=loc,
                            radius=r,
                            pk=torch.IntTensor(pk),
                            c = c,
                            combined=combined,
                            ))

        self.size = len(self.data)
        self.pk = pk
        self.r = r

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
