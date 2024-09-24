'''
Date: 2024-03-11 18:00:06
LastEditors: jackeymiao
LastEditTime: 2024-04-07 09:48:19
FilePath: /ADNet/gen_data.py
'''
import argparse
import os
import numpy as np
import torch

from utils.data_utils import check_extension, save_dataset


def generate_MultiPM_data(n_samples, n_users, pk, radius):
    data = []
    for _ in range(n_samples):
        c = torch.FloatTensor(1, n_users).uniform_(2, 4)
        for i in range(len(pk) - 1):
            c_temp = c[-1]
            factor = torch.FloatTensor(1, n_users).uniform_(0.8, 0.88)
            c = torch.cat((c, c_temp * factor))

        loc = torch.FloatTensor(n_users, 2).uniform_(0, 1)
        combined = torch.cat((loc,c[0].reshape(n_users, 1)),axis=1)
        data.append(dict(loc=loc,
                    radius=radius,
                    pk=torch.IntTensor(pk),
                    c = c,
                    combined=combined,
                    ))
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")
    parser.add_argument("--problem", type=str, default='MultiPM',
                        help="Problem, 'MultiPM' to generate")

    parser.add_argument("--dataset_size", type=int, default=1000, help="Size of the dataset")
    parser.add_argument('--graph_size', type=int, default=2000,
                        help="number of users")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")
    parser.add_argument('--pk', nargs='+', type=int, default=[2, 4, 7, 9, 10, 13, 15])

    opts = parser.parse_args()

    assert opts.filename is None or (len(opts.problems) == 1 and len(opts.graph_sizes) == 1), \
        "Can only specify filename when generating a single dataset"

    torch.manual_seed(1234)
    problem = opts.problem
    n_users = opts.graph_size

    datadir = os.path.join(opts.data_dir, problem)
    os.makedirs(datadir, exist_ok=True)

    if problem == 'MultiPM':
        if n_users == 20:
            radius = 0.32
        elif n_users == 50:
            radius = 0.24
        elif n_users ==100:
            radius = 0.16
        else:
            radius = 0.16
        pk = opts.pk
        filename = os.path.join(datadir, f"{problem}_{n_users}_{radius}_{pk}.pkl")
        dataset = generate_MultiPM_data(opts.dataset_size, n_users, pk, radius)
    else:
        assert False, "Unknown problem: {}".format(problem)

    save_dataset(dataset, filename)



