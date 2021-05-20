import os
from argparse import ArgumentParser
import torch
import numpy as np

from regretnet import datasets as ds
from regretnet.regretnet import RegretNet, RegretNetUnitDemand
from regretnet.datasets import Dataloader
import pdb

def normalize_allocs(allocs):
    allocs = allocs.clamp_min(1e-8)
    norm_allocs = allocs / allocs.sum(dim=-1).unsqueeze(-1)

    return norm_allocs

def get_entropy(allocs):
    entropy = -1.0 * allocs * torch.log(allocs)

    loss = entropy.sum(dim=-1).sum(dim=-1)
    return loss

def get_unfairness(allocs, n_agents, n_items):
    allocs = allocs.cpu()
    d = 0.0
    C = [[i for i in range(n_agents)]]
    D = (torch.ones(1, n_items, n_items) * d)
    L, n, m = allocs.shape
    unfairness = torch.zeros(L, m)
    for i, C_i in enumerate(C):
        for u in range(m):
            for v in range(m):
                subset_allocs_diff = (allocs[:, C_i, u] - allocs[:, C_i, v]).abs()
                D2 = 1 - (1 - D) if n == 1 else 2 - (2 - D)
                unfairness[:, u] += (subset_allocs_diff.sum(dim=1) - D2[i, u, v]).clamp_min(0)
    
    tvf_alloc = unfairness.sum(dim=-1)
    return tvf_alloc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = ArgumentParser()

parser.add_argument('--random-seed', type=int, default=0)
parser.add_argument('--n-agents', type=int, default=1)
parser.add_argument('--n-items', type=int, default=2)
parser.add_argument('--dataset', nargs='+', default=[], required=True)

parser.add_argument('--test-num-examples', type=int, default=5000)
parser.add_argument('--test-batch-size', type=int, default=512)

auction_type = "mv"
models = ["../diversity/result/1x2-{}_1_0.0_0/199_checkpoint.pt".format(auction_type),
          "../diversity/result/1x2-{}_1_1.0_0/199_checkpoint.pt".format(auction_type),
          "../fairness/result/1x2-{}_1_0.0_0/199_checkpoint.pt".format(auction_type),
          "../learnable_preferences/result/entropy_ranking/1x2-{}_1_1.0_1.0_0/199_checkpoint.pt".format(auction_type),
          "../learnable_preferences/result/tvf_ranking/1x2-{}_1_1.0_1.0_0/199_checkpoint.pt".format(auction_type)]


args = parser.parse_args()
torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)

average_allocation_distance = []
for modelA_path in models:
    average_allocations = []
    for modelB_path in models:

        modelA_ckpt = torch.load(modelA_path)
        if "pv" in modelA_ckpt['name']:
            modelA = RegretNetUnitDemand(**(modelA_ckpt['arch']))
        else:
            modelA = RegretNet(**(modelA_ckpt['arch']))

        state_dict = modelA_ckpt['state_dict']
        modelA.load_state_dict(state_dict)


        modelB_ckpt = torch.load(modelB_path)
        if "pv" in modelB_ckpt['name']:
            modelB = RegretNetUnitDemand(**(modelB_ckpt['arch']))
        else:
            modelB = RegretNet(**(modelB_ckpt['arch']))

        state_dict = modelB_ckpt['state_dict']
        modelB.load_state_dict(state_dict)

        modelA.to(DEVICE)
        modelB.to(DEVICE)

        modelA.eval()
        modelB.eval()

        # Valuation range setup
        item_ranges = ds.preset_valuation_range(1, 2, args.dataset)
        clamp_op = ds.get_clamp_op(item_ranges)

        test_data = ds.generate_dataset_nxk(1, 2, args.test_num_examples, item_ranges).to(DEVICE)
        test_loader = Dataloader(test_data, batch_size=args.test_batch_size, shuffle=True)

        total_dist = 0
        total_entropy = 0
        total_unfairness = 0

        for i, batch in enumerate(test_loader):
            batch = batch.to(DEVICE)
            allocs_A, _ = modelA(batch)
            allocs_B, _ = modelB(batch)

            unfair_allocs_A = get_unfairness(allocs_A, modelA_ckpt['arch']['n_agents'], modelA_ckpt['arch']['n_items'])
            unfair_allocs_B = get_unfairness(allocs_B, modelA_ckpt['arch']['n_agents'], modelA_ckpt['arch']['n_items'])

            norm_allocs_A = normalize_allocs(allocs_A)
            norm_allocs_B = normalize_allocs(allocs_B)
            
            entropy_A = get_entropy(norm_allocs_A)
            entropy_B = get_entropy(norm_allocs_B)

            total_dist = total_dist + torch.cdist(allocs_A, allocs_B).view(-1).sum().item()
            total_entropy = total_entropy + torch.sqrt(torch.pow(entropy_A - entropy_B, 2)).sum().item()
            total_unfairness = total_unfairness + torch.sqrt(torch.pow(unfair_allocs_A - unfair_allocs_B, 2)).sum().item()

        #print(modelA_path + " , " + modelB_path)
        #print("Average Allocation Distance: {}".format(total_dist / args.test_num_examples))
        average_allocations.append(total_dist / args.test_num_examples)
        #print("Average Entropy Distance: {}".format(total_entropy / args.test_num_examples))
        #print("Average Unfairness Distance: {}".format(total_unfairness / args.test_num_examples))
    average_allocation_distance.append(average_allocations)
print(np.matrix(average_allocation_distance))

