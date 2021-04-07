import os
from argparse import ArgumentParser
import torch
import numpy as np

from regretnet import datasets as ds
from regretnet.regretnet import RegretNet, RegretNetUnitDemand
from regretnet.datasets import Dataloader
import pdb

def normalize_allocs(allocs):
    norm_allocs = allocs / allocs.sum(dim=-1).unsqueeze(-1)
    return norm_allocs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = ArgumentParser()
parser.add_argument('--modelA_ckpt', required=True)
parser.add_argument('--modelB_ckpt', required=True)

parser.add_argument('--random-seed', type=int, default=0)
parser.add_argument('--test-num-examples', type=int, default=5000)
parser.add_argument('--test-batch-size', type=int, default=512)

if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    modelA_ckpt = torch.load(args.modelA_ckpt)
    if "pv" in modelA_ckpt['name']:
        modelA = RegretNetUnitDemand(**(modelA_ckpt['arch']))
    else:
        modelA = RegretNet(**(modelA_ckpt['arch']))
    
    state_dict = modelA_ckpt['state_dict']
    modelA.load_state_dict(state_dict)


    modelB_ckpt = torch.load(args.modelB_ckpt)
    if "pv" in modelB_ckpt['name']:
        modelB = RegretNetUnitDemand(**(modelB_ckpt['arch']))
    else:
        modelB = RegretNet(**(modelB_ckpt['arch']))

    state_dict = modelB_ckpt['state_dict']
    modelB.load_state_dict(state_dict)

    modelA_ckpt['arch']
    # Valuation range setup
    item_ranges = ds.preset_valuation_range(modelA_ckpt['arch']['n_agents'], modelA_ckpt['arch']['n_items'])
    clamp_op = ds.get_clamp_op(item_ranges)
 
    test_data = ds.generate_dataset_nxk(modelA_ckpt['arch']['n_agents'], modelA_ckpt['arch']['n_items'], args.test_num_examples, item_ranges).to(DEVICE)
    test_loader = Dataloader(test_data, batch_size=args.test_batch_size, shuffle=True)

    modelA.to(DEVICE)
    modelB.to(DEVICE)

    modelA.eval()
    modelB.eval()
    
    total = 0
    for i, batch in enumerate(test_loader):
        batch = batch.to(DEVICE)
        allocs_A, _ = modelA(batch)
        allocs_B, _ = modelB(batch)

        allocs_A = normalize_allocs(allocs_A)
        allocs_B = normalize_allocs(allocs_B)

        dist = torch.cdist(allocs_A, allocs_B).view(-1)
        total = total + dist.sum().item()

    print("Allocation Distance: {}".format(total / args.test_num_examples))

