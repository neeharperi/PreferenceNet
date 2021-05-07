import os
from argparse import ArgumentParser
import torch
import numpy as np

from regretnet import datasets as ds
from regretnet.regretnet import RegretNet, RegretNetUnitDemand
from regretnet.datasets import Dataloader
import pdb


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = ArgumentParser()

parser.add_argument('--anchor-model-path', required=True)
parser.add_argument('--positive-model-path', required=True)
parser.add_argument('--negative-model-path', required=True)

parser.add_argument('--random-seed', type=int, default=0)
parser.add_argument('--test-num-examples', type=int, default=5000)
parser.add_argument('--test-batch-size', type=int, default=512)

args = parser.parse_args()
torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)

#########################################################
modelAnchor_ckpt = torch.load(args.anchor_model_path)
if "pv" in modelAnchor_ckpt['name']:
    modelAnchor = RegretNetUnitDemand(**(modelAnchor_ckpt['arch']))
else:
    modelAnchor = RegretNet(**(modelAnchor_ckpt['arch']))

state_dict = modelAnchor_ckpt['state_dict']
modelAnchor.load_state_dict(state_dict)

modelAnchor.to(DEVICE)
modelAnchor.eval()

#########################################################
modelPositive_ckpt = torch.load(args.positive_model_path)
if "pv" in modelPositive_ckpt['name']:
    modelPositive = RegretNetUnitDemand(**(modelPositive_ckpt['arch']))
else:
    modelPositive = RegretNet(**(modelPositive_ckpt['arch']))

state_dict = modelPositive_ckpt['state_dict']
modelPositive.load_state_dict(state_dict)

modelPositive.to(DEVICE)
modelPositive.eval()

#########################################################
modelNegative_ckpt = torch.load(args.negative_model_path)
if "pv" in modelNegative_ckpt['name']:
    modelNegative = RegretNetUnitDemand(**(modelNegative_ckpt['arch']))
else:
    modelNegative = RegretNet(**(modelNegative_ckpt['arch']))

state_dict = modelNegative_ckpt['state_dict']
modelNegative.load_state_dict(state_dict)

modelNegative.to(DEVICE)
modelNegative.eval()
#########################################################

# Valuation range setup
item_ranges = ds.preset_valuation_range(modelAnchor_ckpt['arch']['n_agents'], modelAnchor_ckpt['arch']['n_items'])
clamp_op = ds.get_clamp_op(item_ranges)

test_data = ds.generate_dataset_nxk(modelAnchor_ckpt['arch']['n_agents'], modelAnchor_ckpt['arch']['n_items'], args.test_num_examples, item_ranges).to(DEVICE)
test_loader = Dataloader(test_data, batch_size=args.test_batch_size, shuffle=True)

correct = 0
total = 0

for i, batch in enumerate(test_loader):
    batch = batch.to(DEVICE)
    allocsAnchor, _ = modelAnchor(batch)
    allocsPositive, _ = modelPositive(batch)
    allocsNegative, _ = modelNegative(batch)

    AP_dist = torch.cdist(allocsAnchor, allocsPositive)
    AN_dist = torch.cdist(allocsAnchor, allocsNegative)

    correct = correct + torch.sum(AP_dist < AN_dist)
    total = total + batch.shape[0]

acc = correct / float(total)
print("Preference Contrastive Accuracy: {}".format(acc))