import torch
from tqdm import tqdm as tqdm

import torch.nn.init
from regretnet.utils import calc_agent_util
from preference import datasets as pds
from regretnet.regretnet import RegretNet, RegretNetUnitDemand

import argparse 
from itertools import tee

import numpy as np
import pdb

def window(iterable, size):
    iters = tee(iterable, size)
    for i in range(1, size):
        for each in iters[i:]:
            next(each, None)
    return zip(*iters)

def label_valuation(random_bids, allocs, actual_payments, type, args):
    if type == "entropy":
        assert args.n_items > 1, "Entropy regularization requires num_items > 1"
        
        allocs = allocs.clamp_min(1e-8)
        norm_allocs = allocs / allocs.sum(dim=-1).unsqueeze(-1)
        
        entropy = -1.0 * norm_allocs * torch.log(norm_allocs)
        valuation = entropy.sum(dim=-1).sum(dim=-1)
        optimize = "max"

    elif type == "tvf":
        d = args.tvf_distance
        C = [[i for i in range(args.n_agents)]]
        D = (torch.ones(1, args.n_items, args.n_items) * d)
        L, n, m = allocs.shape
        unfairness = torch.zeros(L, m)
        for i, C_i in enumerate(C):
            for u in range(m):
                for v in range(m):
                    subset_allocs_diff = (allocs[:, C_i, u] - allocs[:, C_i, v]).abs()
                    D2 = 1 - (1 - D) if n == 1 else 2 - (2 - D)
                    unfairness[:, u] += (subset_allocs_diff.sum(dim=1) - D2[i, u, v]).clamp_min(0)
        
        valuation = unfairness.sum(dim=-1)
        optimize = "min"

    elif type == "utility":
        valuation = calc_agent_util(random_bids, allocs, actual_payments).view(-1)
        optimize = "min"

    elif type == "quota":
        assert args.n_agents > 1, "Quota regularization requires num_agents > 1"

        allocs = allocs.clamp_min(1e-8)
        norm_allocs = allocs / allocs.sum(dim=-2).unsqueeze(-1)
        valuation = torch.tensor([norm_alloc.min().item() for norm_alloc in norm_allocs])
        optimize = "max"

    else:
        assert False, "Valuation type {} not supported".format(type)

    return valuation, optimize

def label_assignment(valuation, type, optimize, args):
    if type == "threshold":
        thresh = args.preference_threshold
        labels = valuation > thresh if optimize == "max" else valuation < thresh 
        tnsr = torch.tensor([torch.tensor(int(i)) for i in labels]).float()

    else:
        assert False, "Assignment type {} not supported".format(type)

    return tnsr

def label_preference(random_bids, allocs, actual_payments, type, args):
    valuation_fn, assignment_fn = type.split("_")
    valuation, optimize = label_valuation(random_bids, allocs, actual_payments, valuation_fn, args)
    label = label_assignment(valuation, assignment_fn, optimize, args)

    return label


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', required=True)

parser.add_argument('--random-seed', type=int, default=0)
parser.add_argument('--test-num-examples', type=int, default=1000)
parser.add_argument('--test-batch-size', type=int, default=2048)
# Preference
parser.add_argument('--preference', default=[], nargs='+', required=True)
parser.add_argument('--preference-threshold', type=float, default=0.75)
parser.add_argument('--preference-passband', default=[], nargs='+')
parser.add_argument('--preference-quota', type=float, default=0.4)
parser.add_argument('--tvf-distance', type=float, default=0.0)

parser.add_argument('--dataset', nargs='+', default=[], required=True)
# architectural arguments
parser.add_argument('--hidden-layer-size', type=int, default=100)
parser.add_argument('--n-hidden-layers', type=int, default=2)
parser.add_argument('--separate', action='store_true')
parser.add_argument('--name', default='testing_name')
parser.add_argument('--unit', action='store_true')  # not saved in arch but w/e

parser.add_argument('--preference-synthetic-pct', type=float, default=0.0)
parser.add_argument('--preference-label-noise', type=float, default=0.0)

parser.add_argument('--num_pts', type=int, default=101)

args = parser.parse_args()
torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)

pds.dataset_override(args)

model_ckpt = torch.load(args.model_path)
if "pv" in model_ckpt['name']:
    model = RegretNetUnitDemand(**(model_ckpt['arch']))
else:
    model = RegretNet(**(model_ckpt['arch']))

state_dict = model_ckpt['state_dict']
model.load_state_dict(state_dict)


item_ranges = pds.preset_valuation_range(args.n_agents, args.n_items)
clamp_op = pds.get_clamp_op(item_ranges)

preference_type = []
mixed_preference_weight = 0
for i in range(len(args.preference)):
    if i % 2 == 0:
        preference_type.append((args.preference[i], float(args.preference[i+1])))
        mixed_preference_weight = mixed_preference_weight + float(args.preference[i+1])

assert mixed_preference_weight == 1, "Preference weights don't sum to 1."

allocs = pds.generate_random_allocations(args.test_num_examples, args.n_agents, args.n_items, args.unit, args, preference=None)
val_type = args.preference[0].split("_")[0]
vals, _ = label_valuation(None, allocs, None, val_type, args)
thresh = []

for pct in np.linspace(0, 1, args.num_pts, endpoint=False):
    thresh.append(np.quantile(vals, pct))

classification = []
for th in tqdm(thresh):
    args.preference_threshold = th
    type = val_type + "_threshold"
    bids = pds.generate_random_allocations_payments(args.test_num_examples, args.n_agents, args.n_items, args.unit, item_ranges, args, type, label_preference)
    test_loader = pds.Dataloader(bids.to(DEVICE), batch_size=args.test_batch_size, shuffle=True, balance=False, args=args)

    model.to(DEVICE)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = batch.to(DEVICE)
            allocs, payments = model(batch)

            res =  label_preference(batch.cpu(), allocs.cpu(), payments.cpu(), type, args)
            correct = correct + torch.sum(res).item()
            total = total + res.shape[0]
    
    acc = correct/float(total)
    print("Classification Accuracy: {} @ Threshold {}".format(acc, th))
    classification.append(acc)

AUC = 0
height = 1.0 / args.num_pts

for st, en in window(classification, 2):
    AUC = AUC + 0.5 * height * (st + en)


print("AUC: {}".format(AUC))