import torch
from tqdm import tqdm as tqdm

import torch.nn.init

from regretnet.utils import calc_agent_util, optimize_misreports
from regretnet import datasets as ds
from regretnet.regretnet import RegretNet, RegretNetUnitDemand
from preference.network import PreferenceNet
import torch
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm as tqdm

from preference import datasets as pds
from preference import preference
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
        norm_allocs = allocs / allocs.sum(dim=-2).unsqueeze(-2)

        valuation =  norm_allocs.min(-1)[0].min(-1)[0]
        optimize = "max"

    elif type == "human":
        if args.preference_file is None:
            assert False, "Preference File Required"
        
        file = torch.load(args.preference_file)
        gt_allocs = torch.stack(list(file.keys()))
        gt_labels = torch.tensor([(file[key]["Yes"] / float(1e-8 + file[key]["Yes"] + file[key]["No"])) > args.preference_threshold for key in file.keys()])
        
        norm_allocs = allocs / allocs.sum(dim=-1).unsqueeze(-1)
        rounded_allocs = 5 * torch.round(20 * norm_allocs)

        valuation = []    
        for i, alloc in tqdm(enumerate(rounded_allocs), total = rounded_allocs.shape[0]):
            idx = torch.argmin(torch.cdist(gt_allocs, alloc).sum(dim=-1).sum(dim=-1)).item()

            valuation.append(gt_labels[idx])

        valuation = torch.tensor(valuation)
        optimize = None
        
    else:
        assert False, "Valuation type {} not supported".format(type)

    return valuation, optimize

def label_assignment(valuation, type, optimize, args):
    if type == "threshold":
        thresh = np.quantile(valuation, args.preference_threshold)
        labels = valuation > thresh if optimize == "max" else valuation < thresh 
        
        tnsr = torch.tensor([torch.tensor(int(i)) for i in labels]).float()

    elif type == "ranking":
        thresh = np.quantile(valuation, args.preference_threshold) if optimize == "max" else np.quantile(valuation, 1 - args.preference_threshold)
        cnt = torch.zeros_like(valuation)
        
        for _ in range(args.preference_ranking_pairwise_samples):
            idx = torch.randperm(len(valuation))

            if optimize == "max":
                cnt = cnt + (valuation > valuation[idx])
            elif optimize == "min":
                cnt = cnt + (valuation < valuation[idx])
            else:
                assert False, "Optimize type {} is not supported".format(optimize)

        labels = cnt > (args.preference_threshold * args.preference_ranking_pairwise_samples)
        
        tnsr = torch.tensor([torch.tensor(int(i)) for i in labels]).float()
    
    elif type == "bandpass":
        pass_band = []

        for i in range(len(args.preference_passband)):
            if i % 2 == 0:
                if optimize == "max":
                    thresh_low = np.quantile(valuation, float(args.preference_passband[i]))
                    thresh_high = np.quantile(valuation, float(args.preference_passband[i + 1]))
                elif optimize == "min":
                    thresh_high = np.quantile(valuation, 1 - float(args.preference_passband[i]))
                    thresh_low = np.quantile(valuation, 1 - float(args.preference_passband[i + 1]))
                else:
                    assert False, "Optimize type {} is not supported".format(optimize)

                pass_band.append((thresh_low, thresh_high))

        labels_band = []
        for thresh_low, thresh_high in pass_band:
            label = torch.tensor([(thresh_low < val < thresh_high).item() for val in valuation])
            labels_band.append(label)

        labels = torch.sum(torch.stack(labels_band), dim=0).bool()

        tnsr = torch.tensor([torch.tensor(int(i)) for i in labels]).float()

    elif type == "quota":
        thresh = args.preference_quota / args.n_agents
        labels = valuation > thresh
        
        tnsr = torch.tensor([torch.tensor(int(i)) for i in labels]).float()

    elif type == "preference":
        tnsr = torch.tensor([torch.tensor(int(i)) for i in valuation]).float()

    else:
        assert False, "Assignment type {} not supported".format(type)

    return tnsr

def label_preference(random_bids, allocs, actual_payments, type, args):
    valuation_fn, assignment_fn = type.split("_")
    valuation, optimize = label_valuation(random_bids, allocs, actual_payments, valuation_fn, args)
    label = label_assignment(valuation, assignment_fn, optimize, args)

    return label

def classificationAccuracy(model, validationData):
    correct = 0
    total = 0
    
    with torch.no_grad():
        model.eval()

        for data in validationData:
            bids, allocs, payments, label = data
            bids, allocs, payments, label = bids.to(DEVICE), allocs.to(DEVICE), payments.to(DEVICE), label.to(DEVICE)

            pred = model(bids, allocs, payments)
            pred = pred > 0.5

            correct = correct + torch.sum(pred == label)
            total = total + allocs.shape[0]

    return correct / float(total)

def train_preference(model, train_loader, test_loader, args):
    BCE = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.model_lr, betas=(0.5, 0.999), weight_decay=0.005)

    for _ in range(args.preference_num_epochs):
        epochLoss = 0
        model.train()
        for _, data in enumerate(train_loader, 1):
            bids, allocs, payments, label = data
            bids, allocs, payments, label = bids.to(DEVICE), allocs.to(DEVICE), payments.to(DEVICE), label.to(DEVICE)
        
            pred = model(bids, allocs, payments)

            optimizer.zero_grad()
            Loss = BCE(pred, label)
            epochLoss = epochLoss + Loss.item()

            Loss.backward()
            optimizer.step()

    accuracy = classificationAccuracy(model, test_loader)
    print("Classification Accuracy: {}".format(accuracy) )

    return model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', required=True)

parser.add_argument('--random-seed', type=int, default=0)
parser.add_argument('--test-num-examples', type=int, default=100000)
parser.add_argument('--test-batch-size', type=int, default=2048)
parser.add_argument('--misreport-lr', type=float, default=1e-1)
parser.add_argument('--misreport-iter', type=int, default=25)
parser.add_argument('--test-misreport-iter', type=int, default=1000)
parser.add_argument('--n-agents', type=int, default=1)
parser.add_argument('--n-items', type=int, default=2)
# Preference
parser.add_argument('--preference-num-examples', type=int, default=60000)
parser.add_argument('--preference-test-num-examples', type=int, default=20000)
parser.add_argument('--batch-size', type=int, default=2048)
parser.add_argument('--model-lr', type=float, default=1e-3)

parser.add_argument('--preference-num-epochs', type=int, default=50)

parser.add_argument('--preference', default=[], nargs='+', required=True)
parser.add_argument('--preference-file')
parser.add_argument('--preference-threshold', type=float, default=0.75)
parser.add_argument('--preference-passband', default=[], nargs='+')
parser.add_argument('--preference-quota', type=float, default=0.8)
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

args = parser.parse_args()
torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)

ds.dataset_override(args)
model_ckpt = torch.load(args.model_path)

assert model_ckpt["arch"]["n_agents"] == args.n_agents and  model_ckpt["arch"]["n_items"] == args.n_items, "model-ckpt does not match n_agents and n_items in args" 
item_ranges = ds.preset_valuation_range(args.n_agents, args.n_items, args.dataset)
clamp_op = ds.get_clamp_op(item_ranges)
model_ckpt['arch']['clamp_op'] = clamp_op

if "pv" in model_ckpt['name']:
    model = RegretNetUnitDemand(**(model_ckpt['arch']))
else:
    model = RegretNet(**(model_ckpt['arch']))

state_dict = model_ckpt['state_dict']
model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()

preference_type = []
mixed_preference_weight = 0
for i in range(len(args.preference)):
    if i % 2 == 0:
        preference_type.append((args.preference[i], float(args.preference[i+1])))
        mixed_preference_weight = mixed_preference_weight + float(args.preference[i+1])

assert mixed_preference_weight == 1, "Preference weights don't sum to 1."

accuracy = 0
for type, weight in preference_type:
    preference_net = PreferenceNet(args.n_agents, args.n_items, args.hidden_layer_size).to(DEVICE)

    train_bids, train_allocs, train_payments, train_labels = pds.generate_random_allocations_payments(int(args.preference_num_examples), args.n_agents, args.n_items, args.unit, item_ranges, args, type, label_preference)
    test_bids, test_allocs, test_payments, test_labels = pds.generate_random_allocations_payments(int(args.preference_test_num_examples), args.n_agents, args.n_items, args.unit, item_ranges, args, type, label_preference)

    preference_train_loader = pds.Dataloader((train_bids).to(DEVICE), (train_allocs).to(DEVICE), (train_payments).to(DEVICE), (train_labels).to(DEVICE), batch_size=args.batch_size, shuffle=True, balance=True, args=args)
    preference_test_loader = pds.Dataloader((test_bids).to(DEVICE), (test_allocs).to(DEVICE), (test_payments).to(DEVICE), (test_labels).to(DEVICE), batch_size=args.test_batch_size, shuffle=True, balance=False, args=args)

    preference_net = train_preference(preference_net, preference_train_loader, preference_test_loader, args)
    preference_net.eval()

    correct = 0
    total = 0
    test_data = ds.generate_dataset_nxk(args.n_agents, args.n_items, args.test_num_examples, item_ranges).to(DEVICE)
    test_loader = ds.Dataloader(test_data, batch_size=args.test_batch_size, shuffle=True)

    for i, batch in enumerate(test_loader):
        batch = batch.to(DEVICE)

        allocs, payments = model(batch)
        res = preference_net(batch, allocs, payments)         
        correct = correct + torch.sum(res > 0.5).item()
        total = total + res.shape[0]
    
    acc = correct/float(total)
    accuracy = accuracy + (weight * acc)
    print("{} Preference Accuracy: {}".format(type, acc))

print("Preference Accuracy: {}".format(accuracy))
