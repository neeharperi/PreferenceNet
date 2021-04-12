import os
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import datasets as ds
from datasets import Dataloader
from network import PreferenceNet
from regretnet.regretnet import RegretNet, RegretNetUnitDemand

import json
import pdb 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = ArgumentParser()
parser.add_argument('--random-seed', type=int, default=0)
parser.add_argument('--num-examples', type=int, default=160000)
parser.add_argument('--test-num-examples', type=int, default=30000)
parser.add_argument('--n-agents', type=int, default=1)
parser.add_argument('--n-items', type=int, default=2)
parser.add_argument('--hidden-dim', type=int, default=100)
parser.add_argument('--num-epochs', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=2000)
parser.add_argument('--test-batch-size', type=int, default=512)
parser.add_argument('--model-lr', type=float, default=1e-3)

# parser.add_argument('--min-payment-ratio', type=float, default=0.)  # Price of fairness; used with delayed fairness
# dataset selection: specifies a configuration of agent/item/valuation
parser.add_argument('--dataset', nargs='+', default=[])
parser.add_argument('--resume', default="")
# architectural arguments
parser.add_argument('--name', default='testing_name')
parser.add_argument('--unit', action='store_true')  # not saved in arch but w/e

parser.add_argument('--preference_type', default='entropy_classification')
parser.add_argument('--preference_thresh', type=float, default=0.685)

parser.add_argument('--regretnet_ckpt', default='none')

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

def preference(random_bids, allocs, actual_payments, type="entropy_classification", thresh=0.685, samples=1000, pct=0.75):
    if type == "entropy_classification":
        allocs = allocs.clamp_min(1e-8)
        norm_allocs = allocs / allocs.sum(dim=-1).unsqueeze(-1)
        
        entropy = -1.0 * norm_allocs * torch.log(norm_allocs)
        entropy_alloc = entropy.sum(dim=-1).sum(dim=-1)
        labels = entropy_alloc > thresh    #0.685
        tnsr = torch.tensor([torch.tensor(int(i)) for i in labels]).float()

        return tnsr

    elif type == "entropy_ranking":
        allocs = allocs.clamp_min(1e-8)
        norm_allocs = allocs / allocs.sum(dim=-1).unsqueeze(-1)
        
        entropy = -1.0 * norm_allocs * torch.log(norm_allocs)
        entropy_alloc = entropy.sum(dim=-1).sum(dim=-1)
        entropy_cnt = torch.zeros_like(entropy_alloc)
        
        for i in range(samples):
            idx = torch.randperm(len(entropy_alloc))
            entropy_cnt = entropy_cnt + (entropy_alloc > entropy_alloc[idx])

        labels = entropy_cnt > (pct * samples)
        tnsr = torch.tensor([torch.tensor(int(i)) for i in labels]).float()

        return tnsr
    
    elif type == "tvf_classification":
        d = 0.0
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
        
        tvf_alloc = unfairness.sum(dim=-1)
        labels = tvf_alloc < thresh  #0.175
        tnsr = torch.tensor([torch.tensor(int(i)) for i in labels]).float()

        return tnsr

    elif type == "tvf_ranking":
        d = 0.0
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
        
        tvf_alloc = unfairness.sum(dim=-1)
        tvf_cnt = torch.zeros_like(tvf_alloc)
        
        for i in range(samples):
            idx = torch.randperm(len(tvf_alloc))
            tvf_cnt = tvf_cnt + (tvf_alloc < tvf_alloc[idx])

        labels = tvf_cnt > (pct * samples)
        tnsr = torch.tensor([torch.tensor(int(i)) for i in labels]).float()

        return tnsr

    assert False, "Invalid Preference Type"

if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # Replaces n_items, n_agents, name
    ds.dataset_override(args)

    # Valuation range setup
    item_ranges = ds.preset_valuation_range(args.n_agents, args.n_items, args.dataset)
    clamp_op = ds.get_clamp_op(item_ranges)


    if not os.path.exists("result/{}".format(args.name)):
        os.makedirs("result/{}".format(args.name))


    BCE = nn.BCELoss()
    model = PreferenceNet(args.n_agents, args.n_items, args.hidden_dim)

    optimizer = optim.Adam(model.parameters(), lr=args.model_lr, betas=(0.5, 0.999), weight_decay=0.005)

    if  torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        optimizer = optim.Adam(model.module.parameters(), lr=args.model_lr, betas=(0.5, 0.999), weight_decay=0.005)
    
    if args.regretnet_ckpt == "none":
        item_ranges = ds.preset_valuation_range(args.n_agents, args.n_items)
        clamp_op = ds.get_clamp_op(item_ranges)

        train_bids, train_allocs, train_payments, train_labels = ds.generate_random_allocations_payments(args.num_examples, args.n_agents, args.n_items, args.unit, item_ranges, args, preference)
        train_bids, train_allocs, train_payments, train_labels = train_bids.to(DEVICE), train_allocs.to(DEVICE), train_payments.to(DEVICE), train_labels.to(DEVICE)
        train_loader = Dataloader(train_bids, train_allocs, train_payments, train_labels, batch_size=args.batch_size, shuffle=True, args=args)
        
        test_bids, test_allocs, test_payments, test_labels = ds.generate_random_allocations_payments(args.test_num_examples, args.n_agents, args.n_items, args.unit, item_ranges, args, preference)
        test_bids, test_allocs, test_payments, test_labels = test_bids.to(DEVICE), test_allocs.to(DEVICE), test_payments.to(DEVICE), test_labels.to(DEVICE)
        test_loader = Dataloader(test_bids, test_allocs, test_payments, test_labels, batch_size=args.test_batch_size, shuffle=True, args=args)
    else:
        model_ckpt = torch.load(args.regretnet_ckpt)
        if "pv" in model_ckpt['name']:
            regretnet_model = RegretNetUnitDemand(**(model_ckpt['arch']))
        else:
            regretnet_model = RegretNet(**(model_ckpt['arch']))
        
        state_dict = model_ckpt['state_dict']
        regretnet_model.load_state_dict(state_dict)
        regretnet_model.to(DEVICE)
        regretnet_model.eval()
        
        item_ranges = ds.preset_valuation_range(model_ckpt['arch']['n_agents'], model_ckpt['arch']['n_items'])
        clamp_op = ds.get_clamp_op(item_ranges)

        train_bids, train_allocs, train_payments, train_labels = ds.generate_regretnet_allocations(args.num_examples, args.n_agents, args.n_items, args.unit, item_ranges, args, preference)
        train_bids, train_allocs, train_payments, train_labels = train_bids.to(DEVICE), train_allocs.to(DEVICE), train_payments.to(DEVICE), train_labels.to(DEVICE)

        train_loader = Dataloader(train_bids, train_allocs, train_payments, train_labels, batch_size=args.batch_size, shuffle=True, args=args)
        
        test_bids, test_allocs, test_payments, test_labels = ds.generate_regretnet_allocations(args.test_num_examples, args.n_agents, args.n_items, args.unit, item_ranges, args, preference)
        test_bids, test_allocs, test_payments, test_labels = test_bids.to(DEVICE), test_allocs.to(DEVICE), test_payments.to(DEVICE), test_labels.to(DEVICE)
        test_loader = Dataloader(test_bids, test_allocs, test_payments, test_labels, batch_size=args.test_batch_size, shuffle=True, args=args)
    
    model.to(DEVICE)
    best_accuracy = 0
    for STEP in range(args.num_epochs):
        epochLoss = 0
        model.train()
        for batchCount, data in enumerate(train_loader, 1):
            bids, allocs, payments, label = data
            bids, allocs, payments, label = bids.to(DEVICE), allocs.to(DEVICE), payments.to(DEVICE), label.to(DEVICE)
        
            pred = model(bids, allocs, payments)

            optimizer.zero_grad()
            Loss = BCE(pred, label)
            epochLoss = epochLoss + Loss.item()

     
            Loss.backward()
            optimizer.step()

        print("Epoch " + str(STEP + 1) + " Training Loss: " + str(epochLoss/len(train_loader)) + " | Learning Rate: " + str(optimizer.param_groups[0]['lr']))
        accuracy = classificationAccuracy(model, test_loader)
        print("Classification Accuracy: {}".format(accuracy) )

        if accuracy > best_accuracy:
            modelState = {
                "experimentName": args.name,
                "State_Dictionary": model.state_dict(),
                }

            torch.save(modelState, "result/{}/{}_{}.pth".format(args.name, "synthetic" if args.regretnet_ckpt == "none" else "regretnet" ,args.preference_type))
            best_accuracy = accuracy

    print("Best Accuracy: {}".format(best_accuracy))