import os
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import datasets as ds
from datasets import Dataloader
from network import PreferenceNet
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

parser.add_argument('--preference_type', default='entropy')
parser.add_argument('--preference_thresh', type=float, default=0.5)

def classificationAccuracy(model, device, validationData):
    correct = 0
    total = 0

    with torch.no_grad():
        model.eval()
        for data in validationData:
            allocs, label = data
            allocs = allocs.to(device)
            label = label.to(device)

            pred = model(allocs)
            pred = pred > 0.5

            correct = correct + torch.sum(pred == label)
            total = total + allocs.shape[0]

    return correct / float(total)

def preference(allocs, type="entropy", thresh=0.68):
    if type == "entropy":
        allocs = allocs.clamp_min(1e-8)
        norm_allocs = allocs / allocs.sum(dim=-1).unsqueeze(-1)
        
        entropy = -1.0 * norm_allocs * torch.log(norm_allocs)
        entropy_alloc = entropy.sum(dim=-1).sum(dim=-1)

        labels = entropy_alloc > thresh    
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
    
    train_data, train_labels = ds.generate_random_allocations(args.num_examples, args.n_agents, args.n_items, args.unit, preference)
    train_data = train_data.to(DEVICE)
    train_labels = train_labels.to(DEVICE)
    train_loader = Dataloader(train_data, train_labels, batch_size=args.batch_size, shuffle=True)
    
    test_data, test_labels = ds.generate_random_allocations(args.test_num_examples, args.n_agents, args.n_items, args.unit, preference)
    test_data = test_data.to(DEVICE)
    test_labels = test_labels.to(DEVICE)
    test_loader = Dataloader(test_data, test_labels, batch_size=args.test_batch_size, shuffle=True)

    model.to(DEVICE)
    best_accuracy = 0
    for STEP in range(args.num_epochs):
        epochLoss = 0
        model.train()

        for batchCount, data in enumerate(train_loader, 1):
            allocs, label = data
            allocs = allocs.to(DEVICE)
            label = label.to(DEVICE)
            
            pred = model(allocs)

            optimizer.zero_grad()
            Loss = BCE(pred, label)
            epochLoss = epochLoss + Loss.item()

            Loss.backward()
            optimizer.step()

        print("Epoch " + str(STEP + 1) + " Training Loss: " + str(epochLoss/len(train_loader)) + " | Learning Rate: " + str(optimizer.param_groups[0]['lr']))
        accuracy = classificationAccuracy(model, DEVICE, test_loader)
        print("Classification Accuracy: {}".format(accuracy) )

        if accuracy > best_accuracy:
            modelState = {
                "experimentName": args.name,
                "State_Dictionary": model.state_dict(),
                }

            torch.save(modelState, "result/{}/{}.pth".format(args.name, args.preference_type))
            best_accuracy = accuracy

    print("Best Accuracy: {}".format(best_accuracy))