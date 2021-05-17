import os
import datasets as ds
import torch
import argparse
import random
import pdb

parser = argparse.ArgumentParser()

parser.add_argument('--random-seed', type=int, default=0)
parser.add_argument('--num-examples', type=int, default=100000)
# Preference
parser.add_argument('--preference', default=[], nargs='+', required=True)
parser.add_argument('--preference-threshold', type=float, default=0.75)
parser.add_argument('--tvf-distance', type=float, default=0.0)
parser.add_argument('--tolerance', type=float, default=0.01)

parser.add_argument('--dataset', nargs='+', default=[], required=True)
parser.add_argument('--unit', action='store_true')  # not saved in arch but w/e
args = parser.parse_args()

ds.dataset_override(args)
item_ranges = ds.preset_valuation_range(args.n_agents, args.n_items)
clamp_op = ds.get_clamp_op(item_ranges)

random_bids, allocs, actual_payments, labels, valuation = ds.generate_random_allocations_payments(args.num_examples, args.n_agents, args.n_items, True, item_ranges, args, args.preference[0], ds.label_preference)
thresh = valuation[labels.bool()].max().item()
rounded_allocs = 5 * torch.round(20 * allocs)
val_type = args.preference[0].split("_")[0]
rounded_vals = ds.label_valuation(None, rounded_allocs, None, val_type, args)[0] / 100

Data = {"Allocations" : rounded_allocs,
        "Valuation" : rounded_vals}

torch.save(Data, open("Data/{}x{}_survey_noise.pth".format(args.n_agents, args.n_items), "wb"))

if args.n_agents == 2 and args.n_items == 2:
    prompt = '''Company A's ad was shown to {X}% DEMOGRAPHIC1 and {Y}% DEMOGRAPHIC2.
Considering the given definition of fairness, is this fair?
<div>
<table border="0" cellpadding="1" cellspacing="1" style="width:500px;">
    <tbody>
        <tr>
            <td>Company A<br />
            DEMOGRAPHIC1<br />
            <progress id="file" max="100" value="{X}"> {X}% </progress><br />
            {X}%</td>
            <td>Company A<br />
            DEMOGRAPHIC2<br />
            <progress id="file" max="100" value="{Y}"> {Y}% </progress><br />
            {Y}%</td>
        </tr>
    </tbody>
</table>
</div>
Company B's ad was shown to {W}% DEMOGRAPHIC1 and {Z}% DEMOGRAPHIC2.
<div>
<table border="0" cellpadding="1" cellspacing="1" style="width:500px;">
    <tbody>
        <tr>
            <td>Company B<br />
            DEMOGRAPHIC1<br />
            <progress id="file" max="100" value="{W}"> {W}% </progress><br />
            {W}%</td>
            <td>Company B<br />
            DEMOGRAPHIC2<br />
            <progress id="file" max="100" value="{Z}"> {Z}% </progress><br />
            {Z}%</td>
        </tr>
    </tbody>
</table>
</div>'''

    questions = open("Survey/{}x{}_survey_noise.txt".format(args.n_agents, args.n_items), "w")
    questions.write("[[AdvancedFormat]]\n")
    params = set()

    for alloc, val in zip(rounded_allocs, rounded_vals):
        val = round(val.item(), 2)

        X = alloc[0][0].item()
        Y = alloc[0][1].item()
        W = alloc[1][0].item()
        Z = alloc[1][1].item()

        param = (X,Y,W,Z)

        if param in params:
            continue

        params.add(param)
        question = prompt.format(X=X, Y=Y, W=W, Z=Z)
        questions.write("[[Question:MC]]\n")
        questions.write(question + "\n\n")
        questions.write("[[Choices]]\n")
        questions.write("Yes\n")
        questions.write("No\n\n")


    print("Number of Questions: {}".format(len(params)))

elif args.n_agents == 1 and args.n_items == 2:
    prompt = '''Company A's ad was shown to {X}% DEMOGRAPHIC1 and {Y}% DEMOGRAPHIC2.
Considering the given definition of fairness, is this fair?
<div>
<table border="0" cellpadding="1" cellspacing="1" style="width:500px;">
    <tbody>
        <tr>
            <td>Company A<br />
            DEMOGRAPHIC1<br />
            <progress id="file" max="100" value="{X}"> {X}% </progress><br />
            {X}%</td>
            <td>Company A<br />
            DEMOGRAPHIC2<br />
            <progress id="file" max="100" value="{Y}"> {Y}% </progress><br />
            {Y}%</td>
        </tr>
    </tbody>
</div>'''

    questions = open("Survey/{}x{}_survey_noise.txt".format(args.n_agents, args.n_items), "w")
    questions.write("[[AdvancedFormat]]\n")
    params = set()

    for alloc, val in zip(rounded_allocs, rounded_vals):
        val = round(val.item(), 2)

        X = alloc[0][0].item()
        Y = alloc[0][1].item()


        param = (X,Y)

        if param in params:
            continue

        params.add(param)
        question = prompt.format(X=X, Y=Y)
        questions.write("[[Question:MC]]\n")
        questions.write(question + "\n\n")
        questions.write("[[Choices]]\n")
        questions.write("Yes\n")
        questions.write("No\n\n")
    print("Number of Questions: {}".format(len(params)))

