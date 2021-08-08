import os
import datasets as ds
import torch
import argparse
import random
import pdb

parser = argparse.ArgumentParser()

parser.add_argument('--random-seed', type=int, default=0)
parser.add_argument('--num-examples', type=int, default=1000)
# Preference
parser.add_argument('--preference', default=[], nargs='+', required=True)
parser.add_argument('--preference-threshold', type=float, default=0.75)
parser.add_argument('--tvf-distance', type=float, default=0.0)
parser.add_argument('--tolerance', type=float, default=0.01)

parser.add_argument('--dataset', nargs='+', default=[], required=True)
parser.add_argument('--multiplierA', type=int, default=1)
parser.add_argument('--multiplierB', type=int, default=1)
parser.add_argument('--unit', action='store_true')  # not saved in arch but w/e
args = parser.parse_args()

ds.dataset_override(args)
item_ranges = ds.preset_valuation_range(args, args.n_agents, args.n_items, args.dataset)
clamp_op = ds.get_clamp_op(item_ranges)

random_bids, allocs, actual_payments = ds.generate_random_allocations_payments(args.num_examples, args.n_agents, args.n_items, True, item_ranges, args)
rounded_allocs = 5 * torch.round(20 * allocs)
allocs1 = rounded_allocs[torch.randperm(rounded_allocs.shape[0])]
allocs2 = rounded_allocs[torch.randperm(rounded_allocs.shape[0])]

Data = {"Allocations" : rounded_allocs}

torch.save(Data, open("Data/{}x{}_survey_preference.pth".format(args.n_agents, args.n_items), "wb"))

if args.n_agents == 2 and args.n_items == 2:
    prompt = '''<table></td><td> Case 1: Company A’s ad was shown to {A}% DEMOGRAPHIC1 and {B}% DEMOGRAPHIC2.<br> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Company B’s ad was shown to {C}% DEMOGRAPHIC1 and {D}% DEMOGRAPHIC2.</td></tr>
<div>
<table border="0" cellpadding="1" cellspacing="1" style="width:500px;">
    <tbody>
        <tr>
            <td>Company A<br />
            DEMOGRAPHIC1<br />
            <progress id="file" max="100" value="{A}"> {A}% </progress><br />
            {A}%</td>
            <td>Company A<br />
            DEMOGRAPHIC2<br />
            <progress id="file" max="100" value="{B}"> {B}% </progress><br />
            {B}%</td>
            <td>Company B<br />
            DEMOGRAPHIC1<br />
            <progress id="file" max="100" value="{C}"> {C}% </progress><br />
            {C}%</td>
            <td>Company B<br />
            DEMOGRAPHIC2<br />
            <progress id="file" max="100" value="{D}"> {D}% </progress><br />
            {D}%</td>
        </tr>
    </tbody>
</table>
</div>
<br>

<tr><td>Case 2: Company C’s ad was shown to {X}% DEMOGRAPHIC1 and {Y}% DEMOGRAPHIC2. <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Company D’s ad was shown to {W}% DEMOGRAPHIC1 and {Z}% DEMOGRAPHIC2.</td></tr></table>

<div>
<table border="0" cellpadding="1" cellspacing="1" style="width:500px;">
    <tbody>
        <tr>
            <td>Company C<br />
            DEMOGRAPHIC1<br />
            <progress id="file" max="100" value="{X}"> {X}% </progress><br />
            {X}%</td>
            <td>Company C<br />
            DEMOGRAPHIC2<br />
            <progress id="file" max="100" value="{Y}"> {Y}% </progress><br />
            {Y}%</td>
            <td>Company D<br />
            DEMOGRAPHIC1<br />
            <progress id="file" max="100" value="{W}"> {W}% </progress><br />
            {W}%</td>
            <td>Company D<br />
            DEMOGRAPHIC2<br />
            <progress id="file" max="100" value="{Z}"> {Z}% </progress><br />
            {Z}%</td>
        </tr>
    </tbody>
</table>
</div><br>
Which is more fair, Case 1 or Case 2?'''

    questions = open("Survey/{}x{}_survey_preference.txt".format(args.n_agents, args.n_items), "w")
    questions.write("[[AdvancedFormat]]\n")
    params = set()

    for alloc1, alloc2 in zip(allocs1, allocs2):
        A = alloc1[0][0].item()
        B = alloc1[0][1].item()
        C = alloc1[1][0].item()
        D = alloc1[1][1].item()
        
        X = alloc2[0][0].item()
        Y = alloc2[0][1].item()
        W = alloc2[1][0].item()
        Z = alloc2[1][1].item()

        param = (A,B,C,D,X,Y,W,Z)
        
        setA = set()
        setB = set()

        setA.add(A)
        setA.add(B)
        setA.add(C)
        setA.add(D)

        setB.add(X)
        setB.add(Y)
        setB.add(W)
        setB.add(Z)

        if setA == setB:
            continue

        if param in params:
            continue

        params.add(param)
        question = prompt.format(A=A, B=B, C=C, D=D, X=X, Y=Y, W=W, Z=Z)
        questions.write("[[Question:MC]]\n")
        questions.write(question + "\n\n")

        questions.write("[[Choices]]\n")
        questions.write("Case 1\n")
        questions.write("Case 2\n")

    print("Number of Questions: {}".format(len(params)))

elif args.n_agents == 1 and args.n_items == 2:

    prompt = '''<table></td><td> Case 1: Company A’s ad was shown to {A}% DEMOGRAPHIC1 and {B}% DEMOGRAPHIC2. </td></tr>
<div>
<table border="0" cellpadding="1" cellspacing="1" style="width:500px;">
    <tbody>
        <tr>
            <td>Company A<br />
            DEMOGRAPHIC1<br />
            <progress id="file" max="100" value="{A}"> {A}% </progress><br />
            {A}%</td>
            <td>Company A<br />
            DEMOGRAPHIC2<br />
            <progress id="file" max="100" value="{B}"> {B}% </progress><br />
            {B}%</td>
        </tr>
    </tbody>
</table>
</div>
<br>

<tr><td>Case 2: Company B’s ad was shown to {X}% DEMOGRAPHIC1 and {Y}% DEMOGRAPHIC2. </td></tr></table>

<div>
<table border="0" cellpadding="1" cellspacing="1" style="width:500px;">
    <tbody>
        <tr>
            <td>Company B<br />
            DEMOGRAPHIC1<br />
            <progress id="file" max="100" value="{X}"> {X}% </progress><br />
            {X}%</td>
            <td>Company B<br />
            DEMOGRAPHIC2<br />
            <progress id="file" max="100" value="{Y}"> {Y}% </progress><br />
            {Y}%</td>
        </tr>
    </tbody>
</table>
</div><br>
Which is more fair, Case 1 or Case 2?'''
    
    questions = open("Survey/{}x{}_survey_preference.txt".format(args.n_agents, args.n_items), "w")
    questions.write("[[AdvancedFormat]]\n")
    params = set()
    for alloc1, alloc2 in zip(allocs1, allocs2):
        A = alloc1[0][0].item()
        B = alloc1[0][1].item()
        
        X = alloc2[0][0].item()
        Y = alloc2[0][1].item()

        param = (A,B,X,Y)

        setA = set()
        setB = set()

        setA.add(A)
        setA.add(B)
        setB.add(X)
        setB.add(Y)

        if setA == setB:
            continue

        if param in params:
            continue

        params.add(param)
        question = prompt.format(A=A, B=B, X=X, Y=Y)
        questions.write("[[Question:MC]]\n")
        questions.write(question + "\n\n")
        questions.write("[[Choices]]\n")
        questions.write("Case 1\n")
        questions.write("Case 2\n")

    print("Number of Questions: {}".format(len(params)))

