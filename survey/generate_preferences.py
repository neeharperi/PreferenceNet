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
parser.add_argument('--unit', action='store_true')  # not saved in arch but w/e
args = parser.parse_args()

ds.dataset_override(args)
item_ranges = ds.preset_valuation_range(args.n_agents, args.n_items)
clamp_op = ds.get_clamp_op(item_ranges)

random_bids, allocs, actual_payments = ds.generate_random_allocations_payments(args.num_examples, args.n_agents, args.n_items, True, item_ranges, args)
rounded_allocs = 5 * torch.round(20 * allocs)
allocs1 = rounded_allocs[torch.randperm(rounded_allocs.shape[0])]
allocs2 = rounded_allocs[torch.randperm(rounded_allocs.shape[0])]

Data = {"Allocations" : rounded_allocs}

torch.save(Data, open("Data/{}x{}_survey_preference.pth".format(args.n_agents, args.n_items), "wb"))

if args.n_agents == 2 and args.n_items == 2:
    prompt = "<table><tr><td>Case 1: After Company A and Company B submitted their bids for a {ad_type},</td><td> Company A’s ad was shown to {A}% DEMOGRAPHIC1 and {B}% DEMOGRAPHIC2.</td></tr><tr><td></td><td> Company B’s ad was shown to {C}% DEMOGRAPHIC1 and {D}% DEMOGRAPHIC2.</td></tr><tr><td>Case 2: After Company C and Company D submitted their bids for a {ad_type},</td><td> Company C’s ad was shown to {X}% DEMOGRAPHIC1 and {Y}% DEMOGRAPHIC2.</td></tr><tr><td></td><td> Company D’s ad was shown to {W}% DEMOGRAPHIC1 and {Z}% DEMOGRAPHIC2.</td></tr></table><br>Which is more fair, Case 1, Case 2, or other? If other, please elaborate in the provided text field."
    ad_types = ["toy ad", "phone ad", "internet provider ad", "job posting", 
                "newspaper ad", "magazine ad", "radio ad", "televison ad", "billboard ad"]

    questions = open("Survey/{}x{}_survey_preference.txt".format(args.n_agents, args.n_items), "w")
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
        
        if A == X and B == Y and C == W and D == Z:
            continue

        if param in params:
            continue

        params.add(param)
        ad_type = random.choice(ad_types)
        question = prompt.format(ad_type=ad_type, A=A, B=B, C=C, D=D, X=X, Y=Y, W=W, Z=Z)
        questions.write(question + "\n")

    print("Number of Questions: {}".format(len(params)))

elif args.n_agents == 1 and args.n_items == 2:

    prompt = "Case 1: After Company A submitted their bids for the {ad_type}, Company A’s ad was shown to {A}% DEMOGRAPHIC1 and {B}% DEMOGRAPHIC2.<br>Case 2: After Company B submitted their bids for the {ad_type}, Company B’s ad was shown to {X}% DEMOGRAPHIC1 and {Y}% DEMOGRAPHIC2.<br>Which is more fair, Case 1, Case 2, or other? If other, please elaborate in the provided text field."
    ad_types = ["toy ad", "phone ad", "internet provider ad", "job posting", 
                "newspaper ad", "magazine ad", "radio ad", "televison ad", "billboard ad"]

    questions = open("Survey/{}x{}_survey_preference.txt".format(args.n_agents, args.n_items), "w")
    params = set()
    for alloc1, alloc2 in zip(allocs1, allocs2):
        A = alloc1[0][0].item()
        B = alloc1[0][1].item()
        
        X = alloc2[0][0].item()
        Y = alloc2[0][1].item()

        param = (A,B,X,Y)

        if A == X and B == Y:
            continue

        if param in params:
            continue

        params.add(param)
        ad_type = random.choice(ad_types)
        question = prompt.format(ad_type=ad_type, A=A, B=B, X=X, Y=Y)
        questions.write(question + "\n")

    print("Number of Questions: {}".format(len(params)))