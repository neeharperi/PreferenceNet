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

torch.save(Data, open("Data/survey_preference.txt", "wb"))

prompt = "Case 1: After Company A and Company B submitted their bids for the {ad_type}, Company A’s ad was shown to {A}% DEMOGRAPHIC1 and {B}% DEMOGRAPHIC2. Company B’s ad was shown to {C}% DEMOGRAPHIC1 and {D}% DEMOGRAPHIC2. Case 2: after Company C and Company D submitted their bids for the {ad_type}, Company C’s ad was shown to {X}% DEMOGRAPHIC1 and {Y}% DEMOGRAPHIC2. Company D’s ad was shown to {W}% DEMOGRAPHIC1 and {Z}% DEMOGRAPHIC2. Which is more fair, Case 1 or Case 2?”"
ad_types = ["toy ad", "phone ad", "internet provider ad", "job posting", 
            "newspaper ad", "magazine ad", "radio ad", "televison ad", "billboard ad"]

questions = open("Survey/survey_preference.txt", "w")
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

    if param in params:
        continue

    params.add(param)
    ad_type = random.choice(ad_types)
    question = prompt.format(ad_type=ad_type, A=A, B=B, C=C, D=D, X=X, Y=Y, W=W, Z=Z)
    questions.write(question + "\n")

print("Number of Questions: {}".format(len(params)))
