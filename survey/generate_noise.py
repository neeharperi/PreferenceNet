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
    prompt = "[{val_id}] After Company A and Company B submitted their bids for a {ad_type}, Company A’s ad was shown to {X}% DEMOGRAPHIC1 and {Y}% DEMOGRAPHIC2. Company B’s ad was shown to {W}% DEMOGRAPHIC1 and {Z}% DEMOGRAPHIC2. Considering both companies, according to the given definition of fairness, is this fair?”"
    ad_types = ["toy ad", "phone ad", "internet provider ad", "job posting", 
                "newspaper ad", "magazine ad", "radio ad", "television ad", "billboard ad"]

    questions = open("Survey/{}x{}_survey_noise.txt".format(args.n_agents, args.n_items), "w")
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
        ad_type = random.choice(ad_types)
        question = prompt.format(val_id=val, ad_type=ad_type, X=X, Y=Y, W=W, Z=Z)
        questions.write(question + "\n")

    print("Number of Questions: {}".format(len(params)))

elif args.n_agents == 1 and args.n_items == 2:
    prompt = "[{val_id}] After Company A submitted their bids for the {ad_type}, Company A’s ad was shown to {X}% DEMOGRAPHIC1 and {Y}% DEMOGRAPHIC2. Considering the given definition of fairness, is this fair?”"
    ad_types = ["toy ad", "phone ad", "internet provider ad", "job posting", 
                "newspaper ad", "magazine ad", "radio ad", "televison ad", "billboard ad"]

    questions = open("Survey/{}x{}_survey_noise.txt".format(args.n_agents, args.n_items), "w")
    params = set()

    for alloc, val in zip(rounded_allocs, rounded_vals):
        val = round(val.item(), 2)

        X = alloc[0][0].item()
        Y = alloc[0][1].item()


        param = (X,Y)

        if param in params:
            continue

        params.add(param)
        ad_type = random.choice(ad_types)
        question = prompt.format(val_id=val, ad_type=ad_type, X=X, Y=Y)
        questions.write(question + "\n")

    print("Number of Questions: {}".format(len(params)))
