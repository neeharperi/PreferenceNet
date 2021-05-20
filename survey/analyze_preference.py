import os
import datasets as ds
import torch
import argparse
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
import pdb

parser = argparse.ArgumentParser()

parser.add_argument('--result-file', required=True)
parser.add_argument('--preference', default=[], nargs='+', required=True)
parser.add_argument('--dataset', nargs='+', default=[], required=True)
parser.add_argument('--tvf-distance', type=float, default=0.0)

args = parser.parse_args()

ds.dataset_override(args)


result = pd.read_csv(args.result_file)
result = result[result["Q5"] != "Case 1"]
result = result[result["Q17"] != "Case 2"]

data = {}
for key in result.keys():
    if not key.startswith("Q"):
        continue
    
    str = [s for s in result[key][0].split("\n") if "DEMOGRAPHIC1" in s and "DEMOGRAPHIC2" in s]
    
    if len(str) != 4:
        continue

    agent1 = re.findall('[0-9]+.0', str[0])
    agent2 = re.findall('[0-9]+.0', str[1])

    agent3 = re.findall('[0-9]+.0', str[2])
    agent4 = re.findall('[0-9]+.0', str[3])

    alloc1 = torch.tensor([[float(agent1[0]), float(agent1[1])], [float(agent2[0]), float(agent2[1])]])
    alloc2 = torch.tensor([[float(agent3[0]), float(agent3[1])], [float(agent4[0]), float(agent4[1])]])

    if alloc1 not in data.keys():
        data[alloc1] = {"Yes" : 0, "No" : 0}

    if alloc2 not in data.keys():
        data[alloc2] = {"Yes" : 0, "No" : 0}

    case1 = sum(result[key] == "Case 1") 
    case2 = sum(result[key] == "Case 2") 

    data[alloc1]["Yes"] = data[alloc1]["Yes"] + case1 
    data[alloc1]["No"] = data[alloc1]["No"] + case2 
    data[alloc2]["Yes"] = data[alloc2]["Yes"] + case2 
    data[alloc2]["No"] = data[alloc2]["No"] + case1 

torch.save(data, "Results/preference_elicitation.pth")




