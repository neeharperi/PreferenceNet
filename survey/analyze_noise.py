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
result = result[result["Q5"] != "No"]
result = result[result["Q13"] != "Yes"]

TVF = {}

for key in result.keys():
    if not key.startswith("Q"):
        continue

    str = [s for s in result[key][0].split("\n") if "DEMOGRAPHIC1" in s and "DEMOGRAPHIC2" in s]
    
    if len(str) != 2:
        continue

    agent1 = re.findall('[0-9]+.0', str[0])
    agent2 = re.findall('[0-9]+.0', str[1])

    alloc = torch.tensor([[float(agent1[0]), float(agent1[1])], [float(agent2[0]), float(agent2[1])]]).unsqueeze(0)
    val_type = args.preference[0].split("_")[0]
    val = round((ds.label_valuation(None, alloc, None, val_type, args)[0] / 100).item(), 2)
    
    if val not in TVF.keys():
        TVF[val] = {"Yes" : 0, "No": 0}

    response = result[key][2:]
    TVF[val]["Yes"] = TVF[val]["Yes"] + sum(response == "Yes")
    TVF[val]["No"] = TVF[val]["No"] + sum(response == "No")

noise_dist = {"TVF": [], "Percent" : []}
for val in TVF.keys():

    noise = min(TVF[val]["Yes"], TVF[val]["No"])
    total = TVF[val]["Yes"] + TVF[val]["No"]

    percent =  noise / float(total)

    noise_dist["TVF"].append(val)
    noise_dist["Percent"].append(percent)

noise_dist = pd.DataFrame.from_dict(noise_dist)
ax = sns.barplot(x="TVF", y="Percent", data=noise_dist, color="lightblue")
plt.savefig("Results/preference_noise.pdf")


