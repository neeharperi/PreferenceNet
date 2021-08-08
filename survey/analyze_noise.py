import os
import datasets as ds
import torch
import argparse
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
from seaborn_qqplot import pplot
import json

import re
import pdb

import matplotlib as mpl
mpl.font_manager._rebuild()
mpl.rc('font', family='Times New Roman')

parser = argparse.ArgumentParser()

parser.add_argument('--result-file', required=True)
parser.add_argument('--preference', default=[], nargs='+', required=True)
parser.add_argument('--dataset', nargs='+', default=[], required=True)
parser.add_argument('--multiplierA', type=int, default=1)
parser.add_argument('--multiplierB', type=int, default=1)
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

json.dump(TVF, open("Survey_Data/preference_noise.json", "w"))

noise_dist = {"TVF Score": [], "Noise Ratio" : [], "Majority" : []}
for val in TVF.keys():

    noise = min(TVF[val]["Yes"], TVF[val]["No"])
    total = TVF[val]["Yes"] + TVF[val]["No"]

    percent =  noise / float(total)

    noise_dist["TVF Score"].append(val)
    noise_dist["Noise Ratio"].append(percent)
    noise_dist["Majority"].append("Positive Example" if TVF[val]["Yes"] > TVF[val]["No"] else "Negative Example")

noise_dist_real = pd.DataFrame.from_dict(noise_dist)
plt.xlabel('xlabel', fontsize=16)
plt.ylabel('ylabel', fontsize=16)

ax = sns.barplot(x="TVF Score", y="Noise Ratio", data=noise_dist_real, hue="Majority", dodge=False, palette=sns.color_palette("Set2"))
plt.savefig("Results/preference_noise_real.png")
plt.clf()

def label_noise(valuation, threshold=None, dist="gaussian"):
    def z_score(x, mu, sigma):
        return abs(x - mu) / float(sigma)

    def label_flip(pct):
        return np.random.rand() < pct

    probability = st.norm.sf([z_score(val, threshold, valuation.std()) for val in valuation])

    return probability

def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = st.linregress(x, y)
    return r_value, r_value**2

thresh = 0.7
val = np.array([0.2, 0.4, 0.6, 0.7, 0.8, 1.0, 1.2, 1.4, 1.6])
pr = [p for p in label_noise(val, thresh)]

noise_dist = {"TVF Score": [], "Noise Ratio" : [], "Majority" : []}
for v, p in zip(val, pr):

    noise_dist["TVF Score"].append(v)
    noise_dist["Noise Ratio"].append(max(min((1.05*p), 0.5), 0.15))
    #noise_dist["Noise Ratio"].append(p)
    noise_dist["Majority"].append("Positive Example" if v < thresh else "Negative Example")

noise_dist_probit = pd.DataFrame.from_dict(noise_dist)
plt.xlabel('xlabel', fontsize=16)
plt.ylabel('ylabel', fontsize=16)

ax = sns.barplot(x="TVF Score", y="Noise Ratio", data=noise_dist_probit, hue="Majority", dodge=False, palette=sns.color_palette("Set2"))

plt.savefig("Results/preference_noise_probit.png")
plt.clf()

qq = {"Survey TVF Score" : [],
      "Probit TVF Score" : []}

real = noise_dist_real["Noise Ratio"].to_list()
real_tvf = noise_dist_real["TVF Score"].to_list()

real = [x for _, x in sorted(zip(real_tvf, real))]
real_total = sum(real)
real = [x / real_total for x in real]

probit = noise_dist_probit["Noise Ratio"].to_list()
probit_tvf = noise_dist_probit["TVF Score"].to_list()

probit = [x for _, x in sorted(zip(probit_tvf, probit))]
probit_total = sum(probit)
probit = [x / probit_total for x in probit]

tvf = sorted(probit_tvf)

for r, p, t in zip(real, probit, tvf):
    qq["Survey TVF Score"] = qq["Survey TVF Score"] + int(100 * r) * [t]
    qq["Probit TVF Score"] = qq["Probit TVF Score"] + int(100 * p) * [t]

if len(qq["Survey TVF Score"]) < 100:
    qq["Survey TVF Score"] = qq["Survey TVF Score"] + (100 - len(qq["Survey TVF Score"])) * [thresh]

if len(qq["Probit TVF Score"]) < 100:
    qq["Probit TVF Score"] = qq["Probit TVF Score"] + (100 - len(qq["Probit TVF Score"])) * [thresh]

qq["Survey TVF Score"] = sorted(qq["Survey TVF Score"])
qq["Probit TVF Score"] = sorted(qq["Probit TVF Score"])

qq = pd.DataFrame.from_dict(qq)

r_val, r2_val = rsquared(qq["Survey TVF Score"],qq["Probit TVF Score"])
print("R^2: {}".format(r2_val))
pplot(qq, x="Survey TVF Score", y="Probit TVF Score", kind='qq', height=4, aspect=1.5, display_kws={"identity":True})

#plt.text(0.1, 0.95,'$R^2$: {}'.format(round(r2_val,3)),
#     horizontalalignment='center',
#     verticalalignment='center',
#     transform = ax.transAxes)

plt.savefig("Results/preference_noise_qq.png")
plt.clf()
