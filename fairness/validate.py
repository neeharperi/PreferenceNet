import os
import pandas as pd
import traceback
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import shutil
from tqdm import tqdm as tqdm
import torch.nn.init
from regretnet.utils import calc_agent_util
from regretnet import datasets as ds
from regretnet.regretnet import RegretNet, RegretNetUnitDemand
import argparse 
from itertools import tee
import numpy as np
import pdb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--payment-weight', type=float, default=0.1)
parser.add_argument('--preference-weight', type=float, default=0.45)
parser.add_argument('--regret-weight', type=float, default=0.45)
args = parser.parse_args()

payment_weight = args.payment_weight
preference_weight = args.preference_weight
regret_weight = args.regret_weight

train_args = torch.load("result/{0}/args.pth".format(args.model))
train_args["preference"] = ["tvf_threshold", 1.0]
train_args["preference_threshold"] = 0.8
train_args["tvf_distance"] = 0.0

assert payment_weight + preference_weight + regret_weight == 1.0, "Optimization weights must sum to 1"

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def window(iterable, size):
    iters = tee(iterable, size)
    for i in range(1, size):
        for each in iters[i:]:
            next(each, None)
    return zip(*iters)

def label_valuation(random_bids, allocs, actual_payments, type, args):
    if type == "entropy":
        assert args.n_items > 1, "Entropy regularization requires num_items > 1"
        
        allocs = allocs.clamp_min(1e-8)
        norm_allocs = allocs / allocs.sum(dim=-1).unsqueeze(-1)
        
        entropy = -1.0 * norm_allocs * torch.log(norm_allocs)
        valuation = entropy.sum(dim=-1).sum(dim=-1)
        optimize = "max"

    elif type == "tvf":
        d = args.tvf_distance
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

        valuation = unfairness.sum(dim=-1)
        optimize = "min"

    elif type == "utility":
        valuation = calc_agent_util(random_bids, allocs, actual_payments).view(-1)
        optimize = "min"

    elif type == "quota":
        assert args.n_agents > 1, "Quota regularization requires num_agents > 1"
        assert args.n_agents > 1, "Quota regularization requires num_agents > 1"
        
        allocs = allocs.clamp_min(1e-8)
        norm_allocs = allocs / allocs.sum(dim=-2).unsqueeze(-2)

        valuation =  norm_allocs.min(-1)[0].min(-1)[0]
        optimize = "max"

    elif type == "human":
        if args.preference_file is None:
            assert False, "Preference File Required"
        
        file = torch.load(args.preference_file)
        gt_allocs = torch.stack(list(file.keys()))
        gt_labels = torch.tensor([(file[key]["Yes"] / float(1e-8 + file[key]["Yes"] + file[key]["No"])) > args.preference_threshold for key in file.keys()])
        
        norm_allocs = allocs / allocs.sum(dim=-1).unsqueeze(-1)
        rounded_allocs = 5 * torch.round(20 * norm_allocs)

        valuation = []    
        for i, alloc in enumerate(rounded_allocs):
            idx = torch.argmin(torch.cdist(gt_allocs, alloc).sum(dim=-1).sum(dim=-1)).item()

            valuation.append(gt_labels[idx])

        valuation = torch.tensor(valuation)
        optimize = None 

    else:
        assert False, "Valuation type {} not supported".format(type)

    return valuation, optimize

def label_assignment(valuation, type, optimize, sample_val, args):
    if type == "threshold":
        thresh = np.quantile(sample_val, args.preference_threshold) if optimize == "max" else  np.quantile(sample_val, 1 - args.preference_threshold)
        labels = valuation > thresh if optimize == "max" else valuation < thresh 
        
        tnsr = torch.tensor([torch.tensor(int(i)) for i in labels]).float()

    elif type == "bandpass":
        pass_band = []

        for i in range(len(args.preference_passband)):
            if i % 2 == 0:
                if optimize == "max":
                    thresh_low = np.quantile(sample_val, float(args.preference_passband[i]))
                    thresh_high = np.quantile(sample_val, float(args.preference_passband[i + 1]))
                elif optimize == "min":
                    thresh_high = np.quantile(sample_val, 1 - float(args.preference_passband[i]))
                    thresh_low = np.quantile(sample_val, 1 - float(args.preference_passband[i + 1]))
                else:
                    assert False, "Optimize type {} is not supported".format(optimize)

                pass_band.append((thresh_low, thresh_high))

        labels_band = []
        for thresh_low, thresh_high in pass_band:
            label = torch.tensor([(thresh_low < val < thresh_high).item() for val in valuation])
            labels_band.append(label)

        labels = torch.sum(torch.stack(labels_band), dim=0).bool()

        tnsr = torch.tensor([torch.tensor(int(i)) for i in labels]).float()

    elif type == "quota":
        thresh = args.preference_quota / args.n_agents
        labels = valuation > thresh

        tnsr = torch.tensor([torch.tensor(int(i)) for i in labels]).float()

    elif type == "preference":
        tnsr = torch.tensor([torch.tensor(int(i)) for i in valuation]).float()

    else:
        assert False, "Assignment type {} not supported".format(type)

    return tnsr

def label_preference(random_bids, allocs, actual_payments, valuation_dist, type, args):
    valuation_fn, assignment_fn = type.split("_")

    if assignment_fn == "ranking":
        assignment_fn = "threshold"

    valuation, optimize = label_valuation(random_bids, allocs, actual_payments, valuation_fn, args)
    label = label_assignment(valuation, assignment_fn, optimize, valuation_dist, args)

    return label


def calculate_PCA(model_path, train_args):
    args = dotdict(train_args)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    ds.dataset_override(args)

    model_ckpt = torch.load(model_path)
    if "pv" in model_ckpt['name']:
        model = RegretNetUnitDemand(**(model_ckpt['arch']))
    else:
        model = RegretNet(**(model_ckpt['arch']))

    state_dict = model_ckpt['state_dict']
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    assert model_ckpt["arch"]["n_agents"] == args.n_agents and  model_ckpt["arch"]["n_items"] == args.n_items, "model-ckpt does not match n_agents and n_items in args" 
    item_ranges = ds.preset_valuation_range(args.n_agents, args.n_items, args.dataset)
    clamp_op = ds.get_clamp_op(item_ranges)

    preference_type = []
    mixed_preference_weight = 0
    for i in range(len(args.preference)):
        if i % 2 == 0:
            preference_type.append((args.preference[i], float(args.preference[i+1])))
            mixed_preference_weight = mixed_preference_weight + float(args.preference[i+1])

    assert mixed_preference_weight == 1, "Preference weights don't sum to 1."

    accuracy = 0
    for type, weight in preference_type:
        _, sample_allocs, _ = ds.generate_random_allocations_payments(args.test_num_examples, args.n_agents, args.n_items, args.unit, item_ranges, args)
        val_type = type.split("_")[0]
        valuation_dist, _ = label_valuation(None, sample_allocs, None, val_type, args)

        assert torch.sum(valuation_dist) > 0, "Valuations are all 0"

        correct = 0
        total = 0
        test_data = ds.generate_dataset_nxk(args.n_agents, args.n_items, args.test_num_examples, item_ranges).to(DEVICE)
        test_loader = ds.Dataloader(test_data, batch_size=args.test_batch_size, shuffle=True)

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch = batch.to(DEVICE)
                allocs, payments = model(batch)

                res =  label_preference(batch.cpu(), allocs.cpu(), payments.cpu(), valuation_dist.cpu(), type, args)
                
                correct = correct + torch.sum(res).item()
                total = total + res.shape[0]
        
        acc = correct/float(total)
        accuracy = accuracy + (weight * acc)

    return accuracy

def tflog2pandas(path: str) -> pd.DataFrame:
    """convert single tensorflow log file to pandas DataFrame
    Parameters
    ----------
    path : str
        path to tensorflow log file
    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }

    runlog_data =  {"test/regret_min": [],
                    "test/regret_mean": [],
                    "test/regret_max": [],
                    "test/payment_min": [],
                    "test/payment_mean": [],
                    "test/payment_max": [],
                    "step": []}

    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:

            if tag not in runlog_data:
                continue
            
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            runlog_data[tag] = values
            step_num = list(map(lambda x: x.step, event_list))
            runlog_data["step"] = step_num

    # Dirty catch of DataLossError
    except:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()

    return pd.DataFrame.from_dict(runlog_data)

folder = "run/" + args.model
file = [f for f in os.listdir(folder)][0]
dataFrame = tflog2pandas(folder + "/" + file)
bestStep = None
maxStep = None
optimal = 0

payment_max = 0
regret_max = 0

for rowTable in dataFrame.iterrows():
    payment_max = max(payment_max, rowTable[1]["test/payment_max"])
    regret_max = max(regret_max, rowTable[1]["test/regret_max"])

for rowTable in dataFrame.iterrows():
    step = int(rowTable[1]["step"])

    payment_mean = rowTable[1]["test/payment_mean"]
    regret_mean = rowTable[1]["test/regret_mean"]
    preference_mean = calculate_PCA("result/{}/{}_checkpoint.pt".format(args.model, step), train_args)
    
    payment_score = (payment_mean / payment_max)
    regret_score = (1 - (regret_mean / regret_max))
    pca_score = preference_mean

    opt = payment_weight * payment_score + preference_weight * pca_score + regret_weight * regret_score
    print("Step {}: {} | Regret : {}, Payment: {}, PCA: {}".format(step, opt, regret_score, payment_score, pca_score))

    if opt > optimal:
        optimal = opt
        bestStep = step
    
    maxStep = step

best_model = "result/" + args.model + "/{}_checkpoint.pt".format(bestStep)
#best_model = "result/" + args.model + "/{}_checkpoint.pt".format(maxStep)
print(best_model)
shutil.copy(best_model, "result/" + args.model + "/best_checkpoint.pt")
