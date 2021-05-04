import os
import pandas as pd
import traceback
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse
import numpy as np
import shutil
import pdb

parser = argparse.ArgumentParser()

parser.add_argument('--model', required=True)
parser.add_argument('--payment-weight', type=float, default=0.2)
parser.add_argument('--preference-weight', type=float, default=0.3)
parser.add_argument('--regret-weight', type=float, default=0.5)
args = parser.parse_args()

payment_weight = args.payment_weight
preference_weight = args.preference_weight
regret_weight = args.regret_weight

assert payment_weight + preference_weight + regret_weight == 1.0, "Optimization weights must sum to 1"

# Extraction function
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
                    "test/preference_min": [],
                    "test/preference_mean": [],
                    "test/preference_max": [],
                    "step": []}

    s = path.split('/')[1]
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

folder = "run/" + args.model #"run/quota_quota_1.0/2x2-pv_1_synthetic_1.0_noise_0_0/"
file = [f for f in os.listdir(folder)][0]
dataFrame = tflog2pandas(folder + "/" + file)
bestStep = None
optimal = 0

payment_max = 0
preference_max = 0
regret_min = np.inf

for rowTable in dataFrame.iterrows():
    payment_max = max(payment_max, rowTable[1]["test/payment_max"])
    preference_max = max(preference_max, rowTable[1]["test/preference_max"])
    regret_min = max(0, min(regret_min, rowTable[1]["test/regret_min"]))

for rowTable in dataFrame.iterrows():
    step = int(rowTable[1]["step"])

    payment_mean = rowTable[1]["test/payment_mean"]
    preference_mean = rowTable[1]["test/preference_mean"]
    regret_mean = rowTable[1]["test/regret_mean"]

    opt = payment_weight * (payment_mean / payment_max) + preference_weight * (preference_mean / preference_max) + regret_weight * (regret_min / regret_min)
    print("Step {}: {}".format(step, opt))

    if opt > optimal:
        optimal = opt
        bestStep = step

best_model = "result/" + args.model + "/{}_checkpoint.pt".format(bestStep)
print(best_model)
shutil.copy(best_model, "result/" + args.model + "/best_checkpoint.pt")
