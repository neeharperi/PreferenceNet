import json
import pdb
import os
import torch
from subprocess import PIPE, Popen

def shell(command):
    process = Popen(
        args=command,
        stdout=PIPE,
        shell=True
    )
    return process.communicate()[0].decode("utf8")

'''
masterTable = {}

#RegretNet
constraint = ["fairness", "diversity", "quota"]
size = ["5x10", "10x5"]
auction = ["pv", "mv"]

for val in ["PCA", "Regret", "Payment"]:
    masterTable[val] = {}

    for auc in auction:
        for sz in size:
            masterTable[val]["{}-{}".format(sz, auc)] = {}

            for constr in ["fairness", "diversity", "quota"]:
                masterTable[val]["{}-{}".format(sz, auc)][constr] = {}

                for net in ["RegretNet", "PreferenceNet"]:
                    masterTable[val]["{}-{}".format(sz, auc)][constr][net] = "tba"

for auc in auction:
    for constr in constraint:
        for sz in size:
            if constr == "fairness":
                preference = "tvf_threshold 1.0"
                model_path = "../{}/result/{}-{}_1_0.0_0_no_lagrange/best_checkpoint.pt".format(constr, sz, auc)
                result_path = "../{}/result/{}-{}_1_0.0_0_no_lagrange/test_result.json".format(constr, sz, auc)

            if constr == "diversity":
                preference = "entropy_threshold 1.0"
                model_path = "../{}/result/{}-{}_1_1.0_0_no_lagrange/best_checkpoint.pt".format(constr, sz, auc)
                result_path = "../{}/result/{}-{}_1_1.0_0_no_lagrange/test_result.json".format(constr, sz, auc)

            if constr == "quota":
                preference = "quota_quota 1.0"
                model_path = "../{}/result/{}-{}_1_1.0_0_no_lagrange/best_checkpoint.pt".format(constr, sz, auc)
                result_path = "../{}/result/{}-{}_1_1.0_0_no_lagrange/test_result.json".format(constr, sz, auc)

            dataset = "{}-{} 1".format(sz, auc)

            try:
                print(model_path)
                output = shell("python evaluate.py --model-path {} --preference {} --dataset {}".format(model_path, preference, dataset))
                pca = float(output.split("\n")[1].split(":")[1])

                file = json.load(open(result_path))
                regret_mean, regret_std, payment_mean, payment_std = file["regret_mean"], file["regret_std"], file["payment_mean"], file["payment_std"]
                
                print("Preference Accuracy: {}".format(pca))
                print("Regret: {} / {}".format(regret_mean, regret_std))
                print("Payment: {} / {}".format(payment_mean, payment_std))

                masterTable["PCA"]["{}-{}".format(sz, auc)][constr]["RegretNet"] = str(round(100 * pca, 1))
                masterTable["Regret"]["{}-{}".format(sz, auc)][constr]["RegretNet"] = "{} ({})".format(round(regret_mean, 3), round(regret_std, 3))
                masterTable["Payment"]["{}-{}".format(sz, auc)][constr]["RegretNet"] = "{} ({})".format(round(payment_mean, 2), round(payment_std, 2))
            except:
                print("Error: {} Not Found".format(model_path))
            print("\n")

#PreferenceNet
constraint = ["tvf_ranking_1.0", "entropy_ranking_1.0", "quota_quota_1.0"]
size = ["5x10", "10x5"]
auction = ["pv", "mv"]

for auc in auction:
    for constr in constraint:
        for sz in size:
            if constr == "tvf_ranking_1.0":
                preference = "tvf_threshold 1.0"

            if constr == "entropy_ranking_1.0":
                preference = "tvf_threshold 1.0"

            if constr == "quota_quota_1.0":
                preference = "quota_quota 1.0"

            ids = os.listdir("../learnable_preferences/result/{}/{}-{}_1_synthetic_1.0_noise_0_0".format(constr, sz, auc))

            for id in ids:
                model_path = "../learnable_preferences/result/{}/{}-{}_1_synthetic_1.0_noise_0_0/{}/best_checkpoint.pt".format(constr, sz, auc, id)
                result_path = "../learnable_preferences/result/{}/{}-{}_1_synthetic_1.0_noise_0_0/{}/test_result.json".format(constr,sz, auc, id)
                dataset = "{}-{} 1".format(sz, auc)

                if not os.path.isfile(result_path):
                    continue

                try:
                    print(model_path)
                    output = shell("python evaluate.py --model-path {} --preference {} --dataset {}".format(model_path, preference, dataset))
                    pca = float(output.split("\n")[1].split(":")[1])

                    file = json.load(open(result_path))
                    regret_mean, regret_std, payment_mean, payment_std = file["regret_mean"], file["regret_std"], file["payment_mean"], file["payment_std"]
                    
                    print("Preference Accuracy: {}".format(pca))
                    print("Regret: {} / {}".format(regret_mean, regret_std))
                    print("Payment: {} / {}".format(payment_mean, payment_std))

                    if constr == "tvf_ranking_1.0":
                        preference = "fairness"

                    if constr == "entropy_ranking_1.0":
                        preference = "diversity"

                    if constr == "quota_quota_1.0":
                        preference = "quota"

                    masterTable["PCA"]["{}-{}".format(sz, auc)][preference]["PreferenceNet"] = str(round(100 * pca, 1))
                    masterTable["Regret"]["{}-{}".format(sz, auc)][preference]["PreferenceNet"] = "{} ({})".format(round(regret_mean, 3), round(regret_std, 3))
                    masterTable["Payment"]["{}-{}".format(sz, auc)][preference]["PreferenceNet"] = "{} ({})".format(round(payment_mean, 2), round(payment_std, 2))
                except:
                    print("Error: {} Not Found".format(model_path))
                print("\n")


file = open("table.txt", "w")
for val in ["PCA", "Regret", "Payment"]:
    for auc in auction:
        for sz in size:
            line = "\\multirow{1}{*}{" + sz + " " + auc + "}"
            for constr in ["fairness", "diversity", "quota"]:
                for net in ["RegretNet", "PreferenceNet"]:
                    line = line + " & " + masterTable[val]["{}-{}".format(sz, auc)][constr][net]
            line = line + " \\\\"

            file.write(line)

        file.write("\n")
    file.write("\n")
'''

masterTable = {}

#RegretNet
constraint = ["fairness", "diversity", "quota"]
size = ["2x2"]
auction = ["am"]
multiplier = 8

for val in ["PCA", "Regret", "Payment"]:
    masterTable[val] = {}

    for auc in auction:
        for sz in size:
            masterTable[val]["{}-{}".format(sz, auc)] = {}

            for constr in ["fairness", "diversity", "quota"]:
                masterTable[val]["{}-{}".format(sz, auc)][constr] = {}

                for net in ["RegretNet", "PreferenceNet"]:
                    masterTable[val]["{}-{}".format(sz, auc)][constr][net] = "tba"

for auc in auction:
    for constr in constraint:
        for sz in size:
            if constr == "fairness":
                preference = "tvf_threshold 1.0"
                model_path = "../{}/result/{}-{}_1_0.0_0_{}_no_lagrange/best_checkpoint.pt".format(constr, sz, auc, multiplier)
                result_path = "../{}/result/{}-{}_1_0.0_0_{}_no_lagrange/test_result.json".format(constr, sz, auc, multiplier)

            if constr == "diversity":
                preference = "entropy_threshold 1.0"
                model_path = "../{}/result/{}-{}_1_1.0_0_{}_no_lagrange/best_checkpoint.pt".format(constr, sz, auc, multiplier)
                result_path = "../{}/result/{}-{}_1_1.0_0_{}_no_lagrange/test_result.json".format(constr, sz, auc, multiplier)

            if constr == "quota":
                preference = "quota_quota 1.0"
                model_path = "../{}/result/{}-{}_1_1.0_0_{}_no_lagrange/best_checkpoint.pt".format(constr, sz, auc, multiplier)
                result_path = "../{}/result/{}-{}_1_1.0_0_{}_no_lagrange/test_result.json".format(constr, sz, auc, multiplier)

            dataset = "{}-{} 1".format(sz, auc)

            try:
                print(model_path)
                output = shell("python evaluate.py --model-path {} --preference {} --dataset {}".format(model_path, preference, dataset))
                pca = float(output.split("\n")[1].split(":")[1])

                file = json.load(open(result_path))
                regret_mean, regret_std, payment_mean, payment_std = file["regret_mean"], file["regret_std"], file["payment_mean"], file["payment_std"]
                
                print("Preference Accuracy: {}".format(pca))
                print("Regret: {} / {}".format(regret_mean, regret_std))
                print("Payment: {} / {}".format(payment_mean, payment_std))

                masterTable["PCA"]["{}-{}".format(sz, auc)][constr]["RegretNet"] = str(round(100 * pca, 1))
                masterTable["Regret"]["{}-{}".format(sz, auc)][constr]["RegretNet"] = "{} ({})".format(round(regret_mean, 3), round(regret_std, 3))
                masterTable["Payment"]["{}-{}".format(sz, auc)][constr]["RegretNet"] = "{} ({})".format(round(payment_mean, 2), round(payment_std, 2))
            except:
                print("Error: {} Not Found".format(model_path))
            print("\n")

#PreferenceNet
constraint = ["tvf_ranking_1.0", "entropy_ranking_1.0", "quota_quota_1.0"]
size = ["2x2"]
auction = ["am"]

for auc in auction:
    for constr in constraint:
        for sz in size:
            if constr == "tvf_ranking_1.0":
                preference = "tvf_threshold 1.0"

            if constr == "entropy_ranking_1.0":
                preference = "tvf_threshold 1.0"

            if constr == "quota_quota_1.0":
                preference = "quota_quota 1.0"

            ids = os.listdir("../learnable_preferences/result/{}/{}-{}_1_synthetic_1.0_noise_0_0".format(constr, sz, auc))

            for id in ids:
                args = "../learnable_preferences/result/{}/{}-{}_1_synthetic_1.0_noise_0_0/{}/args.pth".format(constr, sz, auc, id)
                mult = torch.load(args)["multiplierB"]

                if multiplier != mult:
                    continue

                model_path = "../learnable_preferences/result/{}/{}-{}_1_synthetic_1.0_noise_0_0/{}/best_checkpoint.pt".format(constr, sz, auc, id)
                result_path = "../learnable_preferences/result/{}/{}-{}_1_synthetic_1.0_noise_0_0/{}/test_result.json".format(constr,sz, auc, id)
                dataset = "{}-{} 1".format(sz, auc)

                if not os.path.isfile(result_path):
                    continue

                try:
                    print(model_path)
                    output = shell("python evaluate.py --model-path {} --preference {} --dataset {}".format(model_path, preference, dataset))
                    pca = float(output.split("\n")[1].split(":")[1])

                    file = json.load(open(result_path))
                    regret_mean, regret_std, payment_mean, payment_std = file["regret_mean"], file["regret_std"], file["payment_mean"], file["payment_std"]
                    
                    print("Preference Accuracy: {}".format(pca))
                    print("Regret: {} / {}".format(regret_mean, regret_std))
                    print("Payment: {} / {}".format(payment_mean, payment_std))

                    if constr == "tvf_ranking_1.0":
                        preference = "fairness"

                    if constr == "entropy_ranking_1.0":
                        preference = "diversity"

                    if constr == "quota_quota_1.0":
                        preference = "quota"

                    masterTable["PCA"]["{}-{}".format(sz, auc)][preference]["PreferenceNet"] = str(round(100 * pca, 1))
                    masterTable["Regret"]["{}-{}".format(sz, auc)][preference]["PreferenceNet"] = "{} ({})".format(round(regret_mean, 3), round(regret_std, 3))
                    masterTable["Payment"]["{}-{}".format(sz, auc)][preference]["PreferenceNet"] = "{} ({})".format(round(payment_mean, 2), round(payment_std, 2))
                except:
                    print("Error: {} Not Found".format(model_path))
                print("\n")


file = open("table.txt", "w")
for val in ["PCA", "Regret", "Payment"]:
    for auc in auction:
        for sz in size:
            line = "\\multirow{1}{*}{" + sz + " " + auc + "}"
            for constr in ["fairness", "diversity", "quota"]:
                for net in ["RegretNet", "PreferenceNet"]:
                    line = line + " & " + masterTable[val]["{}-{}".format(sz, auc)][constr][net]
            line = line + " \\\\"

            file.write(line)

        file.write("\n")
    file.write("\n")