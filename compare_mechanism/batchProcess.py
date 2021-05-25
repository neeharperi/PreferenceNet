import os
import json
import pdb
'''
#RegretNet
constraint = ["fairness", "diversity", "quota"]
size = ["2x2", "2x4", "4x2", "4x4"]
auction = ["pv", "mv"]

for auc in auction:
    for constr in constraint:
        for sz in size:
            if constr == "fairness":
                preference = "tvf_threshold 1.0"
                model_path = "../{}/result/{}-{}_1_0.0_0/199_checkpoint.pt".format(constr, sz, auc)
                result_path = "../{}/result/{}-{}_1_0.0_0/test_result.json".format(constr, sz, auc)

            if constr == "diversity":
                preference = "tvf_threshold 1.0"
                model_path = "../{}/result/{}-{}_1_1.0_0/199_checkpoint.pt".format(constr, sz, auc)
                result_path = "../{}/result/{}-{}_1_1.0_0/test_result.json".format(constr, sz, auc)

            if constr == "quota":
                preference = "quota_quota 1.0"
                model_path = "../{}/result/{}-{}_1_1.0_0/199_checkpoint.pt".format(constr, sz, auc)
                result_path = "../{}/result/{}-{}_1_1.0_0/test_result.json".format(constr, sz, auc)

            dataset = "{}-{} 1".format(sz, auc)

            try:
                print(model_path)
                os.system("python evaluate.py --model-path {} --preference {} --dataset {}".format(model_path, preference, dataset))
                file = json.load(open(result_path))
                print("Regret Mean: {}".format(file["regret_mean"]))
                print("Payment Mean: {}".format(file["payment_mean"]))
            except:
                print("Error: {} Not Found".format(model_path))
            print("\n")
'''
#PreferenceNet
constraint = ["tvf_ranking_1.0", "entropy_ranking_1.0", "quota_quota_1.0"]
size = ["2x2", "2x4", "4x2", "4x4"]
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

            id = os.listdir("../learnable_preferences/result/{}/{}-{}_1_synthetic_1.0_noise_0_0".format(constr, sz, auc))[0]
            model_path = "../learnable_preferences/result/{}/{}-{}_1_synthetic_1.0_noise_0_0/{}/best_checkpoint.pt".format(constr, sz, auc, id)
            result_path = "../learnable_preferences/result/{}/{}-{}_1_synthetic_1.0_noise_0_0/{}/test_result.json".format(constr,sz, auc, id)
            dataset = "{}-{} 1".format(sz, auc)

            try:
                print(model_path)
                os.system("python evaluate.py --model-path {} --preference {} --dataset {}".format(model_path, preference, dataset))
                file = json.load(open(result_path))
                print("Regret Mean: {}".format(file["regret_mean"]))
                print("Payment Mean: {}".format(file["payment_mean"]))
            except:
                print("Error: {} Not Found".format(model_path))
            print("\n")