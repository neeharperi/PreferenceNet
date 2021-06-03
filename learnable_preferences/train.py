import os
from argparse import ArgumentParser
import torch
import torch.nn as nn
import numpy as np
import shutil

from preference import datasets as pds
from regretnet import datasets as ds
from regretnet.regretnet import RegretNet, train_loop, test_loop, RegretNetUnitDemand
from torch.utils.tensorboard import SummaryWriter
from regretnet.datasets import Dataloader
from preference.network import PreferenceNet
import json
import hashlib
import pdb 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = ArgumentParser()

parser.add_argument('--random-seed', type=int, default=0)
parser.add_argument('--num-examples', type=int, default=160000)
parser.add_argument('--test-num-examples', type=int, default=20000)
parser.add_argument('--test-iter', type=int, default=5)
parser.add_argument('--n-agents', type=int, default=1)
parser.add_argument('--n-items', type=int, default=2)
parser.add_argument('--num-epochs', type=int, default=200)
parser.add_argument('--batch-size', type=int, default=2048)
parser.add_argument('--test-batch-size', type=int, default=2048)
parser.add_argument('--model-lr', type=float, default=1e-3)
parser.add_argument('--misreport-lr', type=float, default=1e-1)
parser.add_argument('--misreport-iter', type=int, default=25)
parser.add_argument('--test-misreport-iter', type=int, default=1000)
parser.add_argument('--payment_power', type=float, default=0.)
parser.add_argument('--rho', type=float, default=1.0)
parser.add_argument('--rho-incr-iter', type=int, default=2500)
parser.add_argument('--rho-incr-amount', type=float, default=1)
parser.add_argument('--lagr-update-iter', type=int, default=25)
parser.add_argument('--rgt-start', type=int, default=0)
parser.add_argument('--preference-start', type=int, default=0)  # Epoch to start minimizing fairness
parser.add_argument('--rho-preference', type=float, default=1.0)
parser.add_argument('--rho-incr-iter-preference', type=int, default=5)
parser.add_argument('--rho-incr-amount-preference', type=float, default=0.)
parser.add_argument('--lagr-update-iter-preference', type=int, default=10)
# Preference
parser.add_argument('--preference-num-examples', type=int, default=80000)
parser.add_argument('--preference-num-self-examples', type=int, default=5000)
parser.add_argument('--preference-test-num-examples', type=int, default=20000)

parser.add_argument('--preference-num-epochs', type=int, default=50)
parser.add_argument('--preference-update-freq', type=int, default=5)
parser.add_argument('--preference-synthetic-pct', type=float, default=1.0)

parser.add_argument('--preference', default=[], nargs='+', required=True)
parser.add_argument('--preference-file')
parser.add_argument('--preference-lambda', type=float, default=1.0)
parser.add_argument('--preference-label-noise', type=float, default=0)
parser.add_argument('--preference-ranking-pairwise-samples', type=int, default=5000)
parser.add_argument('--preference-threshold', type=float, default=0.80)
parser.add_argument('--preference-passband', default=[], nargs='+')
parser.add_argument('--preference-quota', type=float, default=0.80)
parser.add_argument('--tvf-distance', type=float, default=0.0)

parser.add_argument('--dataset', nargs='+', default=[], required=True)
parser.add_argument('--resume', default="")
# architectural arguments
parser.add_argument('--hidden-layer-size', type=int, default=100)
parser.add_argument('--n-hidden-layers', type=int, default=2)
parser.add_argument('--separate', action='store_true')
parser.add_argument('--name', default='testing_name')
parser.add_argument('--unit', action='store_true')  # not saved in arch but w/e

parser.add_argument('--eval_only', action='store_true')  # not saved in arch but w/e
parser.add_argument('--lagrange', action='store_true')  # not saved in arch but w/e

if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # Replaces n_items, n_agents, name
    ds.dataset_override(args)
    if args.lagrange:
        args.name = args.name + "_lagrange"
        
    unique_id = hashlib.md5(json.dumps(vars(args)).encode("utf8")).hexdigest()

    model_name = "{0}/{1}/{2}".format("_".join(args.preference), args.name, unique_id)

    if not os.path.isdir("result/{0}".format(model_name)):
        os.makedirs("result/{0}".format(model_name))

    torch.save(vars(args), "result/{0}/args.pth".format(model_name))

    if not os.path.isfile("result/{0}/{1}_checkpoint.pt".format(model_name, args.num_epochs - 1)):
        # Valuation range setup
        item_ranges = ds.preset_valuation_range(args.n_agents, args.n_items, args.dataset)
        clamp_op = ds.get_clamp_op(item_ranges)
        if args.unit:
            model = RegretNetUnitDemand(args.n_agents, args.n_items, activation='relu', 
                                        hidden_layer_size=args.hidden_layer_size, clamp_op=clamp_op,
                                        n_hidden_layers=args.n_hidden_layers).to(DEVICE)
        else:
            model = RegretNet(args.n_agents, args.n_items, activation='relu',
                            hidden_layer_size=args.hidden_layer_size, clamp_op=clamp_op,
                            n_hidden_layers=args.n_hidden_layers, separate=args.separate).to(DEVICE)

        preference_net = PreferenceNet(args.n_agents, args.n_items, args.hidden_layer_size).to(DEVICE)
        
        if os.path.isdir("run/{0}".format(model_name)):
            shutil.rmtree("run/{0}".format(model_name))

        if os.path.isdir("run/{0}_plot".format(model_name)):
            shutil.rmtree("run/{0}_plot".format(model_name))

        writer = SummaryWriter(log_dir="run/{0}".format(model_name), comment=f"{args}")
        writer.add_text('args', json.dumps(vars(args)) , 0)

        train_data = ds.generate_dataset_nxk(args.n_agents, args.n_items, args.num_examples, item_ranges).to(DEVICE)
        train_loader = Dataloader(train_data, batch_size=args.batch_size, shuffle=True)
        test_data = ds.generate_dataset_nxk(args.n_agents, args.n_items, args.test_num_examples, item_ranges).to(DEVICE)
        test_loader = Dataloader(test_data, batch_size=args.test_batch_size, shuffle=True)

        print("Training Args:")
        print(json.dumps(vars(args), indent=4, sort_keys=True))
        train_loop(model, train_loader, test_loader, args, writer, preference_net, device=DEVICE)
        writer.close()

        result = test_loop(model, test_loader, args, preference_net, device=DEVICE)
        print(f"Experiment:{args.name}")
        print(json.dumps(result, indent=4, sort_keys=True))

    print("Validate Model")
    os.system("python validate.py --model {0}".format(model_name))
    
    print("Test Model")
    os.system("python test.py --plot-name {0}_plot --model {0}".format(model_name))
