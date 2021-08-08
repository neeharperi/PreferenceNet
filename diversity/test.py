import os
from argparse import ArgumentParser
import torch
import numpy as np
from regretnet.datasets import generate_dataset_nxk, preset_valuation_range, generate_linspace_nxk, get_clamp_op
from regretnet.regretnet import RegretNet, train_loop, test_loop, RegretNetUnitDemand
from regretnet.datasets import Dataloader
import json
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = ArgumentParser()
parser.add_argument('--random-seed', type=int, default=0)
parser.add_argument('--test-num-examples', type=int, default=3000)
parser.add_argument('--batch-size', type=int, default=2048)
parser.add_argument('--test-batch-size', type=int, default=512)
parser.add_argument('--misreport-lr', type=float, default=2e-2)
parser.add_argument('--misreport-iter', type=int, default=25)
parser.add_argument('--test-misreport-iter', type=int, default=1000)
parser.add_argument('--model', default="")
# Fairness
parser.add_argument('--diversity', nargs='+', default=[])

# Plotting
parser.add_argument('--plot-name', default='')
parser.add_argument('--plot-num', type=int, default=0)

if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    model_name = args.model
    if not args.model.startswith("result"):
        args.model = os.path.join("result", args.model)

    if os.path.isdir(args.model):
        args.model = max([os.path.join(args.model, fn) for fn in os.listdir(args.model) if "checkpoint" in fn], key=os.path.getctime)

    checkpoint = torch.load(args.model)
    print("Architecture:")
    print(json.dumps(checkpoint['arch'], indent=4, sort_keys=True))
    print("Training Args:")
    print(json.dumps(vars(checkpoint['args']), indent=4, sort_keys=True))

    train_args = checkpoint['args']
    args.n_agents = train_args.n_agents
    args.n_items = train_args.n_items
    args.name = train_args.name
    args.dataset = train_args.dataset

    item_ranges = preset_valuation_range(train_args, args.n_agents, args.n_items, train_args.dataset)
    clamp_op = get_clamp_op(item_ranges)
    if train_args.unit:
        model = RegretNetUnitDemand(**checkpoint['arch'], clamp_op=clamp_op).to(DEVICE)
    else:
        model = RegretNet(**checkpoint['arch'], clamp_op=clamp_op).to(DEVICE)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    if args.n_agents == 1 and args.n_items == 2:
        test_data = generate_linspace_nxk(args.n_agents, args.n_items, item_ranges)
    else:
        test_data = generate_dataset_nxk(args.n_agents, args.n_items, args.test_num_examples, item_ranges).to(DEVICE)

    test_loader = Dataloader(test_data, batch_size=args.test_batch_size, shuffle=True)

    result = test_loop(model, test_loader, args, device=DEVICE)
    print(f"Experiment:{checkpoint['name']}")
    print(json.dumps(result, indent=4, sort_keys=True))

    with open("result/{}/test_result.json".format(model_name), 'w') as f:
        json.dump(result, f)