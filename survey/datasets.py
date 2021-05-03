import torch
from tqdm import tqdm
import numpy as np
import re
import pdb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Dataloader(object):
    def __init__(self, bids, allocs=None, payments=None, labels=None, batch_size=64, shuffle=True, balance=False, args=None):
        self.shuffle = shuffle
        self.balance = balance
        self.batch_size = batch_size
        self.size = bids.size(0)
        self.bids = bids
        self.allocs = allocs
        self.payments = payments
        self.labels = labels

        self.iter = 0
        self.args=args

    def _sampler(self, size, batch_size, shuffle=True, balance=False):
        if balance:
            assert shuffle, "Shuffle must be true to balance the dataset"
            idx = torch.arange(size)
            num_pos = len(idx[self.labels == 1])
            num_neg = len(idx[self.labels == 0])

            if num_pos == 0 or num_neg == 0:
                print("Warning: Balanced Sampling Failed!")
                idxs = torch.randperm(size)
            else:
                pos = torch.tensor(np.random.choice(idx[self.labels == 1], int(0.5 * size)))
                neg = torch.tensor(np.random.choice(idx[self.labels == 0], int(0.5 * size)))
                balanced_idx = torch.cat((pos, neg))
                idxs = balanced_idx[torch.randperm(len(balanced_idx))]
        else:
            if shuffle:
                idxs = torch.randperm(size)
            else:
                idxs = torch.arange(size)

        for batch_idxs in idxs.split(batch_size):
            yield batch_idxs

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter == 0:
            self.sampler = self._sampler(self.size, self.batch_size, shuffle=self.shuffle, balance=self.balance)
        self.iter = (self.iter + 1) % (len(self) + 1)
        idx = next(self.sampler)
        
        if self.allocs is None and self.payments is None and self.labels is None:
            return self.bids[idx]

        return self.bids[idx], self.allocs[idx], self.payments[idx], self.labels[idx]

    def __len__(self):
        return (self.size - 1) // self.batch_size + 1


def dataset_override(args):
    # Preset multiple variables with dataset name
    if args.dataset:
        regex = re.search("(\d+)x(\d+)-(pv|mv)", args.dataset[0])

        # Preset multiple variables with dataset name
        if args.dataset:
            args.n_agents = int(regex.group(1))
            args.n_items = int(regex.group(2))

            if "pv" in regex.group(3):
                args.unit = True



# TODO: Use valuation ranges in a preset file
def preset_valuation_range(n_agents, n_items, dataset=None):
    # defaults
    zeros = torch.zeros(n_agents, n_items)
    ones = torch.ones(n_agents, n_items)
    item_ranges = torch.stack((zeros, ones), dim=2).reshape(n_agents, n_items, 2)
    # modifications
    if dataset:
        if 'manelli' in dataset[0] or 'mv' in dataset[0]:
            multiplier = float(dataset[1]) if len(dataset) > 1 else 1
            if '3x10' in dataset[0]:
                a = [0, 1, 2, 3, 4]
                item_ranges[:, a, 1] = item_ranges[:, a, 1] * multiplier
            else:
                item_ranges[:, :, 1] = item_ranges[:, :, 1]*multiplier
        elif 'pavlov' in dataset[0] or 'pv' in dataset[0]:
            item_ranges = item_ranges + 2
        else:
            print(dataset[0], 'is not a valid dataset name. Defaulting to Manelli-Vincent auction.')
    # item_ranges is a n_agents x n_items x 2 tensor where item_ranges[agent_i][item_j] = [lower_bound, upper_bound].
    assert item_ranges.shape == (n_agents, n_items, 2)
    return item_ranges


def generate_linspace_nxk(n_agents, n_items, item_ranges, s=100):
    # For 2-item auctions only.
    b1 = torch.linspace(*item_ranges[0, 0], s)
    b2 = torch.linspace(*item_ranges[0, 1], s)
    return torch.stack(torch.meshgrid([b1, b2]), dim=2).reshape(s**2, 1, 2)


def generate_dataset_nxk(n_agents, n_items, num_examples, item_ranges):
    range_diff = item_ranges[:, :, 1] - item_ranges[:, :, 0]
    return range_diff * torch.rand(num_examples, n_agents, n_items) + item_ranges[:, :, 0]


def get_clamp_op(item_ranges: torch.Tensor):
    def clamp_op(batch):
        samples, n_agents, n_items = batch.shape
        for i in range(n_agents):
            for j in range(n_items):
                lower = item_ranges[i, j, 0]
                upper = item_ranges[i, j, 1]
                batch[:, i, j] = batch[:, i, j].clamp_min(lower).clamp_max(upper)
    return clamp_op

def generate_random_allocations_payments(n_allocations, n_agents, n_items, unit_demand, item_ranges, args, type=None, preference=None):
    """
    Generates random allocations (uniform, unit-demand or not).
    """
    # randomly generate bids in ranges
    random_bids = generate_dataset_nxk(n_agents, n_items, n_allocations, item_ranges)

    # random allocations, before normalization
    random_points = torch.rand(n_allocations, n_agents + 1, n_items + 1)

    if unit_demand:
        agent_normalized = torch.softmax(random_points, dim=-1)
        random_points_2 = torch.rand(n_allocations, n_agents + 1, n_items + 1)
        item_normalized = torch.softmax(random_points_2, dim=-2)
        allocs = torch.min(item_normalized, agent_normalized)[..., 0:-1, 0:-1]
    else:
        allocs = torch.softmax(random_points, dim=-2)[..., 0:-1, 0:-1]

    # compute a random value [0,1] representing % util to charge as payment
    random_frac_payments = torch.rand(n_allocations, n_agents)

    # compute utils assuming bids were truthful
    agent_utils = (random_bids * allocs).sum(dim=-1)

    actual_payments = random_frac_payments * agent_utils

    allocs = allocs.clamp_min(1e-8)
    norm_allocs = allocs / allocs.sum(dim=-1).unsqueeze(-1)

    if type is None and preference is None:
        return random_bids, norm_allocs, actual_payments

    valuation, labels = preference(random_bids, allocs, actual_payments, type, args)

    return random_bids, norm_allocs, actual_payments, labels, valuation

def calc_agent_util(valuations, agent_allocations, payments):
    # valuations of size -1, n_agents, n_items
    # agent_allocations of size -1, n_agents+1, n_items
    # payments of size -1, n_agents
    # allocs should drop dummy dim
    util_from_items = torch.sum(agent_allocations * valuations, dim=2)
    return util_from_items - payments

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

        allocs = allocs.clamp_min(1e-8)
        norm_allocs = allocs / allocs.sum(dim=-2).unsqueeze(-1)
        valuation = torch.tensor([norm_alloc.min().item() for norm_alloc in norm_allocs])
        optimize = "max"

    else:
        assert False, "Valuation type {} not supported".format(type)

    return valuation, optimize

def label_assignment(valuation, type, optimize, args):
    if type == "threshold":
        thresh = np.quantile(valuation, args.preference_threshold)
        labels = valuation > thresh if optimize == "max" else valuation < thresh 
        
        tnsr = torch.tensor([torch.tensor(int(i)) for i in labels]).float()

    elif type == "ranking":
        thresh = np.quantile(valuation, args.preference_threshold) if optimize == "max" else np.quantile(valuation, 1 - args.preference_threshold)
        cnt = torch.zeros_like(valuation)
        
        for _ in range(args.preference_ranking_pairwise_samples):
            idx = torch.randperm(len(valuation))

            if optimize == "max":
                cnt = cnt + (valuation > valuation[idx])
            elif optimize == "min":
                cnt = cnt + (valuation < valuation[idx])
            else:
                assert False, "Optimize type {} is not supported".format(optimize)

        labels = cnt > (args.preference_threshold * args.preference_ranking_pairwise_samples)
        
        tnsr = torch.tensor([torch.tensor(int(i)) for i in labels]).float()
    
    elif type == "bandpass":
        pass_band = []

        for i in range(len(args.preference_passband)):
            if i % 2 == 0:
                if optimize == "max":
                    thresh_low = np.quantile(valuation, float(args.preference_passband[i]))
                    thresh_high = np.quantile(valuation, float(args.preference_passband[i + 1]))
                elif optimize == "min":
                    thresh_high = np.quantile(valuation, 1 - float(args.preference_passband[i]))
                    thresh_low = np.quantile(valuation, 1 - float(args.preference_passband[i + 1]))
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
        labels = valuation > args.preference_quota
        
        tnsr = torch.tensor([torch.tensor(int(i)) for i in labels]).float()

    else:
        assert False, "Assignment type {} not supported".format(type)

    return tnsr

def label_preference(random_bids, allocs, actual_payments, type, args):
    valuation_fn, assignment_fn = type.split("_")
    valuation, optimize = label_valuation(random_bids, allocs, actual_payments, valuation_fn, args)
    label = label_assignment(valuation, assignment_fn, optimize, args)

    return valuation, label