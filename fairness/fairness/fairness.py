import torch
from torch import nn
from regretnet.utils import calc_agent_util
from fairness import featuresets


""" Total variation fairness """


def setup_fairness(args, device):
    if args.fairness and args.fairness[0] == 'tvf':
        if len(args.fairness) == 3:
            _, c_name, d_name = args.fairness
            afeats = torch.load(c_name + '.pt').astype(int)  # file for ad prefs
            ufeats = torch.load(d_name + '.pt').astype(int)  # file for user qualities
            C = featuresets.load_categories(afeats)
            D = featuresets.generate_distance(afeats, ufeats).to(device)
            return ['tvf', C, D]
        if len(args.fairness) == 2:
            # Single category, uniform distance
            _, d = args.fairness
            C = featuresets.single_category(args.n_agents)
            D = featuresets.uniform_distance(1, args.n_items, float(d)).to(device)
            return ['tvf', C, D]


def get_unfairness(batch, allocs, payments, fairness, factor=1):
    # factor is from 0 - 1, for gradual application as epochs increase

    # Later on we can specify names for these different types of loss functions
    # if fairness_type == 'maximin':
    #     return fairness_maximin(batch, allocs, payments, *fairness_params)
    # if fairness_type == 'alloc_restrict':
    #     return fairness_alloc_restrict(batch, allocs, payments, *fairness_params)
    # if fairness_type == 'competitive_bid':
    #     return fairness_competitive_bid(batch, allocs, payments, *fairness_params)
    # if fairness_type == 'exorbitant_bid':
    #     return fairness_exorbitant_bid(batch, allocs, payments, *fairness_params)
    # if fairness_type == 'bid_proportional':
    #     return fairness_bid_proportional(batch, allocs, payments, *fairness_params)
    # if fairness_type == 'alloc_competitive_restrict':
    #     return fairness_alloc_competitive_restrict(batch, allocs, payments, *fairness_params)
    if fairness:
        if fairness[0] == 'total_variation' or fairness[0] == 'tvf':
            return fairness_total_variation(batch, allocs, payments, *fairness[1:], factor)
    return torch.zeros(batch.shape[0]).to(allocs.device)


def fairness_total_variation(batch, allocs, payments, C, D, factor):
    L, n, m = allocs.shape
    unfairness = torch.zeros(L, m).to(allocs.device)
    for i, C_i in enumerate(C):
        for u in range(m):
            for v in range(m):
                # L1 norm of difference (distance between allocation distributions)
                subset_allocs_diff = (allocs[:, C_i, u] - allocs[:, C_i, v]).abs()
                # If allocation distance is greater than user distance, penalize
                D2 = 1 - (1 - D) * factor if n == 1 else 2 - (2 - D) * factor
                unfairness[:, u] += (subset_allocs_diff.sum(dim=1) - D2[i, u, v]).clamp_min(0)
    return unfairness


# Get the max L1 norm variation within each category - useful to track when using uniform distance
def max_variation(batch, allocs, payments, C):
    L, n, m = allocs.shape
    variation = torch.zeros(L, len(C)).to(allocs.device)
    for i, C_i in enumerate(C):
        for u in range(m):
            for v in range(m):
                subset_allocs_diff = (allocs[:, C_i, u] - allocs[:, C_i, v]).abs().sum(dim=1)
                variation[:, i] = variation[:, i].max(subset_allocs_diff)
    return variation


""" Unused functions """


def fairness_maximin(batch, allocs, payments, d=0.5):
    """ Maximizes the minimum utility in order to satisfy d.
    d: maximum allowed difference of utility between any two agents. """
    agent_utils = calc_agent_util(batch, allocs, payments)
    max_agent_utils = agent_utils.max(dim=1).values
    min_agent_utils = agent_utils.min(dim=1).values
    return (-d + max_agent_utils - min_agent_utils).clamp_min(min=0)


def fairness_alloc_restrict(batch, allocs, payments, c=0.7):
    """ c: maximum allowed allocation probability for any agent. """
    return (-c + allocs.sum(dim=2)).clamp_min(min=0).sum(dim=1)


def fairness_competitive_bid(batch, allocs, payments, c=0.7, d=0.5):
    """ Maximin with a required "competitive" bid threshold.
    c: ratio of highest bid to be considered competitive
    d: maximum allocation difference between any competitive bid vs. max allocation."""
    # batch shape: (L samples, N agents, M items)
    # samples x items, each element is c*max bid
    cutoff_bid_item = c * batch.max(dim=1, keepdim=True).values
    # competitiveness below cutoff bid = 0, at max bid = 1.
    competitiveness = ((batch - cutoff_bid_item) / (1 - cutoff_bid_item)).clamp_min(min=0)
    # allocations shape: (n_agents (+1 dummy), M items)
    allocation_disp = (-d + allocs.max(dim=1, keepdim=True).values - allocs).clamp_min(min=0)
    return (competitiveness * allocation_disp).sum(dim=(1, 2))


def fairness_exorbitant_bid(batch, allocs, payments, d=0.5):
    """ Maximin with a required "exorbitant" bid threshold.
    c: ratio of highest bid to be considered competitive
    d: maximum allocation difference between any exorbitant bid vs. max allocation. """
    bid_proportions = batch / batch.sum(dim=2, keepdim=True)
    allocation_disp = (-d + allocs.max(dim=1, keepdim=True).values - allocs).clamp_min(min=0)
    return (bid_proportions * allocation_disp).sum(dim=(1, 2))


def fairness_bid_proportional(batch, allocs, payments, c=0.7):
    """ Bidder's allocation must be proportional to their share of bids.
    c: e.g. our bids take up 40% of auction share, but we only receive 20% of allocations.
    Setting c > 0.5 penalizes the network, but c < 0.5 does not."""
    alloc_proportion = allocs.sum(dim=2, keepdim=True) / allocs.shape[2]
    bid_proportion = batch.sum(dim=2, keepdim=True) / batch.sum(dim=(1,2), keepdim=True)
    return ((c * bid_proportion) - alloc_proportion).clamp_min(min=0).sum(dim=1)


def fairness_alloc_competitive_restrict(batch, allocs, payments, c=0.7, L=.55):
    """ Because of sigmoid think of L > 0.5 as one allocated item """
    m = nn.Sigmoid()
    unfair_allocs = m(allocs - (allocs.max(1, True)[0] * c))
    return (-L + unfair_allocs.sum(2)).clamp_min(0).sum(1)
