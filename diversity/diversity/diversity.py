import torch
import torch.nn.functional as F
import pdb

#allocs of size [..., num_agents, num_items]
def get_entropy(batch, allocs, payments, args, unused=True, factor=1):
    diversity = torch.zeros(allocs.shape[0]).to(allocs.device)
    norm_allocs = torch.zeros(allocs.shape).to(allocs.device)

    allocs = allocs.clamp_min(1e-8)
    for i, e in enumerate(allocs):
        norm_allocs[i] = e / e.sum()

    entropy = -1.0 * norm_allocs * torch.log(norm_allocs)

    for i, e in enumerate(entropy):
        diversity[i] = factor * e.sum()

    return diversity

