import torch
import torch.nn.functional as F
import pdb

#allocs of size [..., num_agents, num_items]
def get_entropy(batch, allocs, payments, args):
    if args.diversity and args.diversity[0] == "entropy":
        allocs = allocs.clamp_min(1e-8)
        norm_allocs = allocs / allocs.sum(dim=-1).unsqueeze(-1)
        
        entropy = -1.0 * norm_allocs * torch.log(norm_allocs)

        loss = float(args.diversity[1]) * entropy.sum(dim=-1).sum(dim=-1)
        
        loss = loss.unsqueeze(1).repeat(1, args.n_items)
        return loss

    return torch.zeros(allocs.shape[0]).to(allocs.device)
