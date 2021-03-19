import torch
import torch.nn.functional as F
import pdb

#allocs of size [..., num_agents, num_items]
def get_entropy(batch, allocs, payments, args):
    if args.diversity:
        allocs = allocs.clamp_min(1e-8)
        norm_allocs = allocs / allocs.sum(dim=-1).unsqueeze(-1)
        
        entropy = -1.0 * norm_allocs * torch.log(norm_allocs)

        loss = args.lambda_entropy * entropy.sum(dim=-1).sum(dim=-1)

        return loss

    return torch.zeros(allocs.shape[0]).to(allocs.device)
