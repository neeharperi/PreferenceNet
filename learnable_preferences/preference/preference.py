import torch
import torch.nn.functional as F
import pdb

#allocs of size [..., num_agents, num_items]
def get_preference(batch, allocs, payments, args, model=None):
    if args.preference:
        allocs = allocs.clamp_min(1e-8)
        norm_allocs = allocs / allocs.sum(dim=-1).unsqueeze(-1)

        model.eval()
        pred = model(norm_allocs)

        return float(args.preference[1]) * pred

    return torch.zeros(allocs.shape[0]).to(allocs.device)

def get_entropy(batch, allocs, payments, args):
    with torch.no_grad():
        allocs = allocs.clamp_min(1e-8)
        norm_allocs = allocs / allocs.sum(dim=-1).unsqueeze(-1)
        
        entropy = -1.0 * norm_allocs * torch.log(norm_allocs)

        loss = entropy.sum(dim=-1).sum(dim=-1)

        return loss