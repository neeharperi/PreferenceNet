import torch
import torch.nn.functional as F
import pdb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#allocs of size [..., num_agents, num_items]
def get_quota(batch, allocs, payments, args):
    if args.quota and args.quota[0] == "quota":
        allocs = allocs.clamp_min(1e-8)
        norm_allocs = allocs / allocs.sum(dim=-2).unsqueeze(-1)
        valuation = torch.tensor([norm_alloc.min().item() for norm_alloc in norm_allocs])

        loss = (float(args.quota[1]) * valuation).to(DEVICE)

        return loss

    return torch.zeros(allocs.shape[0]).to(allocs.device)
