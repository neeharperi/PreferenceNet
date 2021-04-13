import torch
import torch.nn.functional as F
import pdb

#allocs of size [..., num_agents, num_items]
def get_preference(batch, allocs, payments, args, model=None):
    if args.preference:
        if "entropy" in args.preference[0]:
            allocs = allocs.clamp_min(1e-8)
            norm_allocs = allocs / allocs.sum(dim=-1).unsqueeze(-1)

            model.eval()
            pred = model(batch, norm_allocs, payments)

            return float(args.preference[1]) * pred
            
        elif "tvf" in args.preference[0]:
            model.eval()
            pred = model(batch, allocs, payments)

            return float(args.preference[1]) * pred

    return torch.zeros(allocs.shape[0]).to(allocs.device)

def get_entropy(batch, allocs, payments, args):
    with torch.no_grad():
        allocs = allocs.clamp_min(1e-8)
        norm_allocs = allocs / allocs.sum(dim=-1).unsqueeze(-1)
        
        entropy = -1.0 * norm_allocs * torch.log(norm_allocs)

        loss = entropy.sum(dim=-1).sum(dim=-1)

        return loss

def get_tvf(batch, allocs, payments, args):
    d = 0.0
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
    
    loss = unfairness.sum(dim=-1)

    return loss