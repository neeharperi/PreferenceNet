import torch
import torch.nn.functional as F
import pdb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


#allocs of size [..., num_agents, num_items]
def get_preference(batch, allocs, payments, args, model=None):
    if args.preference:
        model.eval()
        pred = model(batch, allocs, payments)

        return float(args.preference_lambda) * pred

    return torch.zeros(allocs.shape[0]).to(allocs.device)

def get_entropy(batch, allocs, payments, args):
    if args.n_items > 1:
        with torch.no_grad():
            allocs = allocs.clamp_min(1e-8)
            norm_allocs = allocs / allocs.sum(dim=-1).unsqueeze(-1)
            
            entropy = -1.0 * norm_allocs * torch.log(norm_allocs)

            loss = entropy.sum(dim=-1).sum(dim=-1)

            return loss

    return torch.zeros(allocs.shape[0]).to(allocs.device)

def get_quota(batch, allocs, payments, args):
    if args.n_agents > 1:
        with torch.no_grad():
            allocs = allocs.clamp_min(1e-8)
            norm_allocs = allocs / allocs.sum(dim=-2).unsqueeze(-1)

            loss = torch.tensor([norm_alloc.min().item() for norm_alloc in norm_allocs])
            
            return loss.to(DEVICE)

    return torch.zeros(allocs.shape[0]).to(allocs.device)

def get_unfairness(batch, allocs, payments, args):
    with torch.no_grad():
        batch, allocs, payments = batch.cpu(), allocs.cpu(), payments.cpu()
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
        
        loss = unfairness.sum(dim=-1).to(DEVICE)

        return loss