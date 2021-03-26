import torch
import torch.nn.functional as F
import pdb

#allocs of size [..., num_agents, num_items]
def get_preference(batch, allocs, payments, args, model=None):
    if args.preference:
        with torch.no_grad():
            model.eval()
            pred = model(allocs)

        return float(args.preference[1]) * pred

    return torch.zeros(allocs.shape[0]).to(allocs.device)
