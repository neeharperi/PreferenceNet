import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from torch.utils.tensorboard import SummaryWriter


cache = {}
plots = None
nxm = None


def create_plot(n_agents, n_items, args):
    global plots, nxm, cache
    if hasattr(args, 'plot_name') and args.plot_name:
        cache = {}
        nxm = (n_agents, n_items)
        plots = plt.subplots(2, 2, figsize=(12, 10))


def add_to_plot_cache(tensor_dict):
    if plots:
        for key, tensor in tensor_dict.items():
            if key not in cache:
                cache[key] = tensor
            else:
                cache[key] = torch.cat((cache[key], tensor), dim=0)


def plot_from_cache():
    global plots
    if plots:
        fig, ((ax1, ax2), (ax3, ax4)) = plots
        batch = cache['batch']
        allocs = cache['allocs']
        regret = cache['regret']
        payment = cache['payment']

        if nxm == (2, 1):
            agt1_bid = batch[:, 0, 0].cpu().detach()
            agt2_bid = batch[:, 1, 0].cpu().detach()

            sc1 = ax1.scatter(agt1_bid, agt2_bid, c=(allocs[:, 0, 0]).cpu().detach(), cmap='cool', alpha=0.3)
            ax1.set_title("Agent 1 allocation")

            sc2 = ax2.scatter(agt1_bid, agt2_bid, c=(allocs[:, 1, 0]).cpu().detach(), cmap='cool', alpha=0.3)
            ax2.set_title("Agent 2 allocation")

            total_regret = (regret[:, 0] + regret[:, 1]).cpu().detach()
            sc3 = ax3.scatter(agt1_bid, agt2_bid, c=total_regret, cmap='cool', alpha=0.3)
            ax3.set_title("Total regret")

            total_payment = (payment[:, 0] + payment[:, 1]).cpu().detach()
            sc4 = ax4.scatter(agt1_bid, agt2_bid, c=total_payment, cmap='cool', alpha=0.3)
            ax4.set_title("Total payment")

            for i, (sc, ax) in enumerate(((sc1, ax1), (sc2, ax2), (sc3, ax3), (sc4, ax4))):
                ax.set_xlabel("Agent 1 bid")
                ax.set_ylabel("Agent 2 bid")
                fig.colorbar(sc, ax=ax)

        if nxm == (1, 2):
            cmap = plt.get_cmap('Oranges', 100)

            item1_bid = batch[:, 0, 0].cpu().detach()
            item2_bid = batch[:, 0, 1].cpu().detach()

            sc1 = ax1.scatter(item1_bid, item2_bid, c=(allocs[:, 0, 0]).cpu().detach(), s=9, cmap=cmap, alpha=0.3)
            ax1.set_title("Item 1 allocation")

            sc2 = ax2.scatter(item1_bid, item2_bid, c=(allocs[:, 0, 1]).cpu().detach(), s=9, cmap=cmap, alpha=0.3)
            ax2.set_title("Item 2 allocation")

            sc3 = ax3.scatter(item1_bid, item2_bid, c=(regret[:, 0]).cpu().detach(), s=9, cmap=cmap, alpha=0.3)
            ax3.set_title("Total regret")

            sc4 = ax4.scatter(item1_bid, item2_bid, c=(payment[:, 0]).cpu().detach(), s=9, cmap=cmap, alpha=0.3)
            ax4.set_title("Total payment")

            for i, (sc, ax) in enumerate(((sc1, ax1), (sc2, ax2), (sc3, ax3), (sc4, ax4))):
                ax.set_xlabel("Bid on item 1")
                ax.set_ylabel("Bid on item 2")
                if i == 0 or i == 1:
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=0, vmax=1))
                    fig.colorbar(sm, ax=ax)
                if i == 2 or i == 3:
                    fig.colorbar(sc, ax=ax)


def save_cache(args):
    plot_dir = os.path.join('plots', f"{args.name}_s{args.random_seed}")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    for tensor_name in ['batch', 'allocs', 'regret', 'payment']:
        tensor_path = os.path.join(plot_dir, f'{tensor_name}.pt')
        torch.save(cache[tensor_name], tensor_path)


def save_plot(args):
    if plots:
        try:
            plot_from_cache()
            save_cache(args)
            fig = plots[0]
            writer = SummaryWriter(log_dir=f"run/{args.plot_name}", comment=f"{args}")
            writer.add_figure(tag=args.plot_name, figure=fig, global_step=args.plot_num)
            writer.close()
        except AttributeError:
            print("Skipping plotting...")
