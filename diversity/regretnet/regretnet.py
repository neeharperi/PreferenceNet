import torch
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm as tqdm

from regretnet.utils import optimize_misreports, tiled_misreport_util, calc_agent_util
from diversity import diversity
import torch.nn.init
import plot_utils

from pprint import pprint


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class View_Cut(nn.Module):
    def __init__(self):
        super(View_Cut, self).__init__()

    def forward(self, x):
        return x[:, :-1, :]


class RegretNetUnitDemand(nn.Module):
    def __init__(self, n_agents, n_items, clamp_op=None, hidden_layer_size=128, n_hidden_layers=2, activation='tanh', 
                 separate=False):
        super(RegretNetUnitDemand, self).__init__()
        self.activation = activation
        if activation == 'tanh':
            self.act = nn.Tanh
        else:
            self.act = nn.ReLU

        self.clamp_op = clamp_op

        self.n_agents = n_agents
        self.n_items = n_items

        self.input_size = self.n_agents * self.n_items
        self.hidden_layer_size = hidden_layer_size
        self.n_hidden_layers = n_hidden_layers
        self.separate = separate

        # outputs are agents (+dummy agent) per item (+ dummy item), plus payments per agent
        self.allocations_size = (self.n_agents + 1) * (self.n_items + 1)
        self.payments_size = self.n_agents

        self.nn_model = nn.Sequential(
            *([nn.Linear(self.input_size, self.hidden_layer_size), self.act()] +
              [l for i in range(self.n_hidden_layers)
               for l in (nn.Linear(self.hidden_layer_size, self.hidden_layer_size), self.act())])
        )

        self.allocation_head = nn.Linear(self.hidden_layer_size, self.allocations_size * 2)
        self.fractional_payment_head = nn.Sequential(
            nn.Linear(self.hidden_layer_size, self.payments_size), nn.Sigmoid()
        )

    def glorot_init(self):
        """
        reinitializes with glorot (aka xavier) uniform initialization
        """

        def initialize_fn(layer):
            if type(layer) == nn.Linear:
                torch.nn.init.xavier_uniform_(layer.weight)

        self.apply(initialize_fn)

    def forward(self, reports):
        x = reports.view(-1, self.n_agents * self.n_items)
        x = self.nn_model(x)

        alloc_scores = self.allocation_head(x)
        alloc_first = F.softmax(alloc_scores[:, 0:self.allocations_size].view(-1, self.n_agents + 1, self.n_items + 1),
                                dim=1)
        alloc_second = F.softmax(
            alloc_scores[:, self.allocations_size:self.allocations_size * 2].view(-1, self.n_agents + 1,
                                                                                  self.n_items + 1), dim=2)
        allocs = torch.min(alloc_first, alloc_second)

        payments = self.fractional_payment_head(x) * torch.sum(
            allocs[:, :-1, :-1] * reports, dim=2
        )

        return allocs[:, :-1, :-1], payments


class RegretNet(nn.Module):
    def __init__(self, n_agents, n_items, hidden_layer_size=128, clamp_op=None, n_hidden_layers=2,
                 activation='tanh', separate=False):
        super(RegretNet, self).__init__()

        # this is for additive valuations only
        self.activation = activation
        if activation == 'tanh':
            self.act = nn.Tanh
        else:
            self.act = nn.ReLU

        self.clamp_op = clamp_op

        self.n_agents = n_agents
        self.n_items = n_items

        self.input_size = self.n_agents * self.n_items
        self.hidden_layer_size = hidden_layer_size
        self.n_hidden_layers = n_hidden_layers
        self.separate = separate

        # outputs are agents (+dummy agent) per item, plus payments per agent
        self.allocations_size = (self.n_agents + 1) * self.n_items
        self.payments_size = self.n_agents

        # Set a_activation to softmax
        self.allocation_head = [nn.Linear(self.hidden_layer_size, self.allocations_size),
                                View((-1, self.n_agents + 1, self.n_items)),
                                nn.Softmax(dim=1),
                                View_Cut()]

        # Set p_activation to frac_sigmoid
        self.payment_head = [
            nn.Linear(self.hidden_layer_size, self.payments_size), nn.Sigmoid()
        ]

        if separate:
            self.nn_model = nn.Sequential()
            self.payment_head = [nn.Linear(self.input_size, self.hidden_layer_size), self.act()] + \
                                [l for i in range(n_hidden_layers)
                                 for l in (nn.Linear(self.hidden_layer_size, self.hidden_layer_size), self.act())] + \
                                self.payment_head

            self.payment_head = nn.Sequential(*self.payment_head)
            self.allocation_head = [nn.Linear(self.input_size, self.hidden_layer_size), self.act()] + \
                                   [l for i in range(n_hidden_layers)
                                    for l in (nn.Linear(self.hidden_layer_size, self.hidden_layer_size), self.act())] + \
                                   self.allocation_head
            self.allocation_head = nn.Sequential(*self.allocation_head)
        else:
            self.nn_model = nn.Sequential(
                *([nn.Linear(self.input_size, self.hidden_layer_size), self.act()] +
                  [l for i in range(self.n_hidden_layers)
                   for l in (nn.Linear(self.hidden_layer_size, self.hidden_layer_size), self.act())])
            )
            self.allocation_head = nn.Sequential(*self.allocation_head)
            self.payment_head = nn.Sequential(*self.payment_head)

    def glorot_init(self):
        """
        reinitializes with glorot (aka xavier) uniform initialization
        """

        def initialize_fn(layer):
            if type(layer) == nn.Linear:
                torch.nn.init.xavier_uniform_(layer.weight)

        self.apply(initialize_fn)

    def forward(self, reports):
        # x should be of size [batch_size, n_agents, n_items
        # should be reshaped to [batch_size, n_agents * n_items]
        # output should be of size [batch_size, n_agents, n_items],
        # either softmaxed per item, or else doubly stochastic
        x = reports.view(-1, self.n_agents * self.n_items)
        x = self.nn_model(x)
        allocs = self.allocation_head(x)

        # frac_sigmoid payment: multiply p = p_tilde * sum(alloc*bid)
        payments = self.payment_head(x) * torch.sum(
            allocs * reports, dim=2
        )

        return allocs, payments


def test_loop(model, loader, args, device='cpu'):
    # regrets and payments are 2d: n_samples x n_agents; unfairs is 1d: n_samples.
    test_regrets = torch.Tensor().to(device)
    test_payments = torch.Tensor().to(device)
    test_entropy = torch.Tensor().to(device)

    plot_utils.create_plot(model.n_agents, model.n_items, args)

    for i, batch in enumerate(loader):
        batch = batch.to(device)
        misreport_batch = batch.clone().detach()
        optimize_misreports(model, batch, misreport_batch, 
                            misreport_iter=args.test_misreport_iter, lr=args.misreport_lr)

        allocs, payments = model(batch)
        truthful_util = calc_agent_util(batch, allocs, payments)
        misreport_util = tiled_misreport_util(misreport_batch, batch, model)

        regrets = misreport_util - truthful_util
        positive_regrets = torch.clamp_min(regrets, 0)
        entropy = diversity.get_entropy(batch, allocs, payments, args)

        # Record entire test data
        test_regrets = torch.cat((test_regrets, positive_regrets), dim=0)
        test_payments = torch.cat((test_payments, payments), dim=0)
        test_entropy = torch.cat((test_entropy, entropy), dim=0)

        plot_utils.add_to_plot_cache({
            "batch": batch,
            "allocs": allocs,
            "regret": regrets,
            "payment": payments
        })

    plot_utils.save_plot(args)

    mean_regret = test_regrets.sum(dim=1).mean(dim=0).item()
    std_regret = test_regrets.sum(dim=1).mean(dim=0).item()

    #result = {
    #    "payment_mean": test_payments.sum(dim=1).mean(dim=0).item(),
    #    # "regret_std": regret_var ** .5,
    #    "regret_mean": mean_regret,
    #    "regret_max": test_regrets.sum(dim=1).max().item(),
    #    "entropy_mean": test_entropy.mean().item(),
    #    "entropy_max": test_entropy.max().item(),
    #}

    result = {
        "payment_min": test_payments.sum(dim=1).min(dim=0)[0].item(),
        "payment_mean": test_payments.sum(dim=1).mean(dim=0).item(),
        "payment_max": test_payments.sum(dim=1).max(dim=0)[0].item(),
        "payment_std": test_payments.sum(dim=1).std(dim=0).item(),
        
        "regret_min": test_regrets.sum(dim=1).min().item(),
        "regret_mean": mean_regret,
        "regret_max": test_regrets.sum(dim=1).max().item(),
        "regret_std": std_regret,

        "entropy_min": test_entropy.min().item(),
        "entropy_mean": test_entropy.mean().item(),
        "entropy_max": test_entropy.max().item(),
        "entropy_std": test_entropy.std().item(),
    }
 
    return result


def train_loop(model, train_loader, test_loader, args, writer, device="cpu"):
    regret_mults = 5.0 * torch.ones((1, model.n_agents)).to(device)
    payment_mult = 1

    if not args.no_lagrange:
        diversity_mults = torch.ones((1, model.n_items)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.model_lr)

    iter = 0
    rho = args.rho
    if not args.no_lagrange:
        rho_diversity = args.rho_diversity

    for epoch in tqdm(range(args.num_epochs)):
        regrets_epoch = torch.Tensor().to(device)
        payments_epoch = torch.Tensor().to(device)
        entropy_epoch = torch.Tensor().to(device)

        for i, batch in enumerate(train_loader):
            iter += 1
            batch = batch.to(device)
            misreport_batch = batch.clone().detach().to(device)
            optimize_misreports(model, batch, misreport_batch, 
                                misreport_iter=args.misreport_iter, lr=args.misreport_lr)

            allocs, payments = model(batch)
            truthful_util = calc_agent_util(batch, allocs, payments)
            misreport_util = tiled_misreport_util(misreport_batch, batch, model)
            regrets = misreport_util - truthful_util
            positive_regrets = torch.clamp_min(regrets, 0)

            payment_loss = payments.sum(dim=1).mean() * payment_mult
            entropy = diversity.get_entropy(batch, allocs, payments, args)

            if epoch < args.rgt_start:
                regret_loss = 0
                regret_quad = 0
            else:
                regret_loss = (regret_mults * positive_regrets).mean()
                regret_quad = (rho / 2.0) * (positive_regrets ** 2).mean()

            if not args.no_lagrange:
                diversity_loss = (diversity_mults * entropy).mean()
                diversity_quad = (rho_diversity / 2.0) * (entropy ** 2).mean()
            else:
                diversity_loss = (entropy).mean()

            # Add batch to epoch stats
            regrets_epoch = torch.cat((regrets_epoch, regrets), dim=0)
            payments_epoch = torch.cat((payments_epoch, payments), dim=0)
            entropy_epoch = torch.cat((entropy_epoch, entropy), dim=0)

            # Calculate loss
            
            if not args.no_lagrange:
                loss_func = regret_loss \
                            + regret_quad \
                            - payment_loss \
                            - diversity_loss \
                            + diversity_quad # increase diversity
            else:
                loss_func = regret_loss \
                            + regret_quad \
                            - payment_loss \
                            - diversity_loss
            
            # update model
            optimizer.zero_grad()
            loss_func.backward()
            optimizer.step()

            # update various fancy multipliers
            # if epoch >= args.rgt_start:
            if iter % args.lagr_update_iter == 0:
                with torch.no_grad():
                    regret_mults += rho * positive_regrets.mean(dim=0)
            if iter % args.rho_incr_iter == 0:
                rho += args.rho_incr_amount

            if not args.no_lagrange:
                if iter % args.lagr_update_iter_diversity == 0:
                    with torch.no_grad():
                        diversity_mults += rho_diversity * entropy.mean(dim=0)
                if iter % args.rho_incr_iter_diversity == 0:
                    rho_diversity += args.rho_incr_amount_diversity

        # Log testing stats and save model
        if epoch % args.test_iter == (args.test_iter - 1):
            test_result = test_loop(model, test_loader, args, device=device)
            for key, value in test_result.items():
                writer.add_scalar(f"test/{key}", value, global_step=epoch)

            arch = {'n_agents': model.n_agents,
                    'n_items': model.n_items,
                    'hidden_layer_size': model.hidden_layer_size,
                    'n_hidden_layers': model.n_hidden_layers,
                    'activation': model.activation,
                    'separate': model.separate}
            torch.save({'name': args.name,
                        'arch': arch,
                        'state_dict': model.state_dict(),
                        'args': args},
                       f"result/{args.name}/{epoch}_checkpoint.pt")

        # Log training stats
        train_stats = {
            "regret_max": regrets_epoch.max().item(),
            "regret_mean": regrets_epoch.mean().item(),

            "payment": payments_epoch.sum(dim=1).mean().item(),

            "entropy_max": entropy_epoch.max().item(),
            "entropy_mean": entropy_epoch.mean().item(),
        }

        pprint(train_stats)

        for key, value in train_stats.items():
            writer.add_scalar(f'train/{key}', value, global_step=epoch)

        if not args.no_lagrange:
            mult_stats = {
                "regret_mult": regret_mults.mean().item(),
                "payment_mult": payment_mult,
                "diversity_mult": diversity_mults.mean().item(),
            }
        else:
            mult_stats = {
                "regret_mult": regret_mults.mean().item(),
                "payment_mult": payment_mult,
            }

        pprint(mult_stats)

        for key, value in mult_stats.items():
            writer.add_scalar(f'multiplier/{key}', value, global_step=epoch)
