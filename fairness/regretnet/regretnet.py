import torch
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm as tqdm

from regretnet.utils import optimize_misreports, tiled_misreport_util, calc_agent_util
from fairness import fairness
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
    fairness_args = fairness.setup_fairness(args, device)
    # regrets and payments are 2d: n_samples x n_agents; unfairs is 1d: n_samples.
    test_regrets = torch.Tensor().to(device)
    test_payments = torch.Tensor().to(device)
    test_unfairs = torch.Tensor().to(device)
    test_variations = torch.Tensor().to(device)

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
        unfairs = fairness.get_unfairness(batch, allocs, payments, fairness_args)
        variations = fairness.max_variation(batch, allocs, payments, fairness_args[1])

        # Record entire test data
        test_regrets = torch.cat((test_regrets, positive_regrets), dim=0)
        test_payments = torch.cat((test_payments, payments), dim=0)
        test_unfairs = torch.cat((test_unfairs, unfairs), dim=0)
        test_variations = torch.cat((test_variations, variations), dim=0)

        plot_utils.add_to_plot_cache({
            "batch": batch,
            "allocs": allocs,
            "regret": regrets,
            "payment": payments
        })

    plot_utils.save_plot(args)

    mean_regret = test_regrets.sum(dim=1).mean(dim=0).item()
    # mean_sq_regret = (test_regrets ** 2).sum(dim=1).mean(dim=0).item()
    # TODO: idk why sometimes the variance is a very small negative number, but this is a hacky fix
    # regret_var = max(mean_sq_regret - mean_regret ** 2, 0)
    result = {
        "payment_mean": test_payments.sum(dim=1).mean(dim=0).item(),
        # "regret_std": regret_var ** .5,
        "regret_mean": mean_regret,
        "regret_max": test_regrets.sum(dim=1).max().item(),
        "unfairness_mean": test_unfairs.mean().item(),
        "unfairness_max": test_unfairs.max().item(),
        "variation_max": test_variations.max().item(),
    }
    # for i in range(model.n_agents):
    #     agent_regrets = test_regrets[:, i]
    #     result[f"regret_agt{i}_std"] = (((agent_regrets ** 2).mean() - agent_regrets.mean() ** 2) ** .5).item()
    #     result[f"regret_agt{i}_mean"] = agent_regrets.mean().item()
    return result


def train_loop(model, train_loader, test_loader, args, writer, device="cpu"):
    fairness_args = fairness.setup_fairness(args, device)

    regret_mults = 5.0 * torch.ones((1, model.n_agents)).to(device)
    payment_mult = 1
    fair_mults = torch.ones((1, model.n_items)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.model_lr)

    iter = 0
    rho = args.rho
    rho_fair = args.rho_fair

    # local_optimum_model = None

    for epoch in tqdm(range(args.num_epochs)):
        regrets_epoch = torch.Tensor().to(device)
        payments_epoch = torch.Tensor().to(device)
        unfairness_epoch = torch.Tensor().to(device)
        variation_epoch = torch.Tensor().to(device)
        # price_of_fair_epoch = torch.Tensor().to(device)
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

            # "Smart initialization" for payment mult?
            # if epoch == 0 and i == 0:
            #     regret_loss = (regret_mults * positive_regrets).mean()
            #     regret_quad = (rho / 2.0) * (positive_regrets ** 2).mean()
            #     payment_mult = 2 * ((regret_loss + regret_quad) / payments.sum(dim=1).mean()).detach().item()
            ###

            payment_loss = payments.sum(dim=1).mean() * payment_mult
            variations = fairness.max_variation(batch, allocs, payments, fairness_args[1])
            unfairness = fairness.get_unfairness(batch, allocs, payments, fairness_args, min(1, epoch))

            if epoch < args.rgt_start:
                regret_loss = 0
                regret_quad = 0
            else:
                regret_loss = (regret_mults * positive_regrets).mean()
                regret_quad = (rho / 2.0) * (positive_regrets ** 2).mean()
                # regret_loss = (regret_mults * (positive_regrets + positive_regrets.max(dim=0).values) / 2).mean()
                # regret_quad = (rho / 2.0) * ((positive_regrets ** 2).mean() +
                #                              (positive_regrets.max(dim=0).values ** 2).mean()) / 2

            # if epoch < args.fair_start:
            #     unfairness_quad = 0
            #     unfairness_loss = 0
            # else:
            unfairness_loss = (fair_mults * unfairness).mean()
            unfairness_quad = (rho_fair / 2.0) * (unfairness ** 2).mean()

            # Price of fairness
            # price_of_fair = torch.zeros(batch.shape[0]).to(device)
            # if local_optimum_model:
            #     opt_allocs, opt_payments = local_optimum_model(batch)
            #     total_payments = payments.sum(dim=1)
            #     total_payments_opt = opt_payments.sum(dim=1)
            #     # need to clamp_min on opt_ratio in case network finds a payment above the optimum
            #     opt_ratio = (total_payments_opt - total_payments).clamp_min(0) / total_payments_opt
            #     # Not sure if we should use same lagrange multiplier as fairness_loss?
            #     price_of_fair = (args.min_payment_ratio - opt_ratio).clamp_min(0)
            # pricefair_loss = price_of_fair.mean(dim=0) * payment_mult

            # Add batch to epoch stats
            regrets_epoch = torch.cat((regrets_epoch, regrets), dim=0)
            payments_epoch = torch.cat((payments_epoch, payments), dim=0)
            variation_epoch = torch.cat((variation_epoch, variations), dim=0)
            unfairness_epoch = torch.cat((unfairness_epoch, unfairness), dim=0)
            # price_of_fair_epoch = torch.cat((price_of_fair_epoch, price_of_fair), dim=0)

            # Calculate loss
            loss_func = regret_loss \
                        + regret_quad \
                        - payment_loss \
                        + unfairness_loss \
                        + unfairness_quad \
                        # + pricefair_loss

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
            # if epoch >= args.fair_start:
            if iter % args.lagr_update_iter_fair == 0:
                with torch.no_grad():
                    fair_mults += rho_fair * unfairness.mean(dim=0)
            if iter % args.rho_incr_iter_fair == 0:
                rho_fair += args.rho_incr_amount_fair
                # if local_optimum_model is None:
                #     local_optimum_model = RegretNet(args.n_agents, args.n_items, activation='relu',
                #                                     hidden_layer_size=args.hidden_layer_size,
                #                                     n_hidden_layers=args.n_hidden_layers,
                #                                     separate=args.separate).to(device)
                #     local_optimum_model.load_state_dict(model.state_dict())

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
            "unfairness_max": unfairness_epoch.max().item(),
            "unfairness_mean": unfairness_epoch.mean().item(),
            "variation_max": variation_epoch.max().item()
        }

        pprint(train_stats)

        for key, value in train_stats.items():
            writer.add_scalar(f'train/{key}', value, global_step=epoch)

        mult_stats = {
            "regret_mult": regret_mults.mean().item(),
            "payment_mult": payment_mult,
            "fair_mult": fair_mults.mean().item(),
        }

        pprint(mult_stats)
        
        for key, value in mult_stats.items():
            writer.add_scalar(f'multiplier/{key}', value, global_step=epoch)
