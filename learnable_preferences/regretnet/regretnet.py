import torch
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm as tqdm

from preference import datasets as pds
from regretnet.utils import optimize_misreports, tiled_misreport_util, calc_agent_util
from preference import preference
import torch.nn.init
import plot_utils

from pprint import pprint
import pdb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def classificationAccuracy(model, validationData):
    correct = 0
    total = 0

    with torch.no_grad():
        model.eval()
        for data in validationData:
            bids, allocs, payments, label = data
            bids, allocs, payments, label = bids.to(DEVICE), allocs.to(DEVICE), payments.to(DEVICE), label.to(DEVICE)
        
            pred = model(bids, allocs, payments)
            pred = pred > 0.5

            correct = correct + torch.sum(pred == label)
            total = total + allocs.shape[0]

    return correct / float(total)
    
def label_preference(random_bids, allocs, actual_payments, args, type="entropy_classification", thresh=0.685, samples=1000, pct=0.75):
    if type == "entropy_classification":
        allocs = allocs.clamp_min(1e-8)
        norm_allocs = allocs / allocs.sum(dim=-1).unsqueeze(-1)
        
        entropy = -1.0 * norm_allocs * torch.log(norm_allocs)
        entropy_alloc = entropy.sum(dim=-1).sum(dim=-1)
        labels = entropy_alloc > thresh    #0.685, 0.6925, 0.625
        tnsr = torch.tensor([torch.tensor(int(i)) for i in labels]).float()

        return tnsr

    elif type == "entropy_ranking":
        allocs = allocs.clamp_min(1e-8)
        norm_allocs = allocs / allocs.sum(dim=-1).unsqueeze(-1)
        
        entropy = -1.0 * norm_allocs * torch.log(norm_allocs)
        entropy_alloc = entropy.sum(dim=-1).sum(dim=-1)
        entropy_cnt = torch.zeros_like(entropy_alloc)
        
        for i in range(samples):
            idx = torch.randperm(len(entropy_alloc))
            entropy_cnt = entropy_cnt + (entropy_alloc > entropy_alloc[idx])

        labels = entropy_cnt > (pct * samples)
        tnsr = torch.tensor([torch.tensor(int(i)) for i in labels]).float()

        return tnsr
    
    elif type == "unfairness_classification":
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
        
        unfairness_alloc = unfairness.sum(dim=-1)

        labels = unfairness_alloc < thresh  #0.175
        tnsr = torch.tensor([torch.tensor(int(i)) for i in labels]).float()

        return tnsr

    elif type == "unfairness_ranking":
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
        
        unfairness_alloc = unfairness.sum(dim=-1)
        unfairness_cnt = torch.zeros_like(unfairness_alloc)
        
        for i in range(samples):
            idx = torch.randperm(len(unfairness_alloc))
            unfairness_cnt = unfairness_cnt + (unfairness_alloc < unfairness_alloc[idx])

        labels = unfairness_cnt > (pct * samples)
        tnsr = torch.tensor([torch.tensor(int(i)) for i in labels]).float()

        return tnsr

    assert False, "Invalid Preference Type"

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


def test_loop(model, loader, args, preference_net=None, device='cpu'):
    # regrets and payments are 2d: n_samples x n_agents; unfairs is 1d: n_samples.
    test_regrets = torch.Tensor().to(device)
    test_payments = torch.Tensor().to(device)
    test_preference = torch.Tensor().to(device)
    test_entropy = torch.Tensor().to(device)
    test_unfairness = torch.Tensor().to(device)

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
        pref = preference.get_preference(batch, allocs, payments, args, preference_net)
        entropy = preference.get_entropy(batch, allocs, payments, args)
        unfairness = preference.get_unfairness(batch, allocs, payments, args)

        # Record entire test data
        test_regrets = torch.cat((test_regrets, positive_regrets), dim=0)
        test_payments = torch.cat((test_payments, payments), dim=0)
        test_preference = torch.cat((test_preference, pref), dim=0)
        test_entropy = torch.cat((test_entropy, entropy), dim=0)
        test_unfairness = torch.cat((test_unfairness, unfairness), dim=0)

        plot_utils.add_to_plot_cache({
            "batch": batch,
            "allocs": allocs,
            "regret": regrets,
            "payment": payments
        })

    plot_utils.save_plot(args)

    mean_regret = test_regrets.sum(dim=1).mean(dim=0).item()

    result = {
        "payment_mean": test_payments.sum(dim=1).mean(dim=0).item(),
        # "regret_std": regret_var ** .5,
        "regret_mean": mean_regret,
        "regret_max": test_regrets.sum(dim=1).max().item(),
        "preference_mean": test_preference.mean().item(),
        "preference_max": test_preference.max().item(),
        "entropy_mean": test_entropy.mean().item(),
        "entropy_max": test_entropy.max().item(),
        "unfairness_mean": test_unfairness.mean().item(),
        "unfairness_max": test_unfairness.max().item(),
    }
 
    return result

def train_preference(model, optimizer, train_loader, test_loader, epoch, args):
    BCE = nn.BCELoss()

    for STEP in range(args.preference_num_epochs):
        epochLoss = 0
        model.train()
        for batchCount, data in enumerate(train_loader, 1):
            bids, allocs, payments, label = data
            bids, allocs, payments, label = bids.to(DEVICE), allocs.to(DEVICE), payments.to(DEVICE), label.to(DEVICE)
        
            pred = model(bids, allocs, payments)

            optimizer.zero_grad()
            Loss = BCE(pred, label)
            epochLoss = epochLoss + Loss.item()

            Loss.backward()
            optimizer.step()

    accuracy = classificationAccuracy(model, test_loader)
    print("Classification Accuracy: {}".format(accuracy) )

    modelState = {
        "experimentName": args.name,
        "State_Dictionary": model.state_dict(),
        }

    torch.save(modelState, f"result/{args.preference[0]}/{args.name}/{epoch}_{args.preference[0]}_checkpoint.pt")

    return model, optimizer

def train_loop(model, train_loader, test_loader, args, writer, preference_net, device="cpu"):
    regret_mults = 5.0 * torch.ones((1, model.n_agents)).to(device)
    payment_mult = 1

    optimizer = optim.Adam(model.parameters(), lr=args.model_lr)
    preference_optimizer = optim.Adam(preference_net.parameters(), lr=args.model_lr, betas=(0.5, 0.999), weight_decay=0.005)

    iter = 0
    rho = args.rho

    preference_train_bids, preference_train_allocs, preference_train_payments, preference_train_labels = [], [], [], []
    preference_test_bids, preference_test_allocs, preference_test_payments, preference_test_labels = [], [], [], []

    for epoch in tqdm(range(args.num_epochs)):
        regrets_epoch = torch.Tensor().to(device)
        payments_epoch = torch.Tensor().to(device)
        preference_epoch = torch.Tensor().to(device)
        entropy_epoch = torch.Tensor().to(device)
        unfairness_epoch = torch.Tensor().to(device)
    
        if "synthetic" in args.preference[0]:
            preference_item_ranges = pds.preset_valuation_range(args.n_agents, args.n_items)
            preference_clamp_op = pds.get_clamp_op(preference_item_ranges)

            train_bids, train_allocs, train_payments, train_labels = pds.generate_random_allocations_payments(args.preference_num_examples, args.n_agents, args.n_items, args.unit, preference_item_ranges, args, label_preference)
            preference_train_bids.append(train_bids)
            preference_train_allocs.append(train_allocs)
            preference_train_payments.append(train_payments)
            preference_train_labels.append(train_labels)
            preference_train_loader = pds.Dataloader(torch.cat(preference_train_bids).to(DEVICE), torch.cat(preference_train_allocs).to(DEVICE), torch.cat(preference_train_payments).to(DEVICE), torch.cat(preference_train_labels).to(DEVICE), batch_size=args.batch_size, shuffle=True, args=args)
            
            test_bids, test_allocs, test_payments, test_labels = pds.generate_random_allocations_payments(args.preference_test_num_examples, args.n_agents, args.n_items, args.unit, preference_item_ranges, args, label_preference)
            preference_test_bids.append(test_bids)
            preference_test_allocs.append(test_allocs)
            preference_test_payments.append(test_payments)
            preference_test_labels.append(test_labels)
            preference_test_loader = pds.Dataloader(torch.cat(preference_test_bids).to(DEVICE), torch.cat(preference_test_allocs).to(DEVICE), torch.cat(preference_test_payments).to(DEVICE), torch.cat(preference_test_labels).to(DEVICE), batch_size=args.test_batch_size, shuffle=True, args=args)
        else:        
            preference_item_ranges = pds.preset_valuation_range(args.n_agents, args.n_items)
            preference_clamp_op = pds.get_clamp_op(preference_item_ranges)

            train_bids, train_allocs, train_payments, train_labels = pds.generate_regretnet_allocations(model, args.n_agents, args.n_items, args.preference_num_examples, preference_item_ranges, args, label_preference)
            preference_train_bids.append(train_bids)
            preference_train_allocs.append(train_allocs)
            preference_train_payments.append(train_payments)
            preference_train_labels.append(train_labels)
            preference_train_loader = pds.Dataloader(torch.cat(preference_train_bids).to(DEVICE), torch.cat(preference_train_allocs).to(DEVICE), torch.cat(preference_train_payments).to(DEVICE), torch.cat(preference_train_labels).to(DEVICE), batch_size=args.batch_size, shuffle=True, args=args)
            
            test_bids, test_allocs, test_payments, test_labels = pds.generate_regretnet_allocations(model, args.n_agents, args.n_items, args.preference_test_num_examples, preference_item_ranges, args, label_preference)
            preference_test_bids.append(test_bids)
            preference_test_allocs.append(test_allocs)
            preference_test_payments.append(test_payments)
            preference_test_labels.append(test_labels)
            preference_test_loader = pds.Dataloader(torch.cat(preference_test_bids).to(DEVICE), torch.cat(preference_test_allocs).to(DEVICE), torch.cat(preference_test_payments).to(DEVICE), torch.cat(preference_test_labels).to(DEVICE), batch_size=args.test_batch_size, shuffle=True, args=args)

        if epoch % args.preference_update_freq == 0:
            preference_net, preference_optimizer = train_preference(preference_net, preference_optimizer, preference_train_loader, preference_test_loader, epoch, args)
            preference_net.eval()

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
            pref = preference.get_preference(batch, allocs, payments, args, preference_net)
            entropy = preference.get_entropy(batch, allocs, payments, args)
            unfairness = preference.get_unfairness(batch, allocs, payments, args)

            if epoch < args.rgt_start:
                regret_loss = 0
                regret_quad = 0
            else:
                regret_loss = (regret_mults * positive_regrets).mean()
                regret_quad = (rho / 2.0) * (positive_regrets ** 2).mean()
    
            preference_loss = pref.mean()
  

            # Add batch to epoch stats
            regrets_epoch = torch.cat((regrets_epoch, regrets), dim=0)
            payments_epoch = torch.cat((payments_epoch, payments), dim=0)
            preference_epoch = torch.cat((preference_epoch, pref), dim=0)
            entropy_epoch = torch.cat((entropy_epoch, entropy), dim=0)
            unfairness_epoch = torch.cat((unfairness_epoch, unfairness), dim=0)

            # Calculate loss
            loss_func = regret_loss \
                        + regret_quad \
                        - payment_loss \
                        - preference_loss # increase preference

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

        # Log testing stats and save model
        if epoch % args.test_iter == (args.test_iter - 1):
            test_result = test_loop(model, test_loader, args, preference_net, device=device)
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
                       f"result/{args.preference[0]}/{args.name}/{epoch}_checkpoint.pt")

        # Log training stats
        train_stats = {
            "regret_max": regrets_epoch.max().item(),
            "regret_mean": regrets_epoch.mean().item(),

            "payment": payments_epoch.sum(dim=1).mean().item(),

            "preference_max": preference_epoch.max().item(),
            "preference_mean": preference_epoch.mean().item(),
            "entropy_max": entropy_epoch.max().item(),
            "entropy_mean": entropy_epoch.mean().item(),
            "unfairness_max": unfairness_epoch.max().item(),
            "unfairness_mean": unfairness_epoch.mean().item(),
        }

        pprint(train_stats)

        for key, value in train_stats.items():
            writer.add_scalar(f'train/{key}', value, global_step=epoch)

        mult_stats = {
            "regret_mult": regret_mults.mean().item(),
            "payment_mult": payment_mult,
        }

        pprint(mult_stats)

        for key, value in mult_stats.items():
            writer.add_scalar(f'multiplier/{key}', value, global_step=epoch)
