import torch
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm as tqdm

from preference import datasets as pds
from regretnet.utils import optimize_misreports, tiled_misreport_util, calc_agent_util
from regretnet import datasets as ds

from preference import preference
import torch.nn.init
import plot_utils
import scipy.stats as st
import random
import numpy as np
from pprint import pprint
import json
import hashlib
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

def selfTraining(model, trainData):
    with torch.no_grad():
        model.eval()

        bids, allocs, payments = trainData
        bids, allocs, payments = bids.to(DEVICE), allocs.to(DEVICE), payments.to(DEVICE)

        pred = model(bids, allocs, payments)
        labels = (pred > 0.5).float().cpu()

    return labels

def label_noise(valuation, labels, noise, threshold=None, dist="gaussian"):
    def z_score(x, mu, sigma):
        return abs(x - mu) / float(sigma)

    def label_flip(pct):
        return np.random.rand() < pct

    if dist == "gaussian":
        probability = st.norm.sf([z_score(val, threshold, valuation.std()) for val in valuation])
        probability = [max(min(1.05 * p , 0.5), 0.15) for p in probability]
        flip = [label_flip(noise * p) for p in probability]
    elif dist == "uniform":
        flip = [label_flip(noise * p) for p in np.ones_like(valuation)]
    else:
        assert False, "Distribution {} is not supported".format(dist)

    perturbed_labels = torch.tensor([l.item() if f == False else not l.item() for l, f in zip(labels, flip)])

    print("{}% of Labels Flipped".format((100.0 * sum(labels != perturbed_labels))/len(labels)))

    return perturbed_labels

def label_valuation(random_bids, allocs, actual_payments, type, args):
    if type == "entropy":
        assert args.n_items > 1, "Entropy regularization requires num_items > 1"
        
        allocs = allocs.clamp_min(1e-8)
        norm_allocs = allocs / allocs.sum(dim=-1).unsqueeze(-1)
        
        entropy = -1.0 * norm_allocs * torch.log(norm_allocs)
        valuation = entropy.sum(dim=-1).sum(dim=-1)
        optimize = "max"

    elif type == "tvf":
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

        valuation = unfairness.sum(dim=-1)
        optimize = "min"

    elif type == "utility":
        valuation = calc_agent_util(random_bids, allocs, actual_payments).view(-1)
        optimize = "min"

    elif type == "quota":
        assert args.n_agents > 1, "Quota regularization requires num_agents > 1"
        
        allocs = allocs.clamp_min(1e-8)
        norm_allocs = allocs / allocs.sum(dim=-2).unsqueeze(-2)

        valuation =  norm_allocs.min(-1)[0].min(-1)[0]
        optimize = "max"
    
    elif type == "human":
        if args.preference_file is None:
            assert False, "Preference File Required"
        
        file = torch.load(args.preference_file)
        gt_allocs = torch.stack(list(file.keys()))
        gt_labels = torch.tensor([(file[key]["Yes"] / float(1e-8 + file[key]["Yes"] + file[key]["No"])) > args.preference_threshold for key in file.keys()])
        
        norm_allocs = allocs / allocs.sum(dim=-1).unsqueeze(-1)
        rounded_allocs = 5 * torch.round(20 * norm_allocs)

        valuation = []    
        for i, alloc in tqdm(enumerate(rounded_allocs), total = rounded_allocs.shape[0]):
            idx = torch.argmin(torch.cdist(gt_allocs, alloc).sum(dim=-1).sum(dim=-1)).item()

            valuation.append(gt_labels[idx])

        valuation = torch.tensor(valuation)
        optimize = None    

    else:
        assert False, "Valuation type {} not supported".format(type)

    return valuation, optimize

def label_assignment(valuation, type, optimize, args):
    if type == "threshold":
        thresh = np.quantile(valuation, args.preference_threshold)
        labels = valuation > thresh if optimize == "max" else valuation < thresh 
        
        if args.preference_label_noise > 0:
            labels = label_noise(valuation, labels, args.preference_label_noise, thresh, "gaussian")
        
        tnsr = torch.tensor([torch.tensor(int(i)) for i in labels]).float()

    elif type == "ranking":
        thresh = np.quantile(valuation, args.preference_threshold) if optimize == "max" else np.quantile(valuation, 1 - args.preference_threshold)
        cnt = torch.zeros_like(valuation)
        
        for _ in tqdm(range(args.preference_ranking_pairwise_samples)):
            idx = torch.randperm(len(valuation))

            if optimize == "max":
                cnt = cnt + (valuation > valuation[idx])
            elif optimize == "min":
                cnt = cnt + (valuation < valuation[idx])
            else:
                assert False, "Optimize type {} is not supported".format(optimize)

        labels = cnt > (args.preference_threshold * args.preference_ranking_pairwise_samples)

        if args.preference_label_noise > 0:
            labels = label_noise(valuation, labels, args.preference_label_noise, thresh, "gaussian")
        
        tnsr = torch.tensor([torch.tensor(int(i)) for i in labels]).float()
    
    elif type == "bandpass":
        pass_band = []

        for i in range(len(args.preference_passband)):
            if i % 2 == 0:
                if optimize == "max":
                    thresh_low = np.quantile(valuation, float(args.preference_passband[i]))
                    thresh_high = np.quantile(valuation, float(args.preference_passband[i + 1]))
                elif optimize == "min":
                    thresh_high = np.quantile(valuation, 1 - float(args.preference_passband[i]))
                    thresh_low = np.quantile(valuation, 1 - float(args.preference_passband[i + 1]))
                else:
                    assert False, "Optimize type {} is not supported".format(optimize)

                pass_band.append((thresh_low, thresh_high))

        labels_band = []
        for thresh_low, thresh_high in pass_band:
            label = torch.tensor([(thresh_low < val < thresh_high).item() for val in valuation])
            labels_band.append(label)

        labels = torch.sum(torch.stack(labels_band), dim=0).bool()

        if args.preference_label_noise > 0:
            labels = label_noise(valuation, labels, args.preference_label_noise, None, "uniform")

        tnsr = torch.tensor([torch.tensor(int(i)) for i in labels]).float()

    elif type == "quota":
        thresh = args.preference_quota / args.n_agents
        labels = valuation > thresh
        
        if args.preference_label_noise > 0:
            labels = label_noise(valuation, labels, args.preference_label_noise, None, "uniform")
        
        tnsr = torch.tensor([torch.tensor(int(i)) for i in labels]).float()


    elif type == "preference":
        tnsr = torch.tensor([torch.tensor(int(i)) for i in valuation]).float()

    else:
        assert False, "Assignment type {} not supported".format(type)

    return tnsr

def label_preference(random_bids, allocs, actual_payments, type, args):
    valuation_fn, assignment_fn = type.split("_")
    valuation, optimize = label_valuation(random_bids, allocs, actual_payments, valuation_fn, args)
    label = label_assignment(valuation, assignment_fn, optimize, args)

    return label

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
    test_quota = torch.Tensor().to(device)

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
        quota = preference.get_quota(batch, allocs, payments, args)

        # Record entire test data
        test_regrets = torch.cat((test_regrets, positive_regrets), dim=0)
        test_payments = torch.cat((test_payments, payments), dim=0)
        test_preference = torch.cat((test_preference, pref), dim=0)
        test_entropy = torch.cat((test_entropy, entropy), dim=0)
        test_unfairness = torch.cat((test_unfairness, unfairness), dim=0)
        test_quota = torch.cat((test_quota, quota), dim=0)

        plot_utils.add_to_plot_cache({
            "batch": batch,
            "allocs": allocs,
            "regret": regrets,
            "payment": payments
        })

    plot_utils.save_plot(args)

    mean_regret = test_regrets.sum(dim=1).mean(dim=0).item()
    std_regret = test_regrets.sum(dim=1).std(dim=0).item()

    result = {
        "payment_min": test_payments.sum(dim=1).min(dim=0)[0].item(),
        "payment_mean": test_payments.sum(dim=1).mean(dim=0).item(),
        "payment_max": test_payments.sum(dim=1).max(dim=0)[0].item(),
        "payment_std": test_payments.sum(dim=1).std(dim=0).item(),
        
        "regret_min": test_regrets.sum(dim=1).min().item(),
        "regret_mean": mean_regret,
        "regret_max": test_regrets.sum(dim=1).max().item(),
        "regret_std": std_regret,
        
        "preference_min": test_preference.min().item(),
        "preference_mean": test_preference.mean().item(),
        "preference_max": test_preference.max().item(),
        "preference_std": test_preference.std().item(),

        "entropy_min": test_entropy.min().item(),
        "entropy_mean": test_entropy.mean().item(),
        "entropy_max": test_entropy.max().item(),
        "entropy_std": test_entropy.std().item(),

        "unfairness_min": test_unfairness.min().item(),
        "unfairness_mean": test_unfairness.mean().item(),
        "unfairness_max": test_unfairness.max().item(),
        "unfairness_std": test_unfairness.std().item(),

        "quota_min": test_quota.min().item(),
        "quota_mean": test_quota.mean().item(),
        "quota_max": test_quota.max().item(),
        "quota_std": test_quota.std().item(),
    }
 
    return result

def train_preference(model, train_loader, test_loader, epoch, args):
    unique_id = hashlib.md5(json.dumps(vars(args)).encode("utf8")).hexdigest()

    BCE = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.model_lr, betas=(0.5, 0.999), weight_decay=0.005)

    for _ in tqdm(range(args.preference_num_epochs)):
        epochLoss = 0
        model.train()
        for _, data in enumerate(train_loader, 1):
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

    modelState = {"experimentName": args.name,
                  "state_dictionary": model.state_dict()
                  }

    torch.save(modelState, "result/{0}/{1}/{2}/{3}_{0}_checkpoint.pt".format("_".join(args.preference), args.name, unique_id, epoch))

    return model

def train_loop(model, train_loader, test_loader, args, writer, preference_net, device="cpu"):
    unique_id = hashlib.md5(json.dumps(vars(args)).encode("utf8")).hexdigest()

    regret_mults = 5.0 * torch.ones((1, model.n_agents)).to(device)
    payment_mult = 1
    
    if args.lagrange:
        preference_mults = torch.ones((1, model.n_items)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.model_lr)

    iter = 0
    rho = args.rho
    
    if args.lagrange:
        rho_preference = args.rho_preference

    preference_train_bids, preference_train_allocs, preference_train_payments, preference_train_labels = [], [], [], []
    preference_test_bids, preference_test_allocs, preference_test_payments, preference_test_labels = [], [], [], []

    preference_item_ranges = ds.preset_valuation_range(args, args.n_agents, args.n_items, args.dataset)
    preference_clamp_op = ds.get_clamp_op(preference_item_ranges)

    preference_type = []
    mixed_preference_weight = 0
    for i in range(len(args.preference)):
        if i % 2 == 0:
            preference_type.append((args.preference[i], float(args.preference[i+1])))
            mixed_preference_weight = mixed_preference_weight + float(args.preference[i+1])

    assert mixed_preference_weight == 1, "Preference weights don't sum to 1."

    for pref in preference_type:
        type, ratio = pref

        ##################################################################################
        if args.preference_synthetic_pct > 0:
            train_bids, train_allocs, train_payments, train_labels = pds.generate_random_allocations_payments(int(ratio * args.preference_synthetic_pct * args.preference_num_examples), args.n_agents, args.n_items, args.unit, preference_item_ranges, args, type, label_preference)
            test_bids, test_allocs, test_payments, test_labels = pds.generate_random_allocations_payments(int(ratio * args.preference_synthetic_pct * args.preference_test_num_examples), args.n_agents, args.n_items, args.unit, preference_item_ranges, args, type, label_preference)

            preference_train_bids.append(train_bids), preference_train_allocs.append(train_allocs), preference_train_payments.append(train_payments), preference_train_labels.append(train_labels)
            preference_test_bids.append(test_bids), preference_test_allocs.append(test_allocs), preference_test_payments.append(test_payments), preference_test_labels.append(test_labels)

        ####################################################################################
        if 1 - args.preference_synthetic_pct > 0:
            train_bids, train_allocs, train_payments, train_labels = pds.generate_regretnet_allocations(model, args.n_agents, args.n_items, int(ratio * (1 - args.preference_synthetic_pct) * args.preference_num_examples), preference_item_ranges, args, type, label_preference)            
            test_bids, test_allocs, test_payments, test_labels = pds.generate_regretnet_allocations(model, args.n_agents, args.n_items, int(ratio * (1 -args.preference_synthetic_pct) * args.preference_test_num_examples), preference_item_ranges, args, type, label_preference)

            preference_train_bids.append(train_bids), preference_train_allocs.append(train_allocs), preference_train_payments.append(train_payments), preference_train_labels.append(train_labels)
            preference_test_bids.append(test_bids), preference_test_allocs.append(test_allocs), preference_test_payments.append(test_payments), preference_test_labels.append(test_labels)

    preference_train_loader = pds.Dataloader(torch.cat(preference_train_bids).to(DEVICE), torch.cat(preference_train_allocs).to(DEVICE), torch.cat(preference_train_payments).to(DEVICE), torch.cat(preference_train_labels).to(DEVICE), batch_size=args.batch_size, shuffle=True, balance=True, args=args)
    preference_test_loader = pds.Dataloader(torch.cat(preference_test_bids).to(DEVICE), torch.cat(preference_test_allocs).to(DEVICE), torch.cat(preference_test_payments).to(DEVICE), torch.cat(preference_test_labels).to(DEVICE), batch_size=args.test_batch_size, shuffle=True, balance=False, args=args)

    preference_net = train_preference(preference_net, preference_train_loader, preference_test_loader, 0, args)
    preference_net.eval()

    for epoch in tqdm(range(args.num_epochs)):
        regrets_epoch = torch.Tensor().to(device)
        payments_epoch = torch.Tensor().to(device)
        preference_epoch = torch.Tensor().to(device)
        entropy_epoch = torch.Tensor().to(device)
        unfairness_epoch = torch.Tensor().to(device)
        quota_epoch = torch.Tensor().to(device)

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
            quota = preference.get_quota(batch, allocs, payments, args)

            if epoch < args.rgt_start:
                regret_loss = 0
                regret_quad = 0
            else:
                regret_loss = (regret_mults * positive_regrets).mean()
                regret_quad = (rho / 2.0) * (positive_regrets ** 2).mean()
    
            
            if args.lagrange:
                preference_loss = (preference_mults * pref).mean()
                preference_quad = (rho_preference / 2.0) * (pref ** 2).mean()
            else:
                preference_loss = pref.mean()

            # Add batch to epoch stats
            regrets_epoch = torch.cat((regrets_epoch, regrets), dim=0)
            payments_epoch = torch.cat((payments_epoch, payments), dim=0)
            preference_epoch = torch.cat((preference_epoch, pref), dim=0)
            entropy_epoch = torch.cat((entropy_epoch, entropy), dim=0)
            unfairness_epoch = torch.cat((unfairness_epoch, unfairness), dim=0)
            quota_epoch = torch.cat((quota_epoch, quota), dim=0)

            if args.lagrange:
            # Calculate loss
                loss_func = regret_loss \
                            + regret_quad \
                            - payment_loss \
                            - preference_loss \
                            + preference_quad # increase preference
            else:
                loss_func = regret_loss \
                            + regret_quad \
                            - payment_loss \
                            - preference_loss
                            
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

            if args.lagrange:
                if iter % args.lagr_update_iter_preference == 0:
                    with torch.no_grad():
                        preference_mults += rho_preference * entropy.mean(dim=0)
                if iter % args.rho_incr_iter_preference == 0:
                    rho_preference += args.rho_incr_amount_preference

        if epoch % args.preference_update_freq == 0 and args.preference_update_freq != -1:
            train_bids, train_allocs, train_payments = pds.generate_regretnet_allocations(model, args.n_agents, args.n_items, args.preference_num_self_examples, preference_item_ranges, args)
            train_labels = selfTraining(preference_net, (train_bids, train_allocs, train_payments))
            preference_train_bids.append(train_bids), preference_train_allocs.append(train_allocs), preference_train_payments.append(train_payments), preference_train_labels.append(train_labels)
                
            preference_train_loader = pds.Dataloader(torch.cat(preference_train_bids).to(DEVICE), torch.cat(preference_train_allocs).to(DEVICE), torch.cat(preference_train_payments).to(DEVICE), torch.cat(preference_train_labels).to(DEVICE), batch_size=args.batch_size, shuffle=True, balance=False, args=args)
                
            preference_net = train_preference(preference_net, preference_train_loader, preference_test_loader, epoch, args)
            preference_net.eval()
        
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
                        "result/{0}/{1}/{2}/{3}_checkpoint.pt".format("_".join(args.preference), args.name, unique_id, epoch))

        # Log training stats
        train_stats = {
            "payment_min": payments_epoch.sum(dim=1).min().item(),
            "payment_mean": payments_epoch.sum(dim=1).mean().item(),
            "payment_max": payments_epoch.sum(dim=1).max().item(),
            
            "regret_min": regrets_epoch.min().item(),
            "regret_mean": regrets_epoch.mean().item(),
            "regret_max": regrets_epoch.max().item(),
            
            "preference_min": preference_epoch.min().item(),
            "preference_mean": preference_epoch.mean().item(),
            "preference_max": preference_epoch.max().item(),

            "entropy_min": entropy_epoch.min().item(),
            "entropy_mean": entropy_epoch.mean().item(),
            "entropy_max": entropy_epoch.max().item(),

            "unfairness_min": unfairness_epoch.min().item(),
            "unfairness_mean": unfairness_epoch.mean().item(),
            "unfairness_max": unfairness_epoch.max().item(),

            "quota_min": quota_epoch.min().item(),
            "quota_mean": quota_epoch.mean().item(),
            "quota_max": quota_epoch.max().item(),
        }

        pprint(train_stats)

        for key, value in train_stats.items():
            writer.add_scalar(f'train/{key}', value, global_step=epoch)

        if args.lagrange:
            mult_stats = {
                "regret_mult": regret_mults.mean().item(),
                "payment_mult": payment_mult,
                "preference_mult": preference_mults.mean().item(),
            }
        else:
            mult_stats = {
                "regret_mult": regret_mults.mean().item(),
                "payment_mult": payment_mult,
            }

        pprint(mult_stats)

        for key, value in mult_stats.items():
            writer.add_scalar(f'multiplier/{key}', value, global_step=epoch)