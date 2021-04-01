import torch
import pdb

class Dataloader(object):
    def __init__(self, data, labels=None, batch_size=64, shuffle=True):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.size = data.size(0)
        self.data = data
        self.labels = labels
        self.iter = 0

    def _sampler(self, size, batch_size, shuffle=True):
        if shuffle:
            idxs = torch.randperm(size)
        else:
            idxs = torch.arange(size)
        for batch_idxs in idxs.split(batch_size):
            yield batch_idxs

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter == 0:
            self.sampler = self._sampler(self.size, self.batch_size, shuffle=self.shuffle)
        self.iter = (self.iter + 1) % (len(self) + 1)
        idx = next(self.sampler)

        if self.labels is None:
            return self.data[idx]
        else:
            return self.data[idx], self.labels[idx]

    def __len__(self):
        return (self.size - 1) // self.batch_size + 1


def dataset_override(args):
    # Preset multiple variables with dataset name
    if args.dataset:
        if args.dataset[0].startswith('1x2'):
            args.n_agents = 1
            args.n_items = 2
            if 'pv' in args.dataset[0]:
                args.unit = True
        if args.dataset[0].startswith('2x1'):
            args.n_agents = 2
            args.n_items = 1
            if 'pv' in args.dataset[0]:
                args.unit = True
        if args.dataset[0].startswith('2x4'):
            args.n_agents = 2
            args.n_items = 4
        if args.dataset[0].startswith('3x10'):
            args.n_agents = 3
            args.n_items = 10

        if args.name == 'testing_name':
            args.name = '_'.join([str(x) for x in args.dataset] +
                                 [str(args.random_seed)])


# TODO: Use valuation ranges in a preset file
def preset_valuation_range(n_agents, n_items, dataset=None):
    # defaults
    zeros = torch.zeros(n_agents, n_items)
    ones = torch.ones(n_agents, n_items)
    item_ranges = torch.stack((zeros, ones), dim=2).reshape(n_agents, n_items, 2)
    # modifications
    if dataset:
        if 'manelli' in dataset[0] or 'mv' in dataset[0]:
            multiplier = float(dataset[1]) if len(dataset) > 1 else 1
            if '3x10' in dataset[0]:
                a = [0, 1, 2, 3, 4]
                item_ranges[:, a, 1] = item_ranges[:, a, 1] * multiplier
            else:
                item_ranges[:, :, 1] = item_ranges[:, :, 1]*multiplier
        elif 'pavlov' in dataset[0] or 'pv' in dataset[0]:
            item_ranges = item_ranges + 2
        else:
            print(dataset[0], 'is not a valid dataset name. Defaulting to Manelli-Vincent auction.')
    # item_ranges is a n_agents x n_items x 2 tensor where item_ranges[agent_i][item_j] = [lower_bound, upper_bound].
    assert item_ranges.shape == (n_agents, n_items, 2)
    return item_ranges


def generate_linspace_nxk(n_agents, n_items, item_ranges, s=100):
    # For 2-item auctions only.
    b1 = torch.linspace(*item_ranges[0, 0], s)
    b2 = torch.linspace(*item_ranges[0, 1], s)
    return torch.stack(torch.meshgrid([b1, b2]), dim=2).reshape(s**2, 1, 2)


def generate_dataset_nxk(n_agents, n_items, num_examples, item_ranges):
    range_diff = item_ranges[:, :, 1] - item_ranges[:, :, 0]
    return range_diff * torch.rand(num_examples, n_agents, n_items) + item_ranges[:, :, 0]


def get_clamp_op(item_ranges: torch.Tensor):
    def clamp_op(batch):
        samples, n_agents, n_items = batch.shape
        for i in range(n_agents):
            for j in range(n_items):
                lower = item_ranges[i, j, 0]
                upper = item_ranges[i, j, 1]
                batch[:, i, j] = batch[:, i, j].clamp_min(lower).clamp_max(upper)
    return clamp_op


def generate_random_allocations(n_allocations, n_agents, n_items, unit_demand, args, preference=None):
    """
    Generates random allocations (uniform, unit-demand or not).
    """
    random_points = torch.rand(n_allocations, n_agents + 1, n_items + 1)
    if unit_demand:
        agent_normalized = torch.softmax(random_points, dim=-1)
        random_points_2 = torch.rand(n_allocations, n_agents + 1, n_items + 1)
        item_normalized = torch.softmax(random_points_2, dim=-2)
        vals = torch.min(item_normalized, agent_normalized)[...,0:-1,0:-1]
    else:
        vals = torch.softmax(random_points, dim=-2)[..., 0:-1, 0:-1]

    if preference is None:
        return vals
    else:
        labels = preference(vals)
        return vals, labels
