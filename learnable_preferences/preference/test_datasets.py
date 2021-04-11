import torch
from pytest import approx
from .datasets import generate_random_allocations_payments

def test_random_allocs():
    n_agents = 2
    n_items = 2
    item_ranges = torch.tensor([[[0.0,1.0],[0.0,1.0]],[[0.0,1.0],[0.0,1.0]]])
    num_examples = 1000

    bids, allocs, payments = generate_random_allocations_payments(num_examples, n_agents, n_items, False, item_ranges, None, None)

    assert ((bids*allocs).sum(dim=-1) >= payments).all()


    bids, allocs, payments = generate_random_allocations_payments(num_examples, n_agents, n_items, True, item_ranges, None,
                                                                  None)
    assert ((bids*allocs).sum(dim=-1) >= payments).all()
