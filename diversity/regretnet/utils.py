import torch

""" RegretNet utility operations """


def calc_agent_util(valuations, agent_allocations, payments):
    # valuations of size -1, n_agents, n_items
    # agent_allocations of size -1, n_agents+1, n_items
    # payments of size -1, n_agents
    # allocs should drop dummy dim
    util_from_items = torch.sum(agent_allocations * valuations, dim=2)
    return util_from_items - payments


def create_combined_misreports(misreports, valuations):
    n_agents = misreports.shape[1]
    n_items = misreports.shape[2]

    # mask might be a constant that could be allocated once outside
    mask = torch.zeros(
        (misreports.shape[0], n_agents, n_agents, n_items), device=misreports.device
    )
    for i in range(n_agents):
        mask[:, i, i, :] = 1.0

    tiled_mis = misreports.view(-1, 1, n_agents, n_items).repeat(1, n_agents, 1, 1)
    tiled_true = valuations.view(-1, 1, n_agents, n_items).repeat(1, n_agents, 1, 1)

    return mask * tiled_mis + (1.0 - mask) * tiled_true


def optimize_misreports(model, current_valuations, current_misreports, misreport_iter=10, lr=1e-1):
    # misreports are same size as valuations and start at same place

    current_misreports.requires_grad_(True)

    for i in range(misreport_iter):
        model.zero_grad()  # TODO move this where it ought to be
        agent_utils = tiled_misreport_util(current_misreports, current_valuations, model)

        (misreports_grad,) = torch.autograd.grad(agent_utils.sum(), current_misreports)

        with torch.no_grad():
            current_misreports += lr * misreports_grad
            model.clamp_op(current_misreports)

    return current_misreports


def tiled_misreport_util(current_misreports, current_valuations, model):
    n_agents = current_valuations.shape[1]
    n_items = current_valuations.shape[2]

    agent_idx = list(range(n_agents))
    tiled_misreports = create_combined_misreports(
        current_misreports, current_valuations
    )
    flatbatch_tiled_misreports = tiled_misreports.view(-1, n_agents, n_items)
    allocations, payments = model(flatbatch_tiled_misreports)
    reshaped_payments = payments.view(
        -1, n_agents, n_agents
    )  # TODO verify this puts things back to the right place
    reshaped_allocations = allocations.view(-1, n_agents, n_agents, n_items)
    # slice out or mask out agent's payments and allocations
    agent_payments = reshaped_payments[:, agent_idx, agent_idx]
    agent_allocations = reshaped_allocations[:, agent_idx, agent_idx, :]
    agent_utils = calc_agent_util(
        current_valuations, agent_allocations, agent_payments
    )  # shape [-1, n_agents]
    return agent_utils
