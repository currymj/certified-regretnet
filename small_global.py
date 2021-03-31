import torch
import numpy as np
import time
from regretnet.regretnet import RegretNet, calc_agent_util
from regretnet.mipcertify.mip_solver import MIPNetwork

torch.manual_seed(1234)
np.random.seed(1234)

def convert_input_dom(truthful_input, player_index):
    """
    Converts a truthful input into a set of input bounds for player i
    :param truthful_input: the current set of truthful bids
    :param player_index: the player whose input can vary
    :return: a tensor of upper and lower bounds in the format which MIPNetwork can accept.
    """
    input_lbs = truthful_input.clone()
    input_ubs = truthful_input.clone()
    input_lbs[player_index, :] = 0.0
    input_ubs[player_index, :] = 1.0
    return torch.stack((input_lbs.flatten(), input_ubs.flatten())).T

model = RegretNet(1, 2, activation='relu', hidden_layer_size=15,n_hidden_layers=1,p_activation='full_linear',a_activation='sparsemax',
                  separate=False)
payment_head = model.payment_head
alloc_head = model.allocation_head
mipnet = MIPNetwork(
    model.nn_model, payment_head, alloc_head, model.n_agents, model.n_items, fractional_payment=False
)

truthful_input = torch.tensor([[0.5,0.5]])
truthful_allocs, truthful_payments = model(truthful_input)
both_util = calc_agent_util(truthful_input, truthful_allocs, truthful_payments).flatten()
input_dom = convert_input_dom(truthful_input, 0)
truthful_util = both_util[0].item()
mipnet.setup_model(
    input_dom, truthful_input, truthful_util, use_obj_function=True, player_ind=0
)

start_solve = time.time()
point_was_found, result_tuple, num_states = mipnet.solve(input_dom, truthful_util)
end_solve = time.time()

print(mipnet.final_util_expr.getValue() - mipnet.final_util_expr_truthful.getValue())
# first step: make sure can get local bound for model at random point

# need to: convert and linearize model to get upper and lower bounds for all relu

# modify MIP code with following:
# two sets of input variables
# two copies of network
# this should be done by having function that takes input variable
print('hello')