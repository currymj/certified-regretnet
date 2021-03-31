import torch
import numpy as np
import time

from regretnet.mipcertify.model import clip_relu_convert, simplify_network, sigmoid_linear_convert
from regretnet.regretnet import RegretNet, calc_agent_util
from regretnet.mipcertify.mip_solver import MIPNetwork

torch.manual_seed(1234)
np.random.seed(1234)

def zero_one_input_dom(truthful_input):
    return torch.stack((torch.zeros_like(truthful_input).flatten(), 0.3*torch.ones_like(truthful_input).flatten())).T

# model_name = '2x2_sparsemax_linearpmt_distill_fast'
model_name = '2x2_sparsemax_in_linsigpmt_scratch_fast'
path = f"model/{model_name}.pt"
checkpoint = torch.load(path, map_location=torch.device('cpu'))
curr_activation = checkpoint['arch']['p_activation']
inner_product = curr_activation == "frac_sigmoid_linear"
if not inner_product:
    checkpoint['arch']['p_activation'] = 'full_relu_clipped'  # add on clip relu otherwise
model = RegretNet(**checkpoint['arch']).to('cpu')
model.load_state_dict(checkpoint['state_dict'])

if inner_product:
    print('inner product')
    payment_head = simplify_network(sigmoid_linear_convert(model.payment_head))
    alloc_head = model.allocation_head
    mipnet = MIPNetwork(
        model.nn_model, payment_head, alloc_head, model.n_agents, model.n_items, fractional_payment=True
    )
else:
    payment_head = clip_relu_convert(model.payment_head)
    alloc_head = model.allocation_head
    mipnet = MIPNetwork(
        model.nn_model, payment_head, alloc_head, model.n_agents, model.n_items, fractional_payment=False
    )
# payment_head = clip_relu_convert(model.payment_head)
# alloc_head = model.allocation_head
# mipnet = MIPNetwork(
#     model.nn_model, payment_head, alloc_head, model.n_agents, model.n_items, fractional_payment=False
# )

truthful_input = torch.ones(model.n_agents, model.n_items)
truthful_allocs, truthful_payments = model(truthful_input)
both_util = calc_agent_util(truthful_input, truthful_allocs, truthful_payments).flatten()
input_dom = zero_one_input_dom(truthful_input)
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
var_results = []
for v in mipnet.gurobi_vars[0]:
    var_results.append(v.x)
better_bid = torch.tensor(var_results).reshape(model.n_agents, model.n_items)

truthful_results = []
for v in mipnet.gurobi_vars_truthful[0]:
    truthful_results.append(v.x)

truthful_bid = torch.tensor(truthful_results).reshape(model.n_agents, model.n_items)

a, p = model(better_bid)
misreport_util = (a*truthful_bid).sum(dim=-1) - p
a, p = model(truthful_bid)
truthful_util = (a*truthful_bid).sum(dim=-1) - p

print('better bid', better_bid)
print('truthful bid', truthful_bid)
print('better gurobi', mipnet.final_util_expr.getValue())
print('truthful gurobi', mipnet.final_util_expr_truthful.getValue())
print('better pytorch', misreport_util)
print('truthful pytorch', truthful_util)

# modify MIP code with following:
# two sets of input variables
# two copies of network
# this should be done by having function that takes input variable
print('hello')