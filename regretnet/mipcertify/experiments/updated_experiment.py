import pickle
import os
import torch
import time
import numpy as np
from torch import nn
import matplotlib.pyplot as plt

from regretnet import ibp
from regretnet.mipcertify.mip_solver import MIPNetwork
from regretnet.regretnet import RegretNet, calc_agent_util, optimize_misreports, tiled_misreport_util
from regretnet.mipcertify.model import clip_relu_convert, clip_relu_remove, simplify_network, sigmoid_linear_convert
from datetime import datetime

RANDOM_SEED = 4321
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
def curr_timestamp():
    return datetime.strftime(datetime.now(), format="%Y-%m-%d_%H-%M-%S")


EXP_TIME = curr_timestamp()


def plot_misreports(experiment_dicts, player_ind):
    for d in experiment_dicts:
        if "regret" in d:
            truthful = d["truthful_input"][player_ind,:].numpy()
            misreport = d["better_bid"][player_ind,:].numpy()
            difference = misreport - truthful
            plt.plot([truthful[0]], [truthful[1]], marker='o', markersize=30*d["regret"], color='blue')
            plt.plot([misreport[0]], [misreport[1]], marker='o', markersize=3, color='red')
            plt.arrow(truthful[0], truthful[1], difference[0], difference[1], head_width=0.01)
    plt.show()

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

def experiment_on_input(
        model,
        truthful_input,
        player_ind,
        inner_product,
        regret_tolerance=None,
        n_samples=100,
        misreport_lr=0.01,
        misreport_iter=1000,
):
    truthful_allocs, truthful_payments = model(truthful_input)
    both_util = calc_agent_util(truthful_input, truthful_allocs, truthful_payments).flatten()
    truthful_util = both_util[player_ind].item()

    input_dom = convert_input_dom(truthful_input, player_ind)
    # the stuff below may be necessary for clip relu
    # converted_payment = clip_relu_convert(model.payment_head)
    # converted_alloc = clip_relu_remove(model.allocation_head)
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

    start_setup = time.time()
    mipnet.setup_model(
        input_dom, truthful_input, truthful_util, use_obj_function=True, player_ind=player_ind
    )
    end_setup = time.time()

    start_solve = time.time()
    point_was_found, result_tuple, num_states = mipnet.solve(input_dom, truthful_util)
    end_solve = time.time()

    # we should return a dict containing:
    # truthful point
    # truthful util
    # better point if it was found
    # better util if it was found
    # regret tolerance

    results_dict = {}
    results_dict["truthful_util"] = truthful_util
    results_dict["truthful_input"] = truthful_input
    results_dict["truthful_allocs"] = truthful_allocs.cpu().detach()
    results_dict["truthful_payments"] = truthful_payments.cpu().detach()
    results_dict["regret_tolerance"] = regret_tolerance
    results_dict["num_states"] = num_states
    results_dict["setup_time"] = end_setup - start_setup
    results_dict["solve_time"] = end_solve - start_solve
    results_dict["agent"] = player_ind


    if True:
        var_results = []
        for v in mipnet.gurobi_vars[0]:
            var_results.append(v.x)
        better_bid = torch.tensor(var_results).reshape(model.n_agents, model.n_items)
        results_dict["better_bid"] = better_bid

        if inner_product:
            results_dict["gurobi_better_frac_payments"] = [
                v.x for v in mipnet.payment_gurobi_vars[-1]
            ]
            results_dict["gurobi_final_payment"] = mipnet.final_player_payment.getValue()
        else:
            results_dict["gurobi_better_payments"] = [
                v.x for v in mipnet.payment_gurobi_vars[-1]
            ]
            results_dict["gurobi_final_payment"] = results_dict["gurobi_better_payments"][player_ind]
        results_dict["gurobi_better_allocs"] = [
            v.x for v in np.array(mipnet.allocation_gurobi_vars[-1]).flatten()
        ]

        results_dict["better_util_gurobi"] = mipnet.final_util_expr.getValue()

        better_allocs, better_payments = model(better_bid)
        better_util = (calc_agent_util(truthful_input, better_allocs, better_payments).flatten())[player_ind].item()

        results_dict["better_allocs"] = better_allocs.cpu().detach()
        results_dict["better_payments"] = better_payments.cpu().detach().flatten()
        results_dict["better_payment_player"] = results_dict["better_payments"][player_ind].item()
        results_dict["better_util"] = better_util

        results_dict["regret"] = better_util - truthful_util

        # randomly sample some points on the original network, to test that none of them beat
        # optimized util.
        max_random_util = 0.0
        for i in range(n_samples):
            # create new point altering only player i's bid
            new_point = truthful_input.clone()
            new_point[player_ind, :] = torch.rand_like(truthful_input[player_ind, :])
            new_alloc, new_payments = model(new_point)
            new_util = (calc_agent_util(truthful_input, new_alloc, new_payments).flatten())[player_ind].item()
            if new_util > max_random_util:
                max_random_util = new_util

        results_dict["max_random_util"] = max_random_util

        results_dict["misreport_lr"] = misreport_lr
        results_dict["misreport_iter"] = misreport_iter

        batch_truthful_input = truthful_input.unsqueeze(0)  # add batch dim
        misreport_batch = batch_truthful_input.clone().detach()
        optimize_misreports(
            model,
            batch_truthful_input,
            misreport_batch,
            misreport_iter=misreport_iter,
            lr=misreport_lr,
        )
        print(misreport_batch)
        misreport_util = (tiled_misreport_util(
            misreport_batch,batch_truthful_input, model
        ).flatten())[player_ind]
        results_dict["misreport_util"] = misreport_util.flatten()[0].item()
        print(results_dict["misreport_util"] - better_util)

    return results_dict


def experiment_and_save(model_name, presampled_points, agent=0):
    path = f"model/{model_name}.pt"
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    curr_activation = checkpoint['arch']['p_activation']
    inner_product = curr_activation == "frac_sigmoid_linear"
    if not inner_product:
        checkpoint['arch']['p_activation'] = 'full_relu_clipped' # add on clip relu otherwise
    model1 = RegretNet(**checkpoint['arch']).to('cpu')
    model1.load_state_dict(checkpoint['state_dict'])

    experiment_results = []
    random_points = presampled_points[(model1.n_agents, model1.n_items)]
    num_samples = len(random_points)

    for sample in range(num_samples):
        truthful_valuations = random_points[sample]
        print(sample)
        print(truthful_valuations)
        experiment_result = experiment_on_input(model1, truthful_valuations, agent, inner_product)
        experiment_results.append(experiment_result)

    return experiment_results


def process_and_plot_results(experiment_results, dir_name, model_name, exp_time):

    os.makedirs(dir_name, exist_ok=True)
    plt.hist([r["solve_time"] for r in experiment_results])
    plt.title("solve time")
    plt.savefig(f"{dir_name}/solve_time_{model_name}_{exp_time}.eps")
    plt.savefig(f"{dir_name}/solve_time_{model_name}_{exp_time}.png")
    plt.close()
    plt.hist([r.get("regret", 0.0) for r in experiment_results])
    plt.title("regret")
    plt.savefig(f"{dir_name}/regret_{model_name}_{exp_time}.eps")
    plt.savefig(f"{dir_name}/regret_{model_name}_{exp_time}.png")
    plt.close()
    plt.hist([r.get("misreport_util", 0.0) - r.get("better_util", 0.0) for r in experiment_results])
    plt.title("PGD vs. MIP regret difference")
    plt.savefig(f"{dir_name}/util_difference_{model_name}_{exp_time}.eps")
    plt.savefig(f"{dir_name}/util_difference_{model_name}_{exp_time}.png")
    plt.close()

    all_util_diffs = []
    all_alloc_max_diffs = []
    all_pay_diffs = []
    for exp_dict in experiment_results:
        gurobi_allocs = np.array(exp_dict["gurobi_better_allocs"])
        gurobi_util = exp_dict["better_util_gurobi"]
        gurobi_pay = exp_dict["gurobi_final_payment"]
        diff = (exp_dict["better_allocs"].flatten().cpu().numpy() - gurobi_allocs).max()
        all_alloc_max_diffs.append(diff)
        all_util_diffs.append(gurobi_util - exp_dict["better_util"])
        all_pay_diffs.append(gurobi_pay - exp_dict["better_payment_player"])
    with open(f"{dir_name}/{model_name}_{exp_time}.pickle", "wb") as fast_results:
        pickle.dump(experiment_results, fast_results)
    with open(f"{dir_name}/{model_name}_{exp_time}_maxdiffs.txt", "w") as maxdiff_file:
        print('max overall util diff', np.max(np.abs(all_util_diffs)), file=maxdiff_file)
        print('max overall alloc diff', np.max(np.abs(all_alloc_max_diffs)), file=maxdiff_file)
        print('max overall payment diff', np.max(np.abs(all_pay_diffs)), file=maxdiff_file)


model_names = ['1x2_sparsemax_linearpmt_distill_fast', '2x2_sparsemax_linearpmt_distill_fast', '2x2_sparsemax_in_linsigpmt_scratch_fast', '1x2_sparsemax_in_linsigpmt_scratch_fast']
# model_names = ['1x2_sparsemax_in_linsigpmt_scratch', '2x2_sparsemax_in_linsigpmt_scratch', '1x2_sparsemax_in_linsigpmt_scratch_fast', '2x2_sparsemax_in_linsigpmt_scratch_fast']
# model_names = ['1x2_sparsemax_linearpmt_distill_fast', '2x2_sparsemax_linearpmt_distill_fast']
# model_names = ['2x2_sparsemax_linearpmt_distill_fast']
# model_names = ['2x2_sparsemax_linearpmt_distill_fast', '2x2_sparsemax_in_linsigpmt_scratch_fast','2x2_sparsemax_in_linsigpmt_scratch']
# model_names = ['2x2_sparsemax_in_linsigpmt_scratch']
# model_names = ['2x3_sparsemax_in_linsigpmt_scratch_fast', '2x3_sparsemax_linearpmt_distill_fast','3x2_sparsemax_in_linsigpmt_scratch_fast', '3x2_sparsemax_linearpmt_distill_fast','3x3_sparsemax_in_linsigpmt_scratch_fast', '3x3_sparsemax_linearpmt_distill_fast']
# model_names = ['2x3_sparsemax_linearpmt_distill_fast', '3x2_sparsemax_linearpmt_distill_fast', '3x3_sparsemax_linearpmt_distill_fast']
# model_names = ['1x2_sparsemax_linearpmt_distill_fast', '1x2_sparsemax_linearpmt_distill_fast_4l','1x2_sparsemax_linearpmt_distill_fast_5l']

results_dict = {}
regrets_dict = {}
output_prefix = 'clipping_scaling/' # don't forget to include trailing slash here
num_samples = 1000
agent=0
presampled_points = {}
presampled_points[(1,2)] = [torch.rand(1, 2) for _ in range(num_samples)]
presampled_points[(2,2)] = [torch.rand(2, 2) for _ in range(num_samples)]
presampled_points[(3,2)] = [torch.rand(3, 2) for _ in range(num_samples)]
presampled_points[(2,3)] = [torch.rand(2, 3) for _ in range(num_samples)]
presampled_points[(3,3)] = [torch.rand(3, 3) for _ in range(num_samples)]

for model_name in model_names:
    dir_name = f"{output_prefix}{model_name}_{EXP_TIME}"
    experiment_results = experiment_and_save(model_name, presampled_points, agent=agent)
    regrets = [r.get("regret", 0.0) for r in experiment_results]
    results_dict[model_name] = experiment_results
    regrets_dict[model_name] = regrets
    process_and_plot_results(experiment_results, dir_name, model_name, EXP_TIME)
