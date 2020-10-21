# MIT License
#
# Copyright (c) 2019 oval-group
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# initial version taken from https://github.com/oval-group/GNN_branching

import gurobipy as grb
import numpy as np
import torch

from torch import nn

from regretnet import ibp
from regretnet.mipcertify.modules import View, Flatten
from regretnet.mipcertify.network_linear_approximation import LinearizedNetwork

def strip_view(network_layers):
    # we should probably try to put minimal logic in here to strip off the layers that need to be stripped off.
    if isinstance(network_layers[-1], ibp.View):
        return network_layers[:-1]
    elif (isinstance(network_layers[-1], ibp.View_Cut) and
        isinstance(network_layers[-2], ibp.Sparsemax) and
        isinstance(network_layers[-3], ibp.View)):
        return network_layers[:-3]
    else:
        return network_layers

class MIPNetwork:

    def __init__(self, base_layers, payment_layers, allocation_layers, n_agents, n_items, fractional_payment=False):
        '''
        layers: A list of Pytorch layers containing only Linear/ReLU/MaxPools
        '''
        self.layers = base_layers
        self.net = nn.Sequential(*base_layers)
        self.n_agents = n_agents
        self.n_items = n_items

        self.payment_layers = payment_layers
        self.payment_head = nn.Sequential(*payment_layers)

        self.allocation_layers = allocation_layers
        self.allocation_head = nn.Sequential(*allocation_layers)

        # Initialize a LinearizedNetwork object to determine the lower and
        # upper bounds at each layer.
        self.lin_net = LinearizedNetwork(base_layers, payment_layers, strip_view(allocation_layers))

        self.fractional_payment = fractional_payment

    def solve(self, inp_domain, truthful_util, timeout=None):
        '''
        inp_domain: Tensor containing in each row the lower and upper bound
                    for the corresponding dimension

        Returns:
        sat     : boolean indicating whether the MIP is satisfiable.
        solution: Feasible point if the MIP is satisfiable,
                  None otherwise.
        timeout : Maximum allowed time to run, if is not None
        '''
        if self.lower_bounds[-1].min() > 0:
            raise NotImplementedError("lower bounds don't yet reach all the way through")
            print("Early stopping")
            # The problem is infeasible, and we haven't setup the MIP
            return (False, None, 0)

        if timeout is not None:
            self.model.setParam('TimeLimit', timeout)

        if self.check_obj_value_callback:
            raise NotImplementedError("this callback is not fixed yet")
            def early_stop_cb(model, where):
                if where == grb.GRB.Callback.MIP:
                    best_bound = model.cbGet(grb.GRB.Callback.MIP_OBJBND)
                    if best_bound > 0:
                        model.terminate()

                if where == grb.GRB.Callback.MIPNODE:
                    nodeCount = model.cbGet(grb.GRB.Callback.MIPNODE_NODCNT)
                    if (nodeCount % 100) == 0:
                        print(f"Running Nb states visited: {nodeCount}")

                if where == grb.GRB.Callback.MIPSOL:
                    obj = model.cbGet(grb.GRB.Callback.MIPSOL_OBJ)
                    if obj < 0:
                        # Does it have a chance at being a valid
                        # counter-example?

                        # Check it with the network
                        input_vals = model.cbGetSolution(self.gurobi_vars[0])

                        with torch.no_grad():
                            if isinstance(input_vals, list):
                                inps = torch.Tensor(input_vals).view(1, -1)
                            else:
                                assert isinstance(input_vals, grb.tupledict)
                                inps = torch.Tensor([val for val in input_vals.values()])
                                inps = inps.view((1,) + self.lower_bounds[0].shape)
                            out = self.net(inps).squeeze()
                            # In case there is several output to the network, get the minimum one.
                            out = out.min().item()

                        if out < 0:
                            model.terminate()
        else:
            def early_stop_cb(model, where):
                if where == grb.GRB.Callback.MIPNODE:
                    nodeCount = model.cbGet(grb.GRB.Callback.MIPNODE_NODCNT)
                    if (nodeCount % 100) == 0:
                        print(f"Running Nb states visited: {nodeCount}")

        self.model.optimize(early_stop_cb)
        nb_visited_states = self.model.nodeCount

        if self.model.status is grb.GRB.INFEASIBLE:
            # Infeasible: No solution
            print("Infeasible.")
            return (False, None, nb_visited_states)
        elif self.model.status is grb.GRB.OPTIMAL:
            print("Optimal.")
            # There is a feasible solution. Return the feasible solution as well.
            len_inp = len(self.gurobi_vars[0])

            # Get the input that gives the feasible solution.
            #input_vals = self.model.cbGetSolution(self.gurobi_vars[0])
            #inps = torch.Tensor([val for val in input_vals.values()])
            #inps = inps.view((1,) + self.lower_bounds[0].shape)
            optim_val = self.final_util_expr.getValue()

            return ((truthful_util - optim_val) < 0, (None, optim_val), nb_visited_states)
        elif self.model.status is grb.GRB.INTERRUPTED:
            print("Interrupted.")
            obj_bound = self.model.ObjBound

            if obj_bound > 0:
                return (False, None, nb_visited_states)
            else:
                # There is a feasible solution. Return the feasible solution as well.
                len_inp = len(self.gurobi_vars[0])

                # Get the input that gives the feasible solution.
                inp = torch.Tensor(len_inp)
                if isinstance(self.gurobi_vars[0], list):
                    for idx, var in enumerate(self.gurobi_vars[0]):
                        inp[idx] = var.x
                else:
                    #assert isinstance(self.gurobi_vars[0], grb.tupledict)
                    inp = torch.zeros_like(self.lower_bounds[0])
                    for idx, var in self.gurobi_vars[0].items():
                        inp[idx] = var.x
                optim_val = self.final_util_expr.getValue()
            return ((truthful_util - optim_val) < 0, (inp, optim_val), nb_visited_states)
        elif self.model.status is grb.GRB.TIME_LIMIT:
            # We timed out, return a None Status
            return (None, None, nb_visited_states)
        else:
            raise Exception("Unexpected Status code")

    def tune(self, param_outfile, tune_timeout):
        self.model.Params.tuneOutput = 1
        self.model.Params.tuneTimeLimit = tune_timeout
        self.model.tune()

        # Get the best set of parameters
        self.model.getTuneResult(0)

        self.model.write(param_outfile)

    def do_interval_analysis(self, inp_domain):
        self.lower_bounds = []
        self.upper_bounds = []

        self.lower_bounds.append(inp_domain.select(-1, 0))
        self.upper_bounds.append(inp_domain.select(-1, 1))
        layer_idx = 1
        current_lb = self.lower_bounds[-1]
        current_ub = self.upper_bounds[-1]
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                pos_weights = torch.clamp(layer.weight, min=0)
                neg_weights = torch.clamp(layer.weight, max=0)

                new_layer_lb = torch.mv(pos_weights, current_lb) + \
                               torch.mv(neg_weights, current_ub) + \
                               layer.bias
                new_layer_ub = torch.mv(pos_weights, current_ub) + \
                               torch.mv(neg_weights, current_lb) + \
                               layer.bias
                self.lower_bounds.append(new_layer_lb)
                self.upper_bounds.append(new_layer_ub)
                current_lb = new_layer_lb
                current_ub = new_layer_ub
            elif isinstance(layer, nn.ReLU):
                current_lb = torch.clamp(current_lb, min=0)
                current_ub = torch.clamp(current_ub, min=0)
            elif type(layer) == View:
                continue
            elif type(layer) == Flatten:
                current_lb = current_lb.view(-1)
                current_ub = current_ub.view(-1)
            else:
                raise NotImplementedError


    def setup_model(self, inp_domain,
                    truthful_input,
                    truthful_util,
                    regret_tolerance=0.0001,
                    use_obj_function=False,
                    bounds="opt",
                    parameter_file=None,
                    player_ind=None):
        '''
        inp_domain: Tensor containing in each row the lower and upper bound
                    for the corresponding dimension

        optimal: If False, don't use any objective function, simply add a constraint on the output
                 If True, perform optimization and use callback to interrupt the solving when a
                          counterexample is found
        bounds: string, indicate what type of method should be used to get the intermediate bounds
        parameter_file: Load a set of parameters for the MIP solver if a path is given.

        Setup the model to be optimized by Gurobi
        '''

        if player_ind is None:
            assert self.n_agents == 1, "Must specify player_ind for >1 agent"
            self.player_ind = 0
        else:
            self.player_ind = player_ind

        if bounds == "opt":
            # First use define_linear_approximation from LinearizedNetwork to
            # compute upper and lower bounds to be able to define Ms
            self.lin_net.define_linear_approximation(inp_domain)

            self.lower_bounds = list(map(torch.Tensor, self.lin_net.lower_bounds))
            self.upper_bounds = list(map(torch.Tensor, self.lin_net.upper_bounds))
            self.payment_lower_bounds = list(map(torch.Tensor, self.lin_net.payment_lower_bounds))
            self.payment_upper_bounds = list(map(torch.Tensor, self.lin_net.payment_upper_bounds))
            self.allocation_lower_bounds = list(map(torch.Tensor, self.lin_net.allocation_lower_bounds))
            self.allocation_upper_bounds = list(map(torch.Tensor, self.lin_net.allocation_upper_bounds))
        elif bounds == "interval":
            raise NotImplementedError("interval stuff is currently not working")
            self.do_interval_analysis(inp_domain)
            if self.lower_bounds[-1][0] > 0:
                # The problem is already guaranteed to be infeasible,
                # Let's not waste time setting up the MIP
                return
        else:
            raise NotImplementedError("Unknown bound computation method.")

        self.gurobi_vars = []
        self.model = grb.Model()
        self.model.setParam('OutputFlag', False)
        self.model.setParam('Threads', 1)
        self.model.setParam('DualReductions', 0)
        if parameter_file is not None:
            self.model.read(parameter_file)

        self.zero_var = self.model.addVar(lb=0, ub=0, obj=0,
                                          vtype=grb.GRB.CONTINUOUS,
                                          name=f'zero')

        # First add the input variables as Gurobi variables.
        if inp_domain.dim() == 2:
            inp_gurobi_vars = self.model.addVars([i for i in range(inp_domain.numel() // 2)],
                                                 lb=self.lower_bounds[0],
                                                 ub=self.upper_bounds[0],
                                                 name='inp')
            inp_gurobi_vars = [var for key, var in inp_gurobi_vars.items()]
        else:
            raise Exception(f"input shape is {inp_domain.shape} but it should be upper and lower bounds for a flat linear input (i.e. N x 2)")
        self.gurobi_vars.append(inp_gurobi_vars)

        self.construct_model_layers(self.gurobi_vars, self.layers, self.lower_bounds, self.upper_bounds, var_name_str='trunk')

        self.payment_gurobi_vars = []
        self.payment_gurobi_vars.append(self.gurobi_vars[-1]) # the inputs to payment are the final ReLUs of trunk

        self.construct_model_layers(self.payment_gurobi_vars, self.payment_layers, self.payment_lower_bounds, self.payment_upper_bounds, var_name_str='payment')

        self.allocation_gurobi_vars = []
        self.allocation_gurobi_vars.append(self.gurobi_vars[-1])
        self.construct_model_layers(self.allocation_gurobi_vars, self.allocation_layers, self.allocation_lower_bounds, self.allocation_upper_bounds, var_name_str='allocation')

        final_alloc = self.allocation_gurobi_vars[-1]
        final_player_alloc = final_alloc[self.player_ind, :]
        if not self.fractional_payment:
            self.final_player_payment = self.payment_gurobi_vars[-1][self.player_ind]
        else:
            shaped_input_vars = np.reshape(np.array(self.gurobi_vars[0]), (self.n_agents, self.n_items))
            player_input_val = shaped_input_vars[self.player_ind, :]
            frac_payment = self.payment_gurobi_vars[-1][self.player_ind]
            alloc_value_expr = grb.quicksum(player_input_val[i]*final_player_alloc[i] for i in range(self.n_items))
            alloc_value = self.model.addVar(name='player_alloc_value')
            self.model.addConstr(alloc_value == alloc_value_expr)
            self.final_player_payment = frac_payment*alloc_value
            self.model.setParam("NonConvex", 2) # needed for quadratic equality constraints (Gurobi 9.0 only)


        player_truthful_input = truthful_input[self.player_ind, :]
        self.final_util_expr = grb.LinExpr(player_truthful_input, final_player_alloc) - self.final_player_payment
        if not use_obj_function:
            self.model.addConstr(self.final_util_expr >= (truthful_util + regret_tolerance))
            self.model.setObjective(0, grb.GRB.MAXIMIZE)
            self.check_obj_value_callback = False
        else:
            # maximize the final utility
            self.model.setObjective(self.final_util_expr, grb.GRB.MAXIMIZE)

            # TODO set this to True and fix the callback code
            # it's not clear that we actually want to do callbacks for this case though
            # for our application we may want to find the worst violation, as opposed to early stopping
            # if we find any violation, as long as this is not too slow.
            self.check_obj_value_callback = False

        # Optimize the model.
        self.model.update()
        #self.model.write('new_debug.lp')

    def construct_model_layers(self, gurobi_vars, layers, lower_bounds, upper_bounds, var_name_str=''):
        layer_idx = 1 # this starts at 1 -- i.e. the second element of lower_bounds/upper_bounds (0 is input)
                      # and is only incremented after hitting a ReLU. Thus passing the lower_bound and not
                      # relu_lower_bounds from lin_net is correct.
        for layer in layers:
            if isinstance(layer, nn.Linear):
                layer_nb_out = layer.out_features
                pre_vars = gurobi_vars[-1]
                if isinstance(pre_vars, grb.tupledict):
                    pre_vars = [var for key, var in sorted(pre_vars.items())]
                # Build all the outputs of the linear layer
                new_vars = self.model.addVars([i for i in range(layer_nb_out)],
                                              lb=lower_bounds[layer_idx],
                                              ub=upper_bounds[layer_idx],
                                              name=f'zhat{layer_idx}_{var_name_str}')
                new_layer_gurobi_vars = [var for key, var in new_vars.items()]
                self.model.addConstrs(
                    ((grb.LinExpr(layer.weight[neuron_idx, :], pre_vars)
                      + layer.bias[neuron_idx].item()) == new_vars[neuron_idx]
                     for neuron_idx in range(layer.out_features)),
                    name=f'lay{layer_idx}_{var_name_str}'
                )
            elif isinstance(layer, nn.ReLU):
                pre_lbs = lower_bounds[layer_idx]
                pre_ubs = upper_bounds[layer_idx]
                if isinstance(gurobi_vars[-1], grb.tupledict):
                    amb_mask = (pre_lbs < 0) & (pre_ubs > 0)
                    if amb_mask.sum().item() != 0:
                        to_new_preubs = pre_ubs[amb_mask]
                        to_new_prelbs = pre_lbs[amb_mask]

                        new_var_idxs = torch.nonzero((pre_lbs < 0) & (pre_ubs > 0)).numpy().tolist()
                        new_var_idxs = [tuple(idxs) for idxs in new_var_idxs]
                        new_layer_gurobi_vars = self.model.addVars(new_var_idxs,
                                                                   lb=0,
                                                                   ub=to_new_preubs,
                                                                   name=f'z{layer_idx}_{var_name_str}')
                        new_binary_vars = self.model.addVars(new_var_idxs,
                                                             lb=0, ub=1,
                                                             vtype=grb.GRB.BINARY,
                                                             name=f'delta{layer_idx}_{var_name_str}')

                        flat_new_vars = [new_layer_gurobi_vars[idx] for idx in new_var_idxs]
                        flat_binary_vars = [new_binary_vars[idx] for idx in new_var_idxs]
                        pre_amb_vars = [gurobi_vars[-1][idx] for idx in new_var_idxs]

                        # C1: Superior to 0
                        # C2: Add the constraint that it's superior to the inputs
                        self.model.addConstrs(
                            (flat_new_vars[idx] >= pre_amb_vars[idx]
                             for idx in range(len(flat_new_vars))),
                            name=f'ReLU_lb{layer_idx}_{var_name_str}'
                        )
                        # C3: Below binary*upper_bound
                        self.model.addConstrs(
                            (flat_new_vars[idx] <= to_new_preubs[idx].item() * flat_binary_vars[idx]
                             for idx in range(len(flat_new_vars))),
                            name=f'ReLU{layer_idx}_ub1-{var_name_str}'
                        )
                        # C4: Below binary*lower_bound
                        self.model.addConstrs(
                            (flat_new_vars[idx] <= (pre_amb_vars[idx]
                                                    - to_new_prelbs[idx].item() * (1 - flat_binary_vars[idx]))
                             for idx in range(len(flat_new_vars))),
                            name=f'ReLU{layer_idx}_ub2-{var_name_str}'
                        )
                    else:
                        new_layer_gurobi_vars = grb.tupledict()

                    for pos in torch.nonzero(pre_lbs >= 0).numpy().tolist():
                        pos = tuple(pos)
                        new_layer_gurobi_vars[pos] = gurobi_vars[-1][pos]
                    for pos in torch.nonzero(pre_ubs <= 0).numpy().tolist():
                        new_layer_gurobi_vars[tuple(pos)] = self.zero_var
                else:
                    assert isinstance(gurobi_vars[-1][0], grb.Var)

                    amb_mask = (pre_lbs < 0) & (pre_ubs > 0)
                    if amb_mask.sum().item() == 0:
                        pass
                        # print("WARNING: No ambiguous ReLU at a layer")
                    else:
                        to_new_preubs = pre_ubs[amb_mask]
                        new_var_idxs = torch.nonzero(amb_mask).squeeze(1).numpy().tolist()
                        new_vars = self.model.addVars(new_var_idxs,
                                                      lb=0,
                                                      ub=to_new_preubs,
                                                      name=f'z{layer_idx}_{var_name_str}')
                        new_binary_vars = self.model.addVars(new_var_idxs,
                                                             lb=0, ub=1,
                                                             vtype=grb.GRB.BINARY,
                                                             name=f'delta{layer_idx}_{var_name_str}')

                        # C1: Superior to 0
                        # C2: Add the constraint that it's superior to the inputs
                        self.model.addConstrs(
                            (new_vars[idx] >= gurobi_vars[-1][idx]
                             for idx in new_var_idxs),
                            name=f'ReLU_lb{layer_idx}_{var_name_str}'
                        )
                        # C3: Below binary*upper_bound
                        self.model.addConstrs(
                            (new_vars[idx] <= pre_ubs[idx].item() * new_binary_vars[idx]
                             for idx in new_var_idxs),
                            name=f'ReLU{layer_idx}_ub1-{var_name_str}'
                        )
                        # C4: Below binary*lower_bound
                        self.model.addConstrs(
                            (new_vars[idx] <= (gurobi_vars[-1][idx]
                                               - pre_lbs[idx].item() * (1 - new_binary_vars[idx]))
                             for idx in new_var_idxs),
                            name=f'ReLU{layer_idx}_ub2-{var_name_str}'
                        )

                    # Get all the variables in a list, such that we have the
                    # output of the layer
                    new_layer_gurobi_vars = []
                    new_idx = 0
                    for idx in range(layer_nb_out):
                        if pre_lbs[idx] >= 0:
                            # Pass through variable
                            new_layer_gurobi_vars.append(gurobi_vars[-1][idx])
                        elif pre_ubs[idx] <= 0:
                            # Blocked variable
                            new_layer_gurobi_vars.append(self.zero_var)
                        else:
                            new_layer_gurobi_vars.append(new_vars[idx])
                layer_idx += 1
            elif isinstance(layer, ibp.Sparsemax):
                pre_vars = gurobi_vars[-1] # these pre_vars should be of dim 2
                assert len(pre_vars.shape) == 2
                # note we have to subtract 1 from sparsemax dim to remove batch dimension
                dim = layer.dim - 1
                if dim == 0:
                    # slide over rows
                    other_dim = 1
                    d = pre_vars.shape[dim]
                    sparsemax_grb_vars = []
                    for col_dim in range(pre_vars.shape[other_dim]): # iterate over columns
                        curr_column = pre_vars[:, col_dim]
                        # sparsemax with these input variables

                        # create sparsemax out_vars, set them to sum to 1
                        sparsemax_z = self.model.addVars(list(range(d)), lb=0, ub=1, name=f'sparsemax_col{col_dim}_{var_name_str}')
                        sparsemax_grb_vars.append([sparsemax_z[i] for i in range(d)])
                        self.model.addConstr(grb.quicksum(sparsemax_z) == 1, name=f'sparsemax_col{col_dim}_sum_{var_name_str}')

                        mu1 = self.model.addVars(list(range(d)), lb=0, name=f'sparsemax_mu1_{col_dim}_{var_name_str}')
                        mu2 = self.model.addVars(list(range(d)), lb=0, name=f'sparsemax_mu2_{col_dim}_{var_name_str}')
                        lam = self.model.addVar(lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY, name=f'sparsemax_lambda_{col_dim}_{var_name_str}')

                        self.model.addConstrs( (((sparsemax_z[i] - curr_column[i]) + mu1[i] - mu2[i] + lam == 0) for i in range(d)), name=f'sparsemax_stationarity_{col_dim}_{var_name_str}')

                        # add z minus 1 and negative z
                        # not sure if we actually need negative z
                        zminusone = self.model.addVars([i for i in range(d)], lb=-1, ub=0, name=f'sparsemax_zminusone_{col_dim}_{var_name_str}')
                        self.model.addConstrs((zminusone[i] == (sparsemax_z[i] - 1) for i in range(d)), name=f'sparsemax_zminusone_{col_dim}_constr_{var_name_str}')
                        negz = self.model.addVars([i for i in range(d)], lb=-1, ub=0, name=f'sparsemax_negz_{col_dim}_{var_name_str}')
                        self.model.addConstrs((-negz[i] == sparsemax_z[i] for i in range(d)), name=f'sparsemax_negz_{col_dim}_{var_name_str}')

                        # complementary slackness
                        for i in range(d):
                            self.model.addSOS(grb.GRB.SOS_TYPE1, [zminusone[i], mu1[i]], wts=[1, 2])
                            self.model.addSOS(grb.GRB.SOS_TYPE1, [negz[i], mu2[i]], wts=[1, 2])
                    # at this point we have sparsemax_grb_vars, with 1 nested list. need to transpose for columns
                    new_layer_gurobi_vars = np.array(sparsemax_grb_vars).transpose()
                elif dim == 1:
                    raise NotImplementedError("row-wise not implemented")
                else:
                    raise NotImplementedError("sparsemax for more than 2D not implemented")

            elif isinstance(layer, ibp.View):
                pre_vars = gurobi_vars[-1] # previous activations
                viewed_vars = np.reshape(pre_vars, layer.shape[1:]) # [1:] is to drop batch dim
                new_layer_gurobi_vars = viewed_vars
            elif isinstance(layer, ibp.View_Cut):
                pre_vars = gurobi_vars[-1]
                viewed_vars = pre_vars[:-1, :]
                new_layer_gurobi_vars = viewed_vars
            elif isinstance(layer, Flatten):
                raise NotImplementedError("flatten should be manually removed right now")
            else:
                raise NotImplementedError

            gurobi_vars.append(new_layer_gurobi_vars)
