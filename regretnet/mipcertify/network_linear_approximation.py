import gurobipy as grb
import torch

from itertools import product

from regretnet import ibp
from regretnet.mipcertify.modules import View, Flatten
from torch import nn

def check_linnet_valid(network_layers):
    assert isinstance(network_layers[0], nn.Linear), "first layer must be linear"

    for i in range(1, len(network_layers)):
        if isinstance(network_layers[i], nn.ReLU):
            assert isinstance(network_layers[i-1], nn.Linear), "relu must follow linear"
        elif isinstance(network_layers[i], nn.Linear):
            assert isinstance(network_layers[i-1], nn.ReLU), "linear must follow relu"
        else:
            raise ValueError("all layers should be linear or relu")

class LinearizedNetwork:

    def __init__(self, base_layers, payment_layers, allocation_layers):
        '''
        layers: A list of Pytorch layers containing only Linear/ReLU

        Note that each Linear must be followed by ReLU.
        '''

        check_linnet_valid(base_layers)
        check_linnet_valid(payment_layers)
        check_linnet_valid(allocation_layers)
        self.layers = base_layers
        self.net = nn.Sequential(*base_layers)
        self.payment_layers = payment_layers
        self.payment_head = nn.Sequential(*payment_layers)

        self.allocation_layers = allocation_layers
        self.allocation_head = nn.Sequential(*allocation_layers)
        # Skip all gradient computation for the weights of the Net
        for param in self.net.parameters():
            param.requires_grad = False
        for param in self.payment_head.parameters():
            param.requires_grad = False

    def remove_maxpools(self, domain):
        from regretnet.mipcertify.model import reluify_maxpool, simplify_network
        if any(map(lambda x: type(x) is nn.MaxPool1d, self.layers)):
            new_layers = simplify_network(reluify_maxpool(self.layers, domain))
            self.layers = new_layers


    def get_upper_bound_random(self, domain):
        '''
        Compute an upper bound of the minimum of the network on `domain`

        Any feasible point is a valid upper bound on the minimum so we will
        perform some random testing.
        '''
        nb_samples = 1024
        nb_inp = domain.shape[:-1]
        # Not a great way of sampling but this will be good enough
        # We want to get rows that are >= 0
        sp_shape = (nb_samples, ) + nb_inp
        rand_samples = torch.Tensor(*sp_shape)
        rand_samples.uniform_(0, 1)

        domain_lb = domain.select(-1, 0).contiguous()
        domain_ub = domain.select(-1, 1).contiguous()
        domain_width = domain_ub - domain_lb

        domain_lb = domain_lb.unsqueeze(0).expand(*sp_shape)
        domain_width = domain_width.unsqueeze(0).expand(*sp_shape)

        with torch.no_grad():
            inps = domain_lb + domain_width * rand_samples
            outs = self.net(inps)

            upper_bound, idx = torch.min(outs, dim=0)

            upper_bound = upper_bound[0].item()
            ub_point = inps[idx].squeeze()

        return ub_point, upper_bound

    def get_upper_bound_pgd(self, domain):
        '''
        Compute an upper bound of the minimum of the network on `domain`

        Any feasible point is a valid upper bound on the minimum so we will
        perform some random testing.
        '''
        nb_samples = 2056
        torch.set_num_threads(1)
        nb_inp = domain.size(0)
        # Not a great way of sampling but this will be good enough
        # We want to get rows that are >= 0
        rand_samples = torch.Tensor(nb_samples, nb_inp)
        rand_samples.uniform_(0, 1)

        best_ub = float('inf')
        best_ub_inp = None

        domain_lb = domain.select(1, 0).contiguous()
        domain_ub = domain.select(1, 1).contiguous()
        domain_width = domain_ub - domain_lb

        domain_lb = domain_lb.view(1, nb_inp).expand(nb_samples, nb_inp)
        domain_width = domain_width.view(1, nb_inp).expand(nb_samples, nb_inp)

        inps = (domain_lb + domain_width * rand_samples)

        with torch.enable_grad():
            batch_ub = float('inf')
            for i in range(1000):
                prev_batch_best = batch_ub

                self.net.zero_grad()
                if inps.grad is not None:
                    inps.grad.zero_()
                inps = inps.detach().requires_grad_()
                out = self.net(inps)

                batch_ub = out.min().item()
                if batch_ub < best_ub:
                    best_ub = batch_ub
                    # print(f"New best lb: {best_lb}")
                    _, idx = out.min(dim=0)
                    best_ub_inp = inps[idx[0]]

                if batch_ub >= prev_batch_best:
                    break

                all_samp_sum = out.sum() / nb_samples
                all_samp_sum.backward()
                grad = inps.grad

                max_grad, _ = grad.max(dim=0)
                min_grad, _ = grad.min(dim=0)
                grad_diff = max_grad - min_grad

                lr = 1e-2 * domain_width / grad_diff
                min_lr = lr.min()

                step = -min_lr*grad
                inps = inps + step

                inps = torch.max(inps, domain_lb)
                inps = torch.min(inps, domain_ub)

        return best_ub_inp, best_ub

    get_upper_bound = get_upper_bound_random

    def get_lower_bound(self, domain, force_optim=False):
        '''
        Update the linear approximation for `domain` of the network and use it
        to compute a lower bound on the minimum of the output.

        domain: Tensor containing in each row the lower and upper bound for
                the corresponding dimension
        '''
        self.define_linear_approximation(domain, force_optim)
        return self.compute_lower_bound()

    def compute_lower_bound(self, node=(-1, None), upper_bound=False,
                            all_optim=False):
        '''
        Compute a lower bound of the function for the given node

        node: (optional) Index (as a tuple) in the list of gurobi variables of the node to optimize
              First index is the layer, second index is the neuron.
              For the second index, None is a special value that indicates to optimize all of them,
              both upper and lower bounds.
        upper_bound: (optional) Compute an upper bound instead of a lower bound
        all_optim: Should the bounds be computed only in the case where they are not already leading to
              non relaxed version. This option is only useful if the batch mode based on None in node is
              used.
        '''
        layer_with_var_to_opt = self.prerelu_gurobi_vars[node[0]]
        is_batch = (node[1] is None)
        if not is_batch:
            if isinstance(node[1], int):
                var_to_opt = layer_with_var_to_opt[node[1]]
            elif (isinstance(node[1], tuple) and isinstance(layer_with_var_to_opt, list)):
                # This is the nested list format
                to_query = layer_with_var_to_opt
                for idx in node[1]:
                    to_query = to_query[idx]
                var_to_opt = to_query
            else:
                raise NotImplementedError

            opt_direct = grb.GRB.MAXIMIZE if upper_bound else grb.GRB.MINIMIZE
            # We will make sure that the objective function is properly set up
            self.model.setObjective(var_to_opt, opt_direct)

            # We will now compute the requested lower bound
            self.model.update()
            self.model.optimize()
            assert self.model.status == 2, "LP wasn't optimally solved"

            return var_to_opt.X
        else:
            print("Batch Gurobi stuff")
            new_lbs = []
            new_ubs = []
            if isinstance(layer_with_var_to_opt, list):
                for var_idx, var in enumerate(layer_with_var_to_opt):
                    curr_lb = self.lower_bounds[node[0]][var_idx]
                    curr_ub = self.upper_bounds[node[0]][var_idx]
                    if (all_optim or
                        ((curr_lb < 0) and (curr_ub > 0))):

                        # Do the maximizing
                        self.model.setObjective(var, grb.GRB.MAXIMIZE)
                        self.model.update()
                        self.model.optimize()
                        assert self.model.status == 2, "LP wasn't optimally solved"
                        new_ubs.append(min(curr_ub, var.X))
                        # print(f"UB was {curr_ub}, now is {new_ubs[-1]}")
                        # Do the minimizing
                        self.model.setObjective(var, grb.GRB.MINIMIZE)
                        self.model.reset()
                        self.model.update()
                        self.model.optimize()
                        assert self.model.status == 2, "LP wasn't optimally solved"
                        new_lbs.append(max(curr_lb, var.X))
                        # print(f"LB was {curr_lb}, now is {new_lbs[-1]}")
                    else:
                        new_ubs.append(curr_ub)
                        new_lbs.append(curr_lb)
            else:
                new_lbs = self.lower_bounds[node[0]].clone()
                new_ubs = self.upper_bounds[node[0]].clone()
                bound_shape = new_lbs.shape
                for chan_idx, row_idx, col_idx in product(range(bound_shape[0]),
                                                          range(bound_shape[1]),
                                                          range(bound_shape[2])):
                    curr_lb = new_lbs[chan_idx, row_idx, col_idx]
                    curr_ub = new_ubs[chan_idx, row_idx, col_idx]
                    if (all_optim or
                        ((curr_lb < 0) and (curr_ub > 0))):
                        var = layer_with_var_to_opt[chan_idx, row_idx, col_idx]

                        # Do the maximizing
                        self.model.setObjective(var, grb.GRB.MAXIMIZE)
                        self.model.update()
                        self.model.optimize()
                        assert self.model.status == 2, "LP wasn't optimally solved"
                        new_ubs[chan_idx, row_idx, col_idx] = min(curr_ub, var.X)
                        # print(f"UB was {curr_ub}, now is {new_ubs[chan_idx, row_idx, col_idx]}")
                        # Do the minimizing
                        self.model.setObjective(var, grb.GRB.MINIMIZE)
                        self.model.reset()
                        self.model.update()
                        self.model.optimize()
                        assert self.model.status == 2, "LP wasn't optimally solved"
                        new_lbs[chan_idx, row_idx, col_idx] = max(curr_lb, var.X)
                        # print(f"LB was {curr_lb}, now is {new_lbs[chan_idx, row_idx, col_idx]}")

            return torch.tensor(new_lbs), torch.tensor(new_ubs)

    def define_linear_approximation(self, input_domain, force_optim=False):
        '''
        input_domain: Tensor containing in each row the lower and upper bound
                      for the corresponding dimension
        '''
        self.lower_bounds = []
        self.upper_bounds = []
        self.relu_lower_bounds = []
        self.relu_upper_bounds = []
        self.gurobi_vars = []
        self.prerelu_gurobi_vars = []
        # These three are nested lists. Each of their elements will itself be a
        # list of the neurons after a layer.

        self.model = grb.Model()
        self.model.setParam('OutputFlag', False)

        ## Do the input layer, which is a special case
        inp_lbs = []
        inp_ubs = []
        inp_gurobi_vars = []
        self.zero_var = self.model.addVar(lb=0, ub=0, obj=0,
                                     vtype=grb.GRB.CONTINUOUS,
                                     name=f'zero')
        if input_domain.dim() == 2:
            # This is a linear input.
            for dim, (lb, ub) in enumerate(input_domain):
                v = self.model.addVar(lb=lb, ub=ub, obj=0,
                                      vtype=grb.GRB.CONTINUOUS,
                                      name=f'inp_{dim}')
                inp_gurobi_vars.append(v)
                inp_lbs.append(lb)
                inp_ubs.append(ub)
        else:
            raise NotImplementedError("This version of the code only works for linear inputs.")
        self.model.update()

        self.lower_bounds.append(torch.tensor(inp_lbs))
        self.upper_bounds.append(torch.tensor(inp_ubs))
        self.gurobi_vars.append(inp_gurobi_vars)
        self.prerelu_gurobi_vars.append(inp_gurobi_vars)


        ## Do the other layers, computing for each of the neuron, its upper
        ## bound and lower bound
        self.construct_model_layers(self.layers, self.lower_bounds, self.upper_bounds, self.relu_lower_bounds,
                                    self.relu_upper_bounds, self.gurobi_vars, self.prerelu_gurobi_vars,
                                    force_optim=force_optim, var_name_str='trunk')


        self.payment_lower_bounds = []
        self.payment_upper_bounds = []
        self.payment_relu_lower_bounds = []
        self.payment_relu_upper_bounds = []
        self.payment_gurobi_vars = []
        self.payment_prerelu_gurobi_vars = []

        # the inputs for the payment head are the outputs of the trunk
        self.payment_lower_bounds.append(torch.tensor(self.relu_lower_bounds[-1])) # outputs after ReLU
        self.payment_upper_bounds.append(torch.tensor(self.relu_upper_bounds[-1]))
        self.payment_gurobi_vars.append(self.gurobi_vars[-1])
        # technically these are post-relu, but we really want them to just be the inputs (??)
        self.payment_prerelu_gurobi_vars.append(self.gurobi_vars[-1])
#
        self.construct_model_layers(self.payment_layers, self.payment_lower_bounds, self.payment_upper_bounds,
                                    self.payment_relu_lower_bounds, self.payment_relu_upper_bounds,
                                    self.payment_gurobi_vars, self.payment_prerelu_gurobi_vars, force_optim=force_optim, var_name_str='payment')

        self.allocation_lower_bounds = []
        self.allocation_upper_bounds = []
        self.allocation_relu_lower_bounds = []
        self.allocation_relu_upper_bounds = []
        self.allocation_gurobi_vars = []
        self.allocation_prerelu_gurobi_vars = []

        # the inputs for the allocation head are also just the outputs of the trunk
        self.allocation_lower_bounds.append(torch.tensor(self.relu_lower_bounds[-1])) # outputs after ReLU
        self.allocation_upper_bounds.append(torch.tensor(self.relu_upper_bounds[-1]))
        self.allocation_gurobi_vars.append(self.gurobi_vars[-1])
        # technically these are post-relu, but we really want them to just be the inputs (??)
        self.allocation_prerelu_gurobi_vars.append(self.gurobi_vars[-1])

        self.construct_model_layers(self.allocation_layers, self.allocation_lower_bounds, self.allocation_upper_bounds,
                                    self.allocation_relu_lower_bounds, self.allocation_relu_upper_bounds,
                                    self.allocation_gurobi_vars, self.allocation_prerelu_gurobi_vars, force_optim=force_optim,
                                    var_name_str='allocation')

        self.model.update()

    def construct_model_layers(self, network_layers, lower_bounds, upper_bounds, relu_lower_bounds, relu_upper_bounds,
                               gurobi_vars, prerelu_gurobi_vars, force_optim=False, var_name_str=''):
        # note: a lot of these input variables are lists that are mutated by this method. keep this in mind.
        # TODO actual docstring here
        layer_idx = 1
        for layer in network_layers:
            is_final = (layer is network_layers[-1])
            new_layer_lb = []
            new_layer_ub = []
            new_layer_gurobi_vars = []
            if isinstance(layer, nn.Linear):
                pre_lb = lower_bounds[-1]
                pre_ub = upper_bounds[-1]
                pre_vars = gurobi_vars[-1]
                if pre_lb.dim() > 1:
                    raise NotImplementedError("This version of the code only works for linear inputs.")
                if layer_idx > 1:
                    # The previous bounds are from a ReLU
                    pre_lb = torch.clamp(pre_lb, 0, None)
                    pre_ub = torch.clamp(pre_ub, 0, None)
                pos_w = torch.clamp(layer.weight, 0, None)
                neg_w = torch.clamp(layer.weight, None, 0)
                out_lbs = pos_w @ pre_lb + neg_w @ pre_ub + layer.bias
                out_ubs = pos_w @ pre_ub + neg_w @ pre_lb + layer.bias

                for neuron_idx in range(layer.weight.size(0)):
                    lin_expr = layer.bias[neuron_idx].item()
                    coeffs = layer.weight[neuron_idx, :]
                    lin_expr += grb.LinExpr(coeffs, pre_vars)

                    out_lb = out_lbs[neuron_idx].item()
                    out_ub = out_ubs[neuron_idx].item()
                    v = self.model.addVar(lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY,
                                          obj=0,
                                          vtype=grb.GRB.CONTINUOUS,
                                          name=f'lay{layer_idx}_{neuron_idx}_{var_name_str}')
                    self.model.addConstr(v == lin_expr)
                    self.model.update()

                    should_opt = (force_optim
                                  or is_final
                                  or ((layer_idx > 1) and (out_lb < 0) and (out_ub > 0))
                                  )
                    if should_opt:
                        self.model.setObjective(v, grb.GRB.MINIMIZE)
                        self.model.optimize()
                        assert self.model.status == 2, "LP wasn't optimally solved"
                        # We have computed a lower bound
                        out_lb = v.X

                        # Let's now compute an upper bound
                        self.model.setObjective(v, grb.GRB.MAXIMIZE)
                        self.model.update()
                        self.model.reset()
                        self.model.optimize()
                        assert self.model.status == 2, "LP wasn't optimally solved"
                        out_ub = v.X

                    new_layer_lb.append(out_lb)
                    new_layer_ub.append(out_ub)
                    new_layer_gurobi_vars.append(v)
                lower_bounds.append(torch.tensor(new_layer_lb))
                upper_bounds.append(torch.tensor(new_layer_ub))
                prerelu_gurobi_vars.append(new_layer_gurobi_vars)
            elif isinstance(layer, nn.ReLU):
                assert isinstance(gurobi_vars[-1][0], grb.Var)
                should_opt = force_optim or is_final
                relu_layer_lb = []
                relu_layer_ub = []
                for neuron_idx, pre_var in enumerate(gurobi_vars[-1]):
                    pre_lb = lower_bounds[-1][neuron_idx]
                    pre_ub = upper_bounds[-1][neuron_idx]

                    v = self.model.addVar(lb=max(0, pre_lb),
                                          ub=max(0, pre_ub),
                                          obj=0,
                                          vtype=grb.GRB.CONTINUOUS,
                                          name=f'ReLU{layer_idx}_{neuron_idx}_{var_name_str}')
                    if pre_lb >= 0 and pre_ub >= 0:
                        # The ReLU is always passing
                        self.model.addConstr(v == pre_var)
                        lb = pre_lb
                        ub = pre_ub
                    elif pre_lb <= 0 and pre_ub <= 0:
                        lb = 0
                        ub = 0
                        # No need to add an additional constraint that v==0
                        # because this will be covered by the bounds we set on
                        # the value of v.
                    else:
                        lb = 0
                        ub = pre_ub
                        self.model.addConstr(v >= pre_var)

                        slope = pre_ub / (pre_ub - pre_lb)
                        bias = - pre_lb * slope
                        self.model.addConstr(v <= slope.item() * pre_var + bias.item())

                    if should_opt:
                        self.model.setObjective(v, grb.GRB.MINIMIZE)
                        self.model.optimize()
                        assert self.model.status == 2, "LP wasn't optimally solved"
                        # We have computed a lower bound
                        lb = v.X

                        # Let's now compute an upper bound
                        self.model.setObjective(v, grb.GRB.MAXIMIZE)
                        self.model.update()
                        self.model.reset()
                        self.model.optimize()
                        assert self.model.status == 2, "LP wasn't optimally solved"
                        ub = v.X

                    relu_layer_lb.append(lb)
                    relu_layer_ub.append(ub)
                    new_layer_gurobi_vars.append(v)
                relu_lower_bounds.append(relu_layer_lb)
                relu_upper_bounds.append(relu_layer_ub)
            elif isinstance(layer, ibp.Sparsemax):
                raise NotImplementedError("sparsemax should be manually removed right now")
            elif isinstance(layer, ibp.View):
                raise NotImplementedError("views should be manually removed right now.")
            elif type(layer) == Flatten:
                raise NotImplementedError("views should be manually removed right now")
            else:
                raise NotImplementedError

            gurobi_vars.append(new_layer_gurobi_vars)

            layer_idx += 1
