# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:14:56 2024

@author: a.stratigakos
"""

import cvxpy as cp
import torch
from torch import nn
from cvxpylayers.torch import CvxpyLayer
from torch.utils.data import Dataset, DataLoader
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d

def to_np(x):
    return x.detach().numpy()

# Define a custom dataset
class MyDataset(Dataset):
    def __init__(self, *inputs):
        self.inputs = inputs

        # Check that all input tensors have the same length (number of samples)
        self.length = len(inputs[0])
        if not all(len(input_tensor) == self.length for input_tensor in inputs):
            raise ValueError("Input tensors must have the same length.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return tuple(input_tensor[idx] for input_tensor in self.inputs)

# Define a custom data loader
def create_data_loader(inputs, batch_size, num_workers=0, shuffle=True):
    dataset = MyDataset(*inputs)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )
    return data_loader
                
class Robust_OPF(nn.Module):        
    
    def __init__(self, num_uncertainties, num_constr, grid, UB_initial, LB_initial, c_viol = 2*1e4, regularization = 0, include_network = True, 
                 add_fixed_box = False):
        super(Robust_OPF, self).__init__()
        ''' Learning uncertainty set for robust OPF with linear recourse policy. 
            Given a Robust-OPF problem parameterized by a polyhedral uncertainty set. This class learns the paperemeters of
            the polyhedral, corresponding to a set of linear inequalities, that minimize the expected downstream cost.
            Uncertainty set Xi = {xi | H@xi <= h}, where H is (size: num_constr*num_uncertainties) and h is vector (size: num_constr)
            - Args:
                num_uncertainties: sources of uncertainty (could be tuned with budget constraint)
                num_constr: number of linear inequalities in the ucertainty set
                grid: dictionary with grid information, see separate functions to read matpower cases
                UB_initial: initial upper bounds for uncertain variables (size: num_uncertainties)
                LB_initial: initial upper bounds for uncertain variables (size: num_uncertainties)
                *** I should change this to initialize with inequalities, rather than upper/lower bounds ***
                c_viol: constraint violation penalties for real-time redispatch 
                '''
                
        Pmax = grid['Pmax']
        C_r_up = grid['C_r_up']
        C_r_down = grid['C_r_down']
        Cost = grid['Cost']    
        
        node_G = grid['node_G']
        node_L = grid['node_L']
        node_W = grid['node_W']
        PTDF = grid['PTDF']        

        self.Pmax = torch.FloatTensor(Pmax)
        self.Cost = torch.FloatTensor(Cost)
        self.C_r_up = torch.FloatTensor(C_r_up)
        self.C_r_down = torch.FloatTensor(C_r_down)
        self.node_G = torch.FloatTensor(node_G)
        self.node_L = torch.FloatTensor(node_L)
        self.node_W = torch.FloatTensor(node_W)
        self.PTDF = torch.FloatTensor(PTDF)
        self.w_exp = torch.FloatTensor(grid['w_exp'])
        self.w_cap = torch.FloatTensor(grid['w_cap'])
        
        # Initialize polytope
        
        # number of unknown net demands (here only considering wind)
        self.num_uncertainties = num_uncertainties
        self.c_viol = c_viol
        # number robust constraints to learn
        self.num_constr = num_constr
        
        # Parameters to be estimated (initialize at the extremes of the box)
        
        # Initialize with upper & lower bounds for each uncertainty (2*num_uncertainties total constraints)
        H_init = np.row_stack((-np.eye(num_uncertainties), np.eye(num_uncertainties)))
        h_init = np.row_stack((-LB_initial, UB_initial)).reshape(-1)

        #H_init = H_init + np.random.normal(loc = 0, scale = 0.1, size = H_init.shape)
        #h_init = h_init + np.random.normal(loc = 0, scale = 0.1, size = h_init.shape)
        
        #H_init = np.vstack(2*[H_init])
        #h_init = np.tile(h_init, 2)
        
        # Initialize additional constraints with some noise
        for i in range(num_constr - len(h_init)):
            
            ind = i%(2*num_uncertainties)            
            
            h_init = np.row_stack((h_init.reshape(-1,1), h_init[ind]*np.ones((1,1)) + np.random.normal(loc=0, scale=0.01) ) ).reshape(-1)
            H_init = np.row_stack((H_init, H_init[ind:ind+1,:] + np.random.normal(loc=0, scale=0.01, size = (1, H_init.shape[1] )) )) 
        #H_init = np.zeros((num_constr, num_uncertainties))
        #h_init = np.zeros(num_constr)
                
        self.H = nn.Parameter(torch.FloatTensor(H_init).requires_grad_())
        self.h = nn.Parameter(torch.FloatTensor(h_init).requires_grad_())
        
        # fix the box uncertainty on the support
        H_box = np.row_stack((-np.eye(num_uncertainties), np.eye(num_uncertainties)))
        h_box = np.row_stack((-LB_initial, UB_initial)).reshape(-1)

        self.H_fixed = torch.FloatTensor(H_box)
        self.h_fixed = torch.FloatTensor(h_box)
        
        #H_bound = np.row_stack((-np.eye(grid['n_wind']), np.eye(grid['n_wind'])))
        #h_bound = np.row_stack((-h_lb, h_ub)).reshape(-1)

        self.grid = grid
        self.regularization = regularization
        self.include_network = include_network
                
        #### Robust DCOPF Layer
        H_param = cp.Parameter((num_constr, num_uncertainties))
        h_param = cp.Parameter((num_constr))
        
        ###### Robust OPF layer for DA market clearing
        
        ### variables    
        # DA Variables
        p_G = cp.Variable((grid['n_unit']), nonneg = True)
        r_up_G = cp.Variable((grid['n_unit']), nonneg = True)
        r_down_G = cp.Variable((grid['n_unit']), nonneg = True)

        # linear decision rules
        W = cp.Variable((grid['n_unit'], num_uncertainties))

        # transmission margin
        f_margin_up = cp.Variable((grid['n_lines']), nonneg = True)
        f_margin_down = cp.Variable((grid['n_lines']), nonneg = True)

        #### DA constraints

        DA_constraints = [p_G + r_up_G<= grid['Pmax'].reshape(-1), p_G - r_down_G >= 0, 
                          W.sum(0) == np.ones(num_uncertainties), 
                          p_G.sum() + grid['w_exp'].sum() == grid['Pd'].sum(), 
                          grid['Line_Capacity'].reshape(-1) - f_margin_up == PTDF@(node_G@p_G + node_W@grid['w_exp'] - node_L@grid['Pd']),
                          grid['Line_Capacity'].reshape(-1) - f_margin_down == -PTDF@(node_G@p_G + node_W@grid['w_exp'] - node_L@grid['Pd'] ), 
                          PTDF@(node_G@p_G + node_W@grid['w_exp'] - node_L@grid['Pd'] ) <= grid['Line_Capacity'].reshape(-1), 
                          PTDF@(node_G@p_G + node_W@grid['w_exp'] - node_L@grid['Pd'] ) >= -grid['Line_Capacity'].reshape(-1)]
        

        ##### Robust constraints
        # additional variables for generator limits
        lambda_up = cp.Variable((grid['n_unit'], num_constr), nonneg = True)
        lambda_down = cp.Variable((grid['n_unit'], num_constr), nonneg = True)

        # additional variables for line margins
        lambda_f_up = cp.Variable((grid['n_lines'], num_constr), nonneg = True)
        lambda_f_down = cp.Variable((grid['n_lines'], num_constr), nonneg = True)
        
        # Reformulation of robust constraints
        # downward reserve bound/ each row are the duals to reformulate each constraints
        
        Robust_constr = [ lambda_down@H_param == W, h_param@lambda_down.T <= r_down_G, 
                         lambda_up@H_param == -W, h_param@lambda_up.T <= r_up_G, 
                         lambda_f_up@H_param == PTDF@(-node_G@W + node_W), h_param@lambda_f_up.T <= f_margin_up, 
                         lambda_f_down@H_param == -PTDF@(-node_G@W + node_W), h_param@lambda_f_down.T <= f_margin_down]

        # Objective function
        DA_cost_expr = Cost@p_G + C_r_up@r_up_G + C_r_down@r_down_G
        objective_funct = cp.Minimize( DA_cost_expr ) 

        if add_fixed_box:
            # fixed robust constraints/ on the support
            # additional variables for generator limits
            lambda_up_fix = cp.Variable((grid['n_unit'], 2*num_uncertainties), nonneg = True)
            lambda_down_fix = cp.Variable((grid['n_unit'], 2*num_uncertainties), nonneg = True)
    
            # additional variables for line margins
            lambda_f_up_fix = cp.Variable((grid['n_lines'], 2*num_uncertainties), nonneg = True)
            lambda_f_down_fix = cp.Variable((grid['n_lines'], 2*num_uncertainties), nonneg = True)
            
            fix_rob_var = [lambda_up_fix,lambda_down_fix, lambda_f_up_fix, lambda_f_down_fix]
            # Reformulation of robust constraints
            # downward reserve bound/ each row are the duals to reformulate each constraints
            
            Robust_constr_fix = [ lambda_down_fix@self.H_fixed == W, self.h_fixed@lambda_down_fix.T <= r_down_G, 
                             lambda_up_fix@self.H_fixed  == -W, self.h_fixed@lambda_up_fix.T <= r_up_G, 
                             lambda_f_up_fix@self.H_fixed  == PTDF@(-node_G@W + node_W), self.h_fixed@lambda_f_up_fix.T <= f_margin_up, 
                             lambda_f_down_fix@self.H_fixed  == -PTDF@(-node_G@W + node_W), self.h_fixed@lambda_f_down_fix.T <= f_margin_down]


            robust_opf_problem = cp.Problem(objective_funct, DA_constraints + Robust_constr + Robust_constr_fix)
             
            self.robust_opf_layer = CvxpyLayer(robust_opf_problem, parameters=[H_param, h_param],
                                               variables = [p_G, r_up_G, r_down_G, W, f_margin_up, f_margin_down, 
                                                            lambda_up, lambda_down, lambda_f_up, lambda_f_down] + fix_rob_var )
                 
        else:
                                
            robust_opf_problem = cp.Problem(objective_funct, DA_constraints + Robust_constr)
             
            self.robust_opf_layer = CvxpyLayer(robust_opf_problem, parameters=[H_param, h_param],
                                               variables = [p_G, r_up_G, r_down_G, W, f_margin_up, f_margin_down, 
                                                            lambda_up, lambda_down, lambda_f_up, lambda_f_down])
            
    def forward(self, H_value, h_value):
        """
        Forward pass: solve robust DCOPF problem, with uncertainty set Xi = {xi | H@xi <= h}

        Args:
            H_value, h_value: current parameterization of inequalities of uncertainty set Xi

        Returns:
            torch.Tensor: outputs of CVXPY layer, see declaration above
        """

        try:
            cvxpy_output = self.robust_opf_layer(H_value, h_value, solver_args={'max_iters':20_000, "solve_method": "ECOS"})
        except:            
            cvxpy_output = self.robust_opf_layer(H_value, h_value, solver_args={'max_iters':20_000, "solve_method": "SCS"})
        return cvxpy_output
    
    def epoch_train(self, loader, opt=None):
        """
        Training: learning parameters w_scen for an uncertainty set Xi = { xi_i}, for i = 1,..., n

        Args:
            train_loader, val_loader: data generators
            optimizer: gradient-descent based algo parameters
            epochs: number of iterations over the whole data set
            patience: for early stoping
            validation: use validation data set to assess performance (not used here)
            relative_tolerance: threshold of percentage loss reduction
            
        Returns:
            torch.Tensor: outputs of CVXPY layer, see declaration above
        """                

        total_loss = 0.
        
        for i, batch_data in enumerate(loader):
            # sample batch data: wind error realizations
            y_batch = batch_data[0]

            # Forward pass: Robust OPF
            decisions_hat = self.forward(self.H, self.h)
            
            p_hat = decisions_hat[0]
            r_up_hat = decisions_hat[1]
            r_down_hat = decisions_hat[2]
            W_hat = decisions_hat[3] # Linear decision rules --- recourse policy
            f_margin_up_hat = decisions_hat[4]
            f_margin_down_hat = decisions_hat[5]
            
            # Evaluate cost-driven error
            # DA cost                
            cost_DA_hat = self.Cost@p_hat + self.C_r_up@r_up_hat + self.C_r_down@r_down_hat

            # RT dispatch cost (penalize violations)
            recourse_actions = -(W_hat@y_batch.T).T
            
            ### Estimate RT redispatch cost: penalize infeasibilities (see Mieth, Poor 2023 )
            # Projection step to avoid infeasibilities due to numerical issues
            aggr_rup_violations = torch.maximum(recourse_actions - r_up_hat, torch.zeros(self.grid['n_unit'])).sum()
            aggr_rdown_violations = torch.maximum(-recourse_actions - r_down_hat, torch.zeros(self.grid['n_unit'])).sum()
                                                                                                                                  
            # exceeding line rating
            rt_injections = (self.PTDF@(self.node_G@recourse_actions.T + self.node_W@y_batch.T)).T
            
            aggr_f_margin_up_violations = torch.maximum( rt_injections - f_margin_up_hat, torch.zeros(self.grid['n_lines']) ).sum()
            aggr_f_margin_down_violations = torch.maximum( -rt_injections - f_margin_down_hat, torch.zeros(self.grid['n_lines'])).sum()

            rt_cost = self.c_viol*(aggr_rup_violations + aggr_rdown_violations + aggr_f_margin_up_violations + aggr_f_margin_down_violations)                
            
            # loss: aggregate DA and RT cost
            loss = cost_DA_hat.mean() + rt_cost                
                
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
            
            total_loss += loss.item() * len(y_batch)
            
        return total_loss / len(loader.dataset)

    
    def train_model(self, train_loader, val_loader, optimizer, epochs = 20, patience=5, verbose = 0, plot = True, validation = False):
        
        if val_loader == None:
            validation = False
            
        best_train_loss = float('inf')
        best_val_loss = float('inf')
        early_stopping_counter = 0
        best_weights = copy.deepcopy(self.state_dict())

        Pmax_tensor = torch.FloatTensor(self.grid['Pmax'].reshape(-1))
        Pmin_tensor = torch.FloatTensor(self.grid['Pmin'].reshape(-1))
        
        for epoch in range(epochs):
            # activate train functionality
            self.train()

            average_train_loss = self.epoch_train(train_loader, optimizer)
            if validation:
                val_loss = self.epoch_train(val_loader)
            else:
                val_loss = average_train_loss

            # visualize current scenarios and the induced convex hull
            if (epoch%10 == 0) and (plot == True): 
                for batch in train_loader:
                    y_batch = batch[0]                    

                    # visualizations for sanity chekc
                    fig, ax = plt.subplots(figsize = (6,4))
                    x = np.linspace(-2, 2, 1000)
                    
                    H_np = to_np(self.H)
                    h_np = to_np(self.h)
                    
                    y = [(h_np[i] - H_np[i,0]*x)/H_np[i,1] for i in range(len(H_np))]
                    
    
                    plt.scatter(y_batch[:,0], y_batch[:,1], label = '_nolegend_')
                    for i in range(len(H_np)):
                        plt.plot(x, y[i], color = 'tab:green')
                    plt.ylim([-2, 2])
                    plt.xlim([-2, 2])
                    plt.title(f'C_viol = {self.c_viol} \$/MWh, Iteration: {epoch}', fontsize = 12)
                    plt.xlabel('Error 1')
                    plt.xlabel('Error 2')
                    plt.legend([f'Ineq. {i}' for i in range(self.num_constr)], fontsize = 12, ncol = 2)
                    plt.show()
                    break
            
            if verbose != -1:
                print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} - Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(self.state_dict())
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print("Early stopping triggered.")
                    # recover best weights
                    self.load_state_dict(best_weights)
                    return
        print('Reached epoch limit.')
        self.load_state_dict(best_weights)
        return
                
class Scenario_Robust_OPF(nn.Module):        
    ''' Deterministic market clearing and feasibility under scenarios, modeling redispatch as recourse. 
        This is equivalent to a robust problem parameterized by a discrete scenario uncertainty set.
        The goal is to learn this parametirization (i.e., the scenarios to select) in a cost-driven, decision-focused way.
        - Args:
            num_uncertainties: sources of uncertainty (could be tuned with budget constraint)
            num_scen: number of discrete scenarios to consider (design parameter)
            support: support of uncertainty, used in projecetion step, size:(2*number of uncertainties), first row is always Upper Bound, second row is Lower Bound
            
            grid: dictionary with grid information, see separate functions to read matpower cases
            c_viol: constraint violation penalties for real-time redispatch 
            '''                
    def __init__(self, num_uncertainties, num_scen, initial_scenarios, support, grid, c_viol = 2*1e4, regularization = 0, include_network = True):
        super(Scenario_Robust_OPF, self).__init__()
        
        Pmax = grid['Pmax']
        C_r_up = grid['C_r_up']
        C_r_down = grid['C_r_down']
        Cost = grid['Cost']    
        
        node_G = grid['node_G']
        node_L = grid['node_L']
        node_W = grid['node_W']
            
        PTDF = grid['PTDF']
        
        self.support_UB = torch.FloatTensor(support[0]) # num_uncertainties * 2
        self.support_LB = torch.FloatTensor(support[1]) # num_uncertainties * 2
        self.Pmax = torch.FloatTensor(Pmax)
        self.Pmin = torch.FloatTensor(np.zeros(grid['n_unit']))
        self.Cost = torch.FloatTensor(Cost)
        self.C_r_up = torch.FloatTensor(C_r_up)
        self.C_r_down = torch.FloatTensor(C_r_down)
        self.node_G = torch.FloatTensor(node_G)
        self.node_L = torch.FloatTensor(node_L)
        self.node_W = torch.FloatTensor(node_W)
        self.PTDF = torch.FloatTensor(PTDF)
        self.w_exp = torch.FloatTensor(grid['w_exp'])
        self.w_cap = torch.FloatTensor(grid['w_cap'])
        
        # Initialize polytope
        
        # number of unknown net demands (here only considering wind)
        self.num_uncertainties = num_uncertainties
        self.c_viol = c_viol
        # number robust constraints to learn
        self.num_constr = num_scen
        
        # Parameters to be estimated (initialize at the extremes of the box)
        self.w_scenarios_param = nn.Parameter(torch.FloatTensor(initial_scenarios).requires_grad_())
        
        # projection step: ensure that scenarios fall within support  
        w_proj = torch.maximum(torch.minimum( self.w_scenarios_param, self.support_UB), self.support_LB)
        # update parameter values
        with torch.no_grad():
            self.w_scenarios_param.copy_(w_proj)
                    
        self.grid = grid
        self.regularization = regularization
        self.include_network = include_network
                
        #### Deterministic Clearing with Robust Scenario constraints (feasibility under scenarios, redispatch as recourse)
        w_scen_param = cp.Parameter((num_scen, num_uncertainties))
                
        ##### DA Variables
                        
        p_G = cp.Variable((grid['n_unit']), nonneg = True)
        r_up_G = cp.Variable((grid['n_unit']), nonneg = True)
        r_down_G = cp.Variable((grid['n_unit']), nonneg = True)

        f_margin_up = cp.Variable((grid['n_lines']), nonneg = True)
        f_margin_down = cp.Variable((grid['n_lines']), nonneg = True)
        
        DA_cost = cp.Variable((1), nonneg = True)
        
        #### recourse decisions
        p_rt_up = cp.Variable((grid['n_unit'], num_scen), nonneg = True)
        p_rt_down = cp.Variable((grid['n_unit'], num_scen), nonneg = True)

        slack_up = cp.Variable(grid['n_nodes'], nonneg = True)
        slack_down = cp.Variable(grid['n_nodes'], nonneg = True)

        #### DA constraints
        DA_constraints = [p_G + r_up_G<= grid['Pmax'].reshape(-1), p_G - r_down_G >= 0, 
                          p_G.sum() + grid['w_exp'].sum() == grid['Pd'].sum(), 
                          grid['Line_Capacity'].reshape(-1) - f_margin_up == PTDF@(node_G@p_G + node_W@grid['w_exp'] - node_L@grid['Pd']),
                          grid['Line_Capacity'].reshape(-1) - f_margin_down == -PTDF@(node_G@p_G + node_W@grid['w_exp'] - node_L@grid['Pd'] ),] 
        

        ##### Feasibility under all scenarios, robust constraints
        Robust_constr = [ p_rt_up[:,s] <= r_up_G for s in range(num_scen)] \
                        + [p_rt_down[:,s] <= r_down_G for s in range(num_scen)]\
                        + [p_rt_up[:,s].sum() - p_rt_down[:,s].sum() + slack_up.sum() - slack_down.sum()+ w_scen_param[s].sum() == 0 for s in range(num_scen)]\
                        + [PTDF@(node_G@(p_rt_up[:,s] - p_rt_down[:,s]) + (slack_up - slack_down) + node_W@w_scen_param[s]) <= f_margin_up for s in range(num_scen)]\
                        + [-PTDF@(node_G@(p_rt_up[:,s] - p_rt_down[:,s]) + (slack_up - slack_down) + node_W@w_scen_param[s]) <= f_margin_down for s in range(num_scen)]\
        
        DA_constraints += [DA_cost == grid['Cost']@p_G + grid['C_r_up']@r_up_G 
                           + grid['C_r_down']@r_down_G + c_viol*(slack_up.sum() + slack_down.sum())]
        
        objective_funct = cp.Minimize( DA_cost )                 
        robust_opf_problem = cp.Problem(objective_funct, DA_constraints + Robust_constr)
         
        self.robust_opf_layer = CvxpyLayer(robust_opf_problem, parameters=[w_scen_param],
                                           variables = [p_G, r_up_G, r_down_G, p_rt_up, p_rt_down, f_margin_up, f_margin_down, DA_cost])
                                        
        
        ###### RT market layer/ full re-dispatch

        r_up_param = cp.Parameter(grid['n_unit'], nonneg = True)
        r_down_param = cp.Parameter(grid['n_unit'], nonneg = True)
        f_margin_up_param = cp.Parameter(grid['n_lines'], nonneg = True)
        f_margin_down_param = cp.Parameter(grid['n_lines'], nonneg = True)
        w_realized_error = cp.Parameter(num_uncertainties)

        recourse_up = cp.Variable((grid['n_unit']), nonneg = True)
        recourse_down = cp.Variable((grid['n_unit']), nonneg = True)
        
        slack_up = cp.Variable(grid['n_nodes'], nonneg = True)
        slack_down = cp.Variable(grid['n_nodes'], nonneg = True)
        
        #l_shed = cp.Variable(grid['n_loads'], nonneg = True)
        
        cost_RT = cp.Variable(1, nonneg = True)
        
        RT_sched_constraints = []

        RT_sched_constraints += [recourse_up <= r_up_param]            
        RT_sched_constraints += [recourse_down <= r_down_param]            
        RT_sched_constraints += [PTDF@(node_G@(recourse_up - recourse_down) + (slack_up -  slack_down) + node_W@(w_realized_error)) <= f_margin_up_param]            
        RT_sched_constraints += [-PTDF@(node_G@(recourse_up - recourse_down) + (slack_up -  slack_down) + node_W@(w_realized_error)) <= f_margin_down_param]            
        #RT_sched_constraints += [-PTDF@(node_G@(p_g_param + recourse_up - recourse_down - g_shed) + node_L@(l_shed-grid['Pd']) + node_W@(grid['w_exp'] + w_realized_error)) <= grid['Line_Capacity'].reshape(-1)]            
                
        # balancing
        RT_sched_constraints += [ recourse_up.sum() - recourse_down.sum() + w_realized_error.sum() + (slack_up -  slack_down).sum() == 0]
        RT_sched_constraints += [cost_RT == c_viol*(slack_up.sum() + slack_down.sum()) ]
            
        objective_funct = cp.Minimize( cost_RT ) 
        rt_problem = cp.Problem(objective_funct, RT_sched_constraints)
         
        self.rt_layer = CvxpyLayer(rt_problem, parameters=[r_up_param, r_down_param, f_margin_up_param, f_margin_down_param, w_realized_error],
                                           variables = [recourse_up, recourse_down, slack_up, slack_down, cost_RT] )


    def forward(self, w_scen):
        """
        Clear DA market with feasibility under scenarios w_scen.

        Args:
            list_inputs: A list of of input tensors/ PDFs.

        Returns:
            torch.Tensor: CVXPY output, see above
        """
        # Pass the combined output to the CVXPY layer
        try:
            cvxpy_output = self.robust_opf_layer(w_scen, solver_args={'max_iters':20_000, "solve_method": "ECOS"})
        except:            
            cvxpy_output = self.robust_opf_layer(w_scen, solver_args={'max_iters':20_000})
        return cvxpy_output
    
    def epoch_train(self, loader, opt=None):
        """
        Training: learning parameters w_scen for an uncertainty set Xi = { xi_i}, for i = 1,..., n

        Args:
            train_loader, val_loader: data generators
            optimizer: gradient-descent based algo parameters
            epochs: number of iterations over the whole data set
            patience: for early stoping
            validation: use validation data set to assess performance (not used here)
            relative_tolerance: threshold of percentage loss reduction
            
        Returns:
            torch.Tensor: outputs of CVXPY layer, see declaration above
        """                

        total_loss = 0.
        
        for i, batch_data in enumerate(loader):
        
            y_batch = batch_data[0]

            # Forward pass: solve robust OPF
            decisions_hat = self.forward(self.w_scenarios_param)
            
            p_hat = decisions_hat[0]
            r_up_hat = decisions_hat[1]
            r_down_hat = decisions_hat[2]
            f_margin_up_hat = decisions_hat[5]
            f_margin_down_hat = decisions_hat[6]
            cost_DA_hat = decisions_hat[-1]
            
            # Project to feasible set (might incur numerical errors)
            p_hat_proj = torch.maximum(torch.minimum(p_hat, self.Pmax), self.Pmin)
            r_up_hat_proj = torch.maximum(torch.minimum(r_up_hat, self.Pmax - p_hat_proj), torch.zeros(self.grid['n_unit']))
            r_down_hat_proj = torch.maximum(torch.minimum(r_down_hat, p_hat_proj), torch.zeros(self.grid['n_unit']))
            f_margin_up_hat_proj = torch.maximum(f_margin_up_hat, torch.zeros(self.grid['n_lines']) ) 
            f_margin_down_hat_proj = torch.maximum(f_margin_down_hat, torch.zeros(self.grid['n_lines']) ) 
            

            # RT dispatch cost (penalize violations)
            try:
                rt_output = self.rt_layer(r_up_hat_proj, r_down_hat_proj, f_margin_up_hat_proj, f_margin_down_hat_proj, 
                                          y_batch, solver_args={'max_iters':50_000, "solve_method": "ECOS"}) 
            except:
                rt_output = self.rt_layer(r_up_hat_proj, r_down_hat_proj, f_margin_up_hat_proj, f_margin_down_hat_proj, 
                                          y_batch, solver_args={'max_iters':50_000, "solve_method": "SCS"}) 
                
            cost_RT_hat = rt_output[-1]
            
            loss = cost_DA_hat.mean() + cost_RT_hat.mean()
            
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                ### Projection step// box support
                w_proj = torch.maximum(torch.minimum( self.w_scenarios_param, self.support_UB), self.support_LB)
                with torch.no_grad():
                    self.w_scenarios_param.copy_(w_proj)

            total_loss += loss.item() * len(y_batch)
            
        return total_loss / len(loader.dataset)
        
    def train_model(self, train_loader, val_loader, optimizer, epochs = 20, patience=5, verbose = 0, validation = True):
        
        if val_loader == None: validation = False
        # best_train_loss = float('inf')
        best_val_loss = float('inf')
        early_stopping_counter = 0
        best_weights = copy.deepcopy(self.state_dict())

        for epoch in range(epochs):
            # activate train functionality
            self.train()

            # visualize current scenarios and the induced convex hull
            if ((epoch)%5 == 0): 

                for batch in train_loader:
                    y_batch = batch[0]
                    
                    fig, ax = plt.subplots(figsize = (6,4))                
                    w_scen = to_np(self.w_scenarios_param)
                    plt.scatter(y_batch[:,0], y_batch[:,1], label = 'Data')                    
                    # Cost-driven convex hull
                    hull = ConvexHull(w_scen)
                    plt.plot(w_scen[:,0], w_scen[:,1], 's', color = 'tab:red', label = 'Scenarios')
                    for j, simplex in enumerate(hull.simplices):
                        if j == 0:
                            plt.plot(w_scen[simplex, 0], w_scen[simplex, 1], color = 'tab:red', linestyle = '--', lw = 2, label = 'Convex Hull')    
                        else:
                            plt.plot(w_scen[simplex, 0], w_scen[simplex, 1], color = 'tab:red', linestyle = '--', lw = 2)    
                    plt.ylim([-2, 2])
                    plt.xlim([-2, 2])
                    plt.title(f'C_viol = {self.c_viol} \$/MWh', fontsize = 12)
                    plt.xlabel('Error 1')
                    plt.xlabel('Error 2')
                    plt.legend(fontsize = 12, loc = 'upper right')
                    plt.show()
                    break

            average_train_loss = self.epoch_train(train_loader, optimizer)

            if validation:
                val_loss = self.epoch_train(val_loader)
            else:
                val_loss = average_train_loss
            
            if verbose != -1:
                print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} - Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(self.state_dict())
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print("Early stopping triggered.")
                    # recover best weights
                    self.load_state_dict(best_weights)
                    return
                
class Contextual_Scenario_Robust_OPF(nn.Module):        
    
    ''' Deterministic market clearing and feasibility under scenarios, modeling redispatch as recourse. 
        This is equivalent to a robust problem parameterized by a discrete scenario uncertainty set.
        The goal is to learn this parametirization (i.e., the scenarios to select) in a cost-driven, decision-focused way.
        Conditionally on the realization of some contextual information, such as nominal point forecasts, weather conditions, etc.
        - Args:
            mlp_param_dict: dictionary with the MLP parameters, see standard MLP class for details
            num_uncertainties: sources of uncertainty (could be tuned with budget constraint)
            num_scen: number of discrete scenarios to consider (design parameter)
            support: support of uncertainty, used in projecetion step, size:(2*number of uncertainties), first row is always Upper Bound, second row is Lower Bound
            
            grid: dictionary with grid information, see separate functions to read matpower cases
            c_viol: constraint violation penalties for real-time redispatch 
        
        *** Note, the support for forecast errors changes as a function of nominal point forecasts: ***
                Upper bound: Wind Capacity - Wind Expected Value
                Lower bound: - Wind Expected Value
            '''                
            
    def __init__(self, mlp_param_dict, num_uncertainties, num_scen, grid, initial_scenarios, c_viol = 2*1e4, 
                 activation = nn.ReLU(), include_network = True, add_fixed_box = True):
        super(Contextual_Scenario_Robust_OPF, self).__init__()
                
        #self.support_UB = torch.FloatTensor(support[0]) # num_uncertainties * 2
        #self.support_LB = torch.FloatTensor(support[1]) # num_uncertainties * 2
        self.Pmax = torch.FloatTensor(grid['Pmax'])
        self.Pmin = torch.FloatTensor(np.zeros(grid['n_unit']))
        self.Cost = torch.FloatTensor(grid['Cost'])
        self.C_r_up = torch.FloatTensor(grid['C_r_up'])
        self.C_r_down = torch.FloatTensor(grid['C_r_down'])
        self.node_G = torch.FloatTensor(grid['node_G'])
        self.node_L = torch.FloatTensor(grid['node_L'])
        self.node_W = torch.FloatTensor(grid['node_W'])
        self.PTDF = torch.FloatTensor(grid['PTDF'])
        self.w_cap = torch.FloatTensor(grid['w_cap'])
        self.c_viol = c_viol        
        self.grid = grid
        self.include_network = include_network
        
        ### Intialize polytope defined by scenarios
        self.num_uncertainties = num_uncertainties # number of uncertain parameters (only wind for this example)
        self.num_constr = num_scen # number of scenarios to learn
        
        # Parameters to be estimated (initialize at the extremes of the box)
        self.w_scenarios_param = nn.Parameter(torch.FloatTensor(initial_scenarios).requires_grad_())
        
        
        # projection step (set as separate function)        
        #w_proj = torch.maximum(torch.minimum( self.w_scenarios_param, self.support_UB), self.support_LB)
        # update parameter values
        #with torch.no_grad():
        #    self.w_scenarios_param.copy_(w_proj)
        
        ### Initialize sequential MLP model to predict feasibility scenarios/ robust vertices
        input_size = mlp_param_dict['input_size']
        hidden_sizes = mlp_param_dict['hidden_sizes']
        output_size = num_uncertainties*num_scen
        
        #!!! *** Some trickery is required to reshape the output layer in a matrix, check to ensure is done correctly ***        
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation)

        self.model = nn.Sequential(*layers)
        
        #### Deterministic DA Market Clearing with Feasibility Scenarios (redispatch modeled as recourse actions)
        ## Declare problem parameters
        
        # Parameters that we **do not** learn: nominal wind production, nominal demand (expected values)
        w_nominal = cp.Parameter((grid['n_wind']), nonneg = True)
        d_nominal = cp.Parameter((grid['n_loads']), nonneg = True)
        
        # Parameters that we want to learn: wind error scenarios
        w_error_scen = cp.Parameter((num_scen, num_uncertainties))
        
        ###### Problem variables
        
        # DA variables
        p_G = cp.Variable((grid['n_unit']), nonneg = True)
        r_up_G = cp.Variable((grid['n_unit']), nonneg = True)
        r_down_G = cp.Variable((grid['n_unit']), nonneg = True)
        DA_cost = cp.Variable((1), nonneg = True)
        
        f_margin_up = cp.Variable((grid['n_lines']), nonneg = True)
        f_margin_down = cp.Variable((grid['n_lines']), nonneg = True)
        
        # recourse actions
        p_rt_up = cp.Variable((grid['n_unit'], num_scen), nonneg = True)
        p_rt_down = cp.Variable((grid['n_unit'], num_scen), nonneg = True)

        slack_up = cp.Variable((grid['n_nodes'], num_scen), nonneg = True)
        slack_down = cp.Variable((grid['n_nodes'], num_scen), nonneg = True)

        #### DA constraints
        DA_constraints = [p_G + r_up_G<= self.Pmax.reshape(-1), p_G - r_down_G >= 0, 
                          p_G.sum() + w_nominal.sum() == d_nominal.sum(), 
                          grid['Line_Capacity'].reshape(-1) - f_margin_up == self.PTDF@(self.node_G@p_G + self.node_W@w_nominal - self.node_L@d_nominal),
                          grid['Line_Capacity'].reshape(-1) - f_margin_down == -self.PTDF@(self.node_G@p_G + self.node_W@w_nominal - self.node_L@d_nominal ),]         

        ##### Feasibility scenarios/ robust constraints per scenario
        Robust_constr = [ p_rt_up[:,s] <= r_up_G for s in range(num_scen)] \
                        + [p_rt_down[:,s] <= r_down_G for s in range(num_scen)]\
                        + [p_rt_up[:,s].sum() - p_rt_down[:,s].sum() + slack_up[:,s].sum() - slack_down[:,s].sum() + w_error_scen[s].sum() == 0 for s in range(num_scen)]\
                        + [self.PTDF@(self.node_G@(p_rt_up[:,s] - p_rt_down[:,s]) + slack_up[:,s] - slack_down[:,s] + self.node_W@w_error_scen[s]) <= f_margin_up for s in range(num_scen)]\
                        + [-self.PTDF@(self.node_G@(p_rt_up[:,s] - p_rt_down[:,s]) + slack_up[:,s] - slack_down[:,s] + self.node_W@w_error_scen[s]) <= f_margin_down for s in range(num_scen)]\
        
        DA_constraints += [DA_cost == self.Cost@p_G + self.C_r_up@r_up_G + self.C_r_down@r_down_G + self.c_viol*(slack_up.sum() + slack_down.sum())]
        
        objective_funct = cp.Minimize( DA_cost ) 
        robust_opf_problem = cp.Problem(objective_funct, DA_constraints + Robust_constr)         
        self.robust_opf_layer = CvxpyLayer(robust_opf_problem, parameters=[d_nominal, w_nominal, w_error_scen],
                                           variables = [p_G, r_up_G, r_down_G, p_rt_up, p_rt_down, f_margin_up, f_margin_down, slack_up, slack_down, DA_cost])
                                        
        
        ###### RT market layer/ full redispatch
        #p_g_param = cp.Parameter(grid['n_unit'], nonneg = True)
        r_up_param = cp.Parameter(grid['n_unit'], nonneg = True)
        r_down_param = cp.Parameter(grid['n_unit'], nonneg = True)
        f_margin_up_param = cp.Parameter(grid['n_lines'], nonneg = True)
        f_margin_down_param = cp.Parameter(grid['n_lines'], nonneg = True)
        w_realized_error = cp.Parameter(num_uncertainties)

        recourse_up = cp.Variable((grid['n_unit']), nonneg = True)
        recourse_down = cp.Variable((grid['n_unit']), nonneg = True)
        
        rt_slack_up = cp.Variable(grid['n_nodes'], nonneg = True)
        rt_slack_down = cp.Variable(grid['n_nodes'], nonneg = True)
                
        cost_RT = cp.Variable(1, nonneg = True)
        
        RT_sched_constraints = []

        RT_sched_constraints += [recourse_up <= r_up_param]            
        RT_sched_constraints += [recourse_down <= r_down_param]            
        RT_sched_constraints += [self.PTDF@(self.node_G@(recourse_up - recourse_down) + (rt_slack_up -  rt_slack_down) + self.node_W@(w_realized_error)) <= f_margin_up_param]            
        RT_sched_constraints += [-self.PTDF@(self.node_G@(recourse_up - recourse_down) + (rt_slack_up -  rt_slack_down) + self.node_W@(w_realized_error)) <= f_margin_down_param]            
                
        # balancing
        RT_sched_constraints += [ recourse_up.sum() - recourse_down.sum() + w_realized_error.sum() + (rt_slack_up -  rt_slack_down).sum() == 0]
        RT_sched_constraints += [cost_RT == self.c_viol*(rt_slack_up.sum() + rt_slack_down.sum()) ]
            
        objective_funct = cp.Minimize( cost_RT ) 
        rt_problem = cp.Problem(objective_funct, RT_sched_constraints)
         
        self.rt_layer = CvxpyLayer(rt_problem, parameters=[r_up_param, r_down_param, f_margin_up_param, f_margin_down_param, w_realized_error],
                                           variables = [recourse_up, recourse_down, rt_slack_up, rt_slack_down, cost_RT] )


    def scenario_projection(self, w_nominal, w_error_scen):
        """
        Projects scenarios to box constraints defined by support (support varies with expected production value)
            Support depends on problem parameterization (e.g., expected wind production)
        """            
        temp_UB = self.w_cap - w_nominal
        temp_LB = - w_nominal
        
        w_proj = w_error_scen
        
        #print(torch.maximum(torch.minimum( w_error_scen[:,0,:], temp_UB), temp_LB).shape)
        with torch.no_grad():
            for j in range(self.num_constr):
                    w_proj[:,j,:] = torch.maximum(torch.minimum( w_error_scen[:,j,:], temp_UB), temp_LB)
        
        
        # w_proj = self.simplex_projection(self.weights.clone())
        # with torch.no_grad():
        #     self.weights.copy_(torch.FloatTensor(w_proj))

        # update parameter values
        #with torch.no_grad():
        #    self.w_scenarios_param.copy_(w_proj)
        return  w_proj  
    
    def forward(self, demand_batch, wind_batch, projection = True):
        """
        For a realization of contextual information:
            i) Predict the robust feasibility scenarios
            ii) Solve deterministic DA market for a sample of nominal demands and wind production

        Returns:
            torch.Tensor: CVXPY output, see above.
        """
        
        # predict the scenarios based on realization of features
        batch_size = len(demand_batch)
        feat_hat = torch.cat([demand_batch, wind_batch], axis = 1)
        
        #w_scen_hat = torch.FloatTensor(np.array([self.model_scen[i](feat_hat) for i in range(self.num_constr)]))
        
        # !!!! Check that it transposes everything correct
        w_scen_hat = self.model(feat_hat).reshape(batch_size, self.num_constr, self.num_uncertainties)
        #standard_output = self.model(feat_hat)
        if projection:        
            # project back to feasible set
            w_scen_hat_proj = self.scenario_projection(wind_batch, w_scen_hat)
        else:
            # project back to feasible set
            w_scen_hat_proj = w_scen_hat            
            
        # Pass the predicted scenarios to the solver
        try:
            cvxpy_output = self.robust_opf_layer(demand_batch, wind_batch, w_scen_hat_proj, solver_args={'max_iters':50_000, "solve_method": "ECOS"})
        except:
            cvxpy_output = self.robust_opf_layer(demand_batch, wind_batch, w_scen_hat_proj, solver_args={'max_iters':50_000, "solve_method": "SCS"})


        return cvxpy_output
    
    def predict(self, demand_batch, wind_batch, projection = True):
        """
        Additional functionality to output the predicted scenarios
        """
        
        # predict the scenarios based on realization of features
        batch_size = len(demand_batch)
        feat_hat = torch.cat([demand_batch, wind_batch], axis = 1)
                
        # !!!! Check that it transpoes everything correct
        with torch.no_grad():
            w_scen_hat = self.model(feat_hat).reshape(batch_size, self.num_constr, self.num_uncertainties)
            #standard_output = self.model(feat_hat)
            
            if projection:
                # project back to feasible set
                w_scen_hat_proj = self.scenario_projection(wind_batch, w_scen_hat)
                return w_scen_hat_proj
            else:
                return w_scen_hat
        
    def epoch_train(self, loader, opt=None):
        """
        Training: learning parameters w_scen for an uncertainty set Xi = { xi_i}, for i = 1,..., n

        Args:
            train_loader, val_loader: data generators
            optimizer: gradient-descent based algo parameters
            epochs: number of iterations over the whole data set
            patience: for early stoping
            validation: use validation data set to assess performance (not used here)
            relative_tolerance: threshold of percentage loss reduction
            
        Returns:
            torch.Tensor: outputs of CVXPY layer, see declaration above
        """                

        total_loss = 0.
        
        for i, batch_data in enumerate(loader):
        
            d_hat = batch_data[0]
            w_hat = batch_data[1]
            w_error_hat = batch_data[2]
            
            # Forward pass: solve robust OPF
            decisions_hat = self.forward(d_hat, w_hat)
            
            p_hat = decisions_hat[0]
            r_up_hat = decisions_hat[1]
            r_down_hat = decisions_hat[2]
            f_margin_up_hat = decisions_hat[5]
            f_margin_down_hat = decisions_hat[6]
            cost_DA_hat = decisions_hat[-1]
            
            # Project to feasible set (might incur numerical errors)
            p_hat_proj = torch.maximum(torch.minimum(p_hat, self.Pmax), self.Pmin)
            r_up_hat_proj = torch.maximum(torch.minimum(r_up_hat, self.Pmax - p_hat_proj), torch.zeros(self.grid['n_unit']))
            r_down_hat_proj = torch.maximum(torch.minimum(r_down_hat, p_hat_proj), torch.zeros(self.grid['n_unit']))
            f_margin_up_hat_proj = torch.maximum(f_margin_up_hat, torch.zeros(self.grid['n_lines']) ) 
            f_margin_down_hat_proj = torch.maximum(f_margin_down_hat, torch.zeros(self.grid['n_lines']) ) 
            

            # RT dispatch cost (penalize violations)
            try:
                rt_output = self.rt_layer(r_up_hat_proj, r_down_hat_proj, f_margin_up_hat_proj, f_margin_down_hat_proj, 
                                          w_error_hat, solver_args={'max_iters':50_000, "solve_method": "ECOS"}) 
            except:
                rt_output = self.rt_layer(r_up_hat_proj, r_down_hat_proj, f_margin_up_hat_proj, f_margin_down_hat_proj, 
                                          w_error_hat, solver_args={'max_iters':50_000, "solve_method": "SCS"}) 
                
            cost_RT_hat = rt_output[-1]
            
            loss = cost_DA_hat.mean() + cost_RT_hat.mean()
            
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
                # ### Projection step// box support
                # w_proj = self.scenario_projection(w_hat, w_error_hat)

            total_loss += loss.item() * len(w_error_hat)
            
        return total_loss / len(loader.dataset)

    def train_model(self, train_loader, val_loader, optimizer, epochs = 20, patience=5, verbose = 0):
        
        # best_train_loss = float('inf')
        best_val_loss = float('inf')
        early_stopping_counter = 0
        best_weights = copy.deepcopy(self.state_dict())

        for epoch in range(epochs):
            # activate train functionality
            self.train()

            average_train_loss = self.epoch_train(train_loader, optimizer)

            if val_loader == []:
                val_loss = average_train_loss
            else:
                val_loss = self.epoch_train(val_loader)
            
            if verbose != -1:
                print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} - Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(self.state_dict())
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print("Early stopping triggered.")
                    # recover best weights
                    self.load_state_dict(best_weights)
                    return

class DA_RobustScen_Clearing(nn.Module):        
    'Solves DA market with robust constraints on a set of scenarios'
    def __init__(self, grid, num_uncertainties, num_scen, c_viol = 2*1e4):
        super(DA_RobustScen_Clearing, self).__init__()
                
        node_G = grid['node_G']
        node_L = grid['node_L']
        node_W = grid['node_W']            
        PTDF = grid['PTDF']

        self.Pmax = torch.FloatTensor(grid['Pmax'])
        self.Pmin = torch.FloatTensor(np.zeros(grid['n_unit']))
        self.Cost = torch.FloatTensor(grid['Cost'])
        self.C_r_up = torch.FloatTensor(grid['C_r_up'])
        self.C_r_down = torch.FloatTensor(grid['C_r_down'])
        
        self.c_viol = c_viol
                
        self.grid = grid
        self.num_uncertainties = num_uncertainties                                                
        
        #### Robust DCOPF Layer
        # Parameters: nominal wind production, nominal demand (expected values), scenarios for errors
        w_nominal = cp.Parameter((grid['n_wind']), nonneg = True)
        d_nominal = cp.Parameter((grid['n_loads']), nonneg = True)
        
        w_error_scen = cp.Parameter((num_scen, num_uncertainties))
        
        ###### DA variables and linear decision rules
        
        ### variables    
        # DA Variables                        
        p_G = cp.Variable((grid['n_unit']), nonneg = True)
        r_up_G = cp.Variable((grid['n_unit']), nonneg = True)
        r_down_G = cp.Variable((grid['n_unit']), nonneg = True)

        f_margin_up = cp.Variable((grid['n_lines']), nonneg = True)
        f_margin_down = cp.Variable((grid['n_lines']), nonneg = True)

        # Recourse variables for scenarios        
        p_rt_up = cp.Variable((grid['n_unit'], num_scen), nonneg = True)
        p_rt_down = cp.Variable((grid['n_unit'], num_scen), nonneg = True)

        #### DA constraints
        
        ### DA constraints 
        DA_constraints = [p_G + r_up_G<= grid['Pmax'].reshape(-1), p_G - r_down_G >= 0, 
                          p_G.sum() + w_nominal.sum() == d_nominal.sum(), 
                          grid['Line_Capacity'].reshape(-1) - f_margin_up == PTDF@(node_G@p_G + node_W@w_nominal - node_L@d_nominal),
                          grid['Line_Capacity'].reshape(-1) - f_margin_down == -PTDF@(node_G@p_G + node_W@w_nominal - node_L@d_nominal ),] 
        

        ##### Robust constraints per scenario
        
        # Reformulation of robust constraints
        # downward reserve bound/ each row are the duals to reformulate each constraints
        
        Robust_constr = [ p_rt_up[:,s] <= r_up_G for s in range(num_scen)] \
                        + [p_rt_down[:,s] <= r_down_G for s in range(num_scen)]\
                        + [p_rt_up[:,s].sum() - p_rt_down[:,s].sum() + w_error_scen[s].sum() == 0 for s in range(num_scen)]\
                        + [PTDF@(node_G@(p_rt_up[:,s] - p_rt_down[:,s]) + node_W@w_error_scen[s]) <= f_margin_up for s in range(num_scen)]\
                        + [-PTDF@(node_G@(p_rt_up[:,s] - p_rt_down[:,s]) + node_W@w_error_scen[s]) <= f_margin_down for s in range(num_scen)]\
        
        DA_cost_expr = self.Cost@p_G + self.C_r_up@r_up_G + self.C_r_down@r_down_G
        objective_funct = cp.Minimize( DA_cost_expr ) 
                
        robust_opf_problem = cp.Problem(objective_funct, DA_constraints + Robust_constr)
         
        self.robust_opf_layer = CvxpyLayer(robust_opf_problem, parameters=[d_nominal, w_nominal, w_error_scen],
                                           variables = [p_G, r_up_G, r_down_G, p_rt_up, p_rt_down, f_margin_up, f_margin_down])
        
    def forward(self, d_hat, w_hat, w_error_scen_hat):
        
        da_solutions = {}
        
        d_nominal_tensor = torch.FloatTensor(d_hat)
        w_hat_tensor = torch.FloatTensor(w_hat)
        w_error_scen_hat = torch.FloatTensor(w_error_scen_hat)
        
        # solve DA market
        da_output = self.robust_opf_layer(d_nominal_tensor, w_hat_tensor, w_error_scen_hat, solver_args={'max_iters':100_000})                
        
        
        
        da_solutions['p_da'] = to_np(da_output[0])
        da_solutions['r_up'] = to_np(da_output[1])
        da_solutions['r_down'] = to_np(da_output[2])
        #da_solutions['p_recourse'] = to_np(da_output[3])
        da_solutions['f_margin_up'] = to_np(da_output[5])
        da_solutions['f_margin_down'] = to_np(da_output[6])
        
        da_solutions['da_cost'] = self.Cost@da_solutions['p_da'] + self.C_r_up@da_solutions['r_up'] + self.C_r_down@da_solutions['r_down']

        return da_solutions
    
class RT_Clearing(nn.Module):        
    
    def __init__(self, grid, num_uncertainties, c_viol = 2*1e4):
        super(RT_Clearing, self).__init__()
        

        Pmax = grid['Pmax']
        C_r_up = grid['C_r_up']
        C_r_down = grid['C_r_down']
        Cost = grid['Cost']    
        
        node_G = grid['node_G']
        node_L = grid['node_L']
        node_W = grid['node_W']
            
        PTDF = grid['PTDF']
        VOLL = grid['VOLL']
        VOWS = grid['VOWS']

        self.Pmax = torch.FloatTensor(Pmax)
        self.Pmin = torch.FloatTensor(np.zeros(grid['n_unit']))
        
        self.node_G = node_G
        self.node_L = node_L
        self.node_W = node_W

        self.Cost = torch.FloatTensor(Cost)
        self.C_r_up = torch.FloatTensor(C_r_up)
        self.C_r_down = torch.FloatTensor(C_r_down)
        
        self.c_viol = c_viol
                
        self.grid = grid
        self.num_uncertainties = num_uncertainties                                                
        
        ###### RT market layer
        r_up_param = cp.Parameter(grid['n_unit'], nonneg = True)
        r_down_param = cp.Parameter(grid['n_unit'], nonneg = True)
        f_margin_up_param = cp.Parameter(grid['n_lines'], nonneg = True)
        f_margin_down_param = cp.Parameter(grid['n_lines'], nonneg = True)
        w_realized_error = cp.Parameter(num_uncertainties)

        recourse_up = cp.Variable((grid['n_unit']), nonneg = True)
        recourse_down = cp.Variable((grid['n_unit']), nonneg = True)
        g_shed = cp.Variable(grid['n_unit'], nonneg = True)
        l_shed = cp.Variable(grid['n_loads'], nonneg = True)
        cost_RT = cp.Variable(1, nonneg = True)

        
        RT_sched_constraints = []

        RT_sched_constraints += [recourse_up <= r_up_param]            
        RT_sched_constraints += [recourse_down <= r_down_param]            
        RT_sched_constraints += [PTDF@(node_G@(recourse_up - recourse_down - g_shed) + node_L@(l_shed) + node_W@(w_realized_error)) <= f_margin_up_param]            
        RT_sched_constraints += [-PTDF@(node_G@(recourse_up - recourse_down - g_shed) + node_L@(l_shed) + node_W@(w_realized_error)) <= f_margin_down_param]            
        #RT_sched_constraints += [-PTDF@(node_G@(p_g_param + recourse_up - recourse_down - g_shed) + node_L@(l_shed-grid['Pd']) + node_W@(grid['w_exp'] + w_realized_error)) <= grid['Line_Capacity'].reshape(-1)]            
                
        # balancing
        RT_sched_constraints += [ recourse_up.sum() - recourse_down.sum() -g_shed.sum() + w_realized_error.sum() + l_shed.sum() == 0]
        RT_sched_constraints += [cost_RT == self.c_viol*(g_shed.sum() + l_shed.sum()) ]
            
        objective_funct = cp.Minimize( cost_RT ) 
        rt_problem = cp.Problem(objective_funct, RT_sched_constraints)
         
        self.rt_layer = CvxpyLayer(rt_problem, parameters=[r_up_param, r_down_param, f_margin_up_param, f_margin_down_param, w_realized_error],
                                           variables = [recourse_up, recourse_down, g_shed, l_shed, cost_RT] )


    def forward(self, sol_dict, w_error_scen, return_cong_slack = False):
        
        p_hat = torch.FloatTensor(sol_dict['p_da'])
        r_up_hat = torch.FloatTensor(sol_dict['r_up'])
        r_down_hat = torch.FloatTensor(sol_dict['r_down'])
        f_margin_up_hat = torch.FloatTensor(sol_dict['f_margin_up'])
        f_margin_down_hat = torch.FloatTensor(sol_dict['f_margin_down'])
                                                  
        # Project to feasible set (might incur numerical errors)
        r_up_hat_proj = torch.maximum(torch.minimum(r_up_hat, self.Pmax - p_hat), self.Pmin)
        r_down_hat_proj = torch.maximum(torch.minimum(r_down_hat, p_hat), self.Pmin)
        
        f_margin_up_hat_proj = torch.maximum(f_margin_up_hat, torch.zeros(self.grid['n_lines']) ) 
        f_margin_down_hat_proj = torch.maximum(f_margin_down_hat, torch.zeros(self.grid['n_lines']) ) 
        

        # RT dispatch cost (penalize violations)
        try:
            rt_output = self.rt_layer(r_up_hat_proj, r_down_hat_proj, f_margin_up_hat_proj, f_margin_down_hat_proj, 
                                              w_error_scen, solver_args={'max_iters':50_000, "solve_method": "ECOS"})                
        except:            
            rt_output = self.rt_layer(r_up_hat_proj, r_down_hat_proj, f_margin_up_hat_proj, f_margin_down_hat_proj, 
                                              w_error_scen, solver_args={'max_iters':50_000, "solve_method": "SCS"})                
            
        cost_RT_hat = rt_output[-1]
        rt_loss = cost_RT_hat.mean()

        if return_cong_slack == False:
            
            return to_np(cost_RT_hat)