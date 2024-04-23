# -*- coding: utf-8 -*-
"""
Learning prescriptive scenarios/ deterministic formulation with recourse dispatch 

Conditional to other features/ parameters/ contextual information

@author: a.stratigakos
"""

import numpy as np
import pandas as pd
from math import ceil
from itertools import product

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch
import gurobipy as gp
import itertools
#import pickle

from sklearn.model_selection import train_test_split
from scipy.spatial import ConvexHull, convex_hull_plot_2d


import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.patches import Ellipse
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch
import matplotlib.transforms as transforms

plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.dpi'] = 600
plt.rcParams['figure.figsize'] = (3.5, 2) # Height can be changed
plt.rcParams['font.size'] = 7
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams["mathtext.fontset"] = 'dejavuserif'

###### RT market layer

def rt_clearing(sol_dict, grid, c_viol, realized_errors, plot = False, verbose = 0):
    ''' Clears the forward (DA) market/ DC-OPF, returns the solutions in dictionary. 
        Creates the problem once in GUROBI, solves for the length of data using horizon step.
        - grid: dictionary with the details of the network
        - demands: net load demands at each node
        - network: if True, solves a DC-OPF, else solves an Economic Dispatch problem
        - horizon: the solution horizon (24 for solving the DA horizon)
        - return_ave_cpu: estimates average cpu time to solve one instance
        - verbose: if ~0, prints GUROBI output
        - plot: if True, creates some plots for check '''
        
    # Declare model parameters and variables
    sol_dict_copy = sol_dict.copy()
    
    # projections to feasible set
    p_hat_proj = np.maximum(np.minimum(sol_dict_copy['p_da'], grid['Pmax']), grid['Pmin'])
    r_up_hat_proj = np.maximum(np.minimum(sol_dict_copy['r_up'], grid['Pmax'] - p_hat_proj), grid['Pmin'])
    r_down_hat_proj = np.maximum(np.minimum(sol_dict_copy['r_down'], p_hat_proj), grid['Pmin'])
    
    f_margin_up_hat_proj = np.maximum(sol_dict_copy['f_margin_up'], np.zeros(grid['n_lines']) ) 
    f_margin_down_hat_proj = np.maximum(sol_dict_copy['f_margin_down'], np.zeros(grid['n_lines']) ) 

    n_samples = len(realized_errors)
    m = gp.Model()
    m.setParam('OutputFlag', verbose)
    # Parameters
    Pmax = grid['Pmax']
    node_G = grid['node_G']
    node_L = grid['node_L']
    Cost = grid['Cost']
    PTDF = grid['PTDF']

    # DA Variables
    recourse_up = m.addMVar((grid['n_unit']), vtype = gp.GRB.CONTINUOUS, lb = 0)
    recourse_down = m.addMVar((grid['n_unit']), vtype = gp.GRB.CONTINUOUS, lb = 0)
    w_realized_error = m.addMVar((grid['n_wind']), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)

    l_shed = m.addMVar((grid['n_loads']), vtype = gp.GRB.CONTINUOUS, lb = 0)
    g_shed = m.addMVar((grid['n_unit']), vtype = gp.GRB.CONTINUOUS, lb = 0)
    
    
    #### Problem Constraints
    #generator technical limits
    m.addConstr( recourse_up <= r_up_hat_proj)
    m.addConstr( recourse_down <= r_down_hat_proj)
    m.addConstr( PTDF@(node_G@(recourse_up - recourse_down - g_shed) + node_L@(l_shed) + grid['node_W']@(w_realized_error)) <= f_margin_up_hat_proj)
    m.addConstr( -PTDF@(node_G@(recourse_up - recourse_down - g_shed) + node_L@(l_shed) + grid['node_W']@(w_realized_error)) <= f_margin_down_hat_proj)
    m.addConstr(  recourse_up.sum() - recourse_down.sum() -g_shed.sum() + w_realized_error.sum() + l_shed.sum() == 0 )
    
                         
    # Set objective and solve
    m.setObjective(c_viol*(g_shed.sum() + l_shed.sum()), gp.GRB.MINIMIZE)                    
    
    total_cost = []
    for i in range(n_samples):
        
        c1 = m.addConstr(w_realized_error == realized_errors[i])
        m.optimize()

        total_cost.append(m.ObjVal)
        
        for cosntr in [c1]:
            m.remove(cosntr)
        m.reset()
    return np.array(total_cost)
    
def dc_opf(grid, demands, network = True, plot = False, verbose = 0, return_ave_cpu = False):
    ''' Clears the forward (DA) market/ DC-OPF, returns the solutions in dictionary. 
        Creates the problem once in GUROBI, solves for the length of data using horizon step.
        - grid: dictionary with the details of the network
        - demands: net load demands at each node
        - network: if True, solves a DC-OPF, else solves an Economic Dispatch problem
        - horizon: the solution horizon (24 for solving the DA horizon)
        - return_ave_cpu: estimates average cpu time to solve one instance
        - verbose: if ~0, prints GUROBI output
        - plot: if True, creates some plots for check '''
        
    # Declare model parameters and variables
    horizon = 1
    n_samples = len(demands)
    m = gp.Model()
    m.setParam('OutputFlag', verbose)
    # Parameters
    Pmax = grid['Pmax']
    node_G = grid['node_G']
    node_L = grid['node_L']
    Cost = grid['Cost']
    PTDF = grid['PTDF']
    VoLL = grid['VOLL']
    VoWS = grid['gshed']

    # DA Variables
    p_G = m.addMVar((grid['n_unit'], horizon), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'p_G')
    slack_u = m.addMVar((grid['n_loads'], horizon), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'slack_up')
    slack_d = m.addMVar((grid['n_loads'], horizon), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'slack_down')
    
    flow_da = m.addMVar((grid['n_lines'], horizon), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    #theta_da = m.addMVar((grid['n_nodes'], horizon), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    
    # DA variables for uncertain parameters    
    node_net_forecast_i = m.addMVar((grid['n_loads'], horizon), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'node demand pred')

    # Store solutions in dict
    Det_solutions = {'p': [], 'flow_da': [], 'theta_da': [], 's_up':[], 's_down':[]}

    #### Problem Constraints
    
    #generator technical limits
    m.addConstrs( p_G[:,t] <= Pmax.reshape(-1) for t in range(horizon))

    if network == True:
        node_inj = m.addMVar((grid['n_nodes'], horizon), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)

        m.addConstrs(node_inj[:,t] == (node_G@p_G[:,t] - node_L@node_net_forecast_i[:,t] - node_L@slack_d[:,t] 
                                           + node_L@slack_u[:,t]) for t in range(horizon))

        m.addConstrs(flow_da[:,t] == PTDF@node_inj[:,t] for t in range(horizon))            
        
        m.addConstrs(flow_da[:,t] <= grid['Line_Capacity'].reshape(-1) for t in range(horizon))
        m.addConstrs(flow_da[:,t] >= -grid['Line_Capacity'].reshape(-1) for t in range(horizon))
        
    # Node balance for t for DC-OPF
    m.addConstrs( p_G[:,t].sum() + slack_u[:,t].sum() - slack_d[:,t].sum() == node_net_forecast_i[:,t].sum() for t in range(horizon))

    # DA cost for specific day/ expression
    DA_cost = sum([Cost@p_G[:,t] + slack_u[:,t].sum()*VoLL + slack_d[:,t].sum()*VoWS for t in range(horizon)]) 
    
    # Loop over days, optimize each day
    ave_cpu_time = 0
    for i in range(n_samples):
        if i%500==0:
            print('Sample: ', i)
            
        # demand is in MW (parameter that varies)
        c1 = m.addConstrs(node_net_forecast_i[:,t] == demands[i*horizon:(i+1)*horizon].T[:,t] for t in range(horizon)) 
                         
        # Set objective and solve
        m.setObjective(DA_cost, gp.GRB.MINIMIZE)                    
        m.optimize()
        ave_cpu_time += m.runtime/n_samples
        #print(m.ObjVal)
        # sanity check 
        if plot:
            if i%10==0:
                plt.plot(p_G.X.T.sum(1), label='p_Gen')
                plt.plot(p_G.X.T.sum(1) + slack_d.X.T.sum(1), '--', label='p_Gen+Slack')
                plt.plot(node_net_forecast_i.X.T.sum(1), 'o', color='black', label='Net Forecast')
                plt.legend()
                plt.show()
        if i%10==0:
            try:
                assert((flow_da.X.T<=grid['Line_Capacity']+.001).all())
                assert((flow_da.X.T>=-grid['Line_Capacity']-.001).all())
            except:
                print('Infeasible flows')
        # append solutions
        Det_solutions['p'].append(p_G.X)
        Det_solutions['s_up'].append(slack_u.X)
        Det_solutions['s_down'].append(slack_d.X)
        Det_solutions['flow_da'].append(flow_da.X)
        #Det_solutions['theta_da'].append(theta_da.X)
            
        # remove constraints with uncertain parameters, reset solution
        for cosntr in [c1]:
            m.remove(cosntr)
        m.reset()

    if return_ave_cpu:
        return Det_solutions, ave_cpu_time
    else:
        return Det_solutions


def robust_dcopf_polyhedral(H, h, w_expected, grid, policy='affine', network = True, slack_var = False, rho = 1, loss = 'cost', verbose = -1, horizon = 1):
    ''' 
    Solves robust DCOPF for polyhedral uncertainty set (covers box uncertainty set as a special case, but it's less efficient that the dual norm)
    Linear decision rules for a recourse policy of generator dispatch decisions    
    Forecast errors take values in uncertainty set: H@x <= h (box is a special case, but the dual norm is more efficient)
    '''
        
    # Grid Parameters
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
    
    # number of robust constraints to be reformulated
    
    num_constr = len(h)
    
    m = gp.Model()
    if verbose == -1:
        m.setParam('OutputFlag', 0)
    
    #m.setParam('NumericFocus', 1)
    #m.setParam('Method', 0)
    #m.setParam('Crossover', 0)
    #m.setParam('Threads', 8)
        
    ### variables    
    # DA Variables
    p_G = m.addMVar((grid['n_unit']), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'p_G')
    r_up_G = m.addMVar((grid['n_unit']), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'upward reserve')
    r_down_G = m.addMVar((grid['n_unit']), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'downward reserve')
    
    # linear decision rules: p(xi) = p_da + W@xi
    W = m.addMVar((grid['n_unit'], grid['n_wind']), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'linear decision rules')
    
    slack_u = m.addMVar((grid['n_loads']), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'slack_up')
    slack_d = m.addMVar((grid['n_loads']), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'slack_down')
    
    f_margin_up = m.addMVar((grid['n_lines']), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'upward reserve')
    f_margin_down = m.addMVar((grid['n_lines']), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'downward reserve')    
    #flow_da = m.addMVar((grid['n_lines']), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)    

    
    ### DA constraints 
    m.addConstr( p_G + r_up_G <= Pmax.reshape(-1))
    m.addConstr( p_G - r_down_G >= 0)
    
    # balancing all errors
    m.addConstr( W.sum(0) == np.ones(grid['n_wind']))
    
    # system balancing constraints (could relax this)
    m.addConstr( p_G.sum() + w_expected.sum() == grid['Pd'].sum())
    
    # network constraints (redundant if robust formulation is applied)        
    m.addConstr(PTDF@(node_G@p_G + node_W@w_expected - node_L@grid['Pd'] ) <= grid['Line_Capacity'].reshape(-1) )
    m.addConstr(PTDF@(node_G@p_G + node_W@w_expected - node_L@grid['Pd'] ) >= -grid['Line_Capacity'].reshape(-1) )

    #m.addConstr(flow_da == PTDF@(node_G@p_G + node_W@w_expected - node_L@grid['Pd'] ) )
    #m.addConstr(flow_da <= grid['Line_Capacity'].reshape(-1) )
    #m.addConstr(flow_da >= -grid['Line_Capacity'].reshape(-1) )

    m.addConstr(grid['Line_Capacity'].reshape(-1) - f_margin_up == PTDF@(node_G@p_G + node_W@w_expected - node_L@grid['Pd'] ) )
    m.addConstr(grid['Line_Capacity'].reshape(-1) - f_margin_down == -PTDF@(node_G@p_G + node_W@w_expected - node_L@grid['Pd'] ) )

    # robust constraints for all xi in H@xi <= h
    # W@xi <= r_down_G
    # -W@xi <= r_up_G
    
    # Reformulation of robust constraints
    # downward reserve bound/ each row are the duals to reformulate each constraints
    lambda_up = m.addMVar((grid['n_unit'], num_constr) , vtype = gp.GRB.CONTINUOUS, lb = 0)
    lambda_down = m.addMVar((grid['n_unit'], num_constr) , vtype = gp.GRB.CONTINUOUS, lb = 0)
    
    m.addConstr( lambda_down@H == W )
    m.addConstr( h@lambda_down.T <= r_down_G)

    m.addConstr( lambda_up@H == -W )
    m.addConstr( h@lambda_up.T <= r_up_G)
            
    #node_inj = m.addMVar((grid['n_nodes'], n_samples), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)        
    if network:
        # dual variables: per line and per features, for upper/lower bound
        lambda_f_up = m.addMVar((grid['n_lines'], num_constr) , vtype = gp.GRB.CONTINUOUS, lb = 0)
        lambda_f_down = m.addMVar((grid['n_lines'], num_constr) , vtype = gp.GRB.CONTINUOUS, lb = 0)
        
        # aux. variables to hold expressions
        #node_rob_inj = m.addMVar((grid['n_nodes'], n_feat), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
        #flow_rob = m.addMVar((grid['n_lines'], n_feat), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
            
        # New constraints
        #m.addConstr(node_rob_inj == node_G@coef - node_L@coef_d)                
        #m.addConstr(flow_rob == PTDF@node_rob_inj)
        
        #!!!!!! Error here
        #  Upper: (-W + diag(xi))@xi  <= f_up - flow_da, H@xi <= h
        #  Lower: (W - diag(xi))@xi  <= f_up + flow_da, H@xi <= h
        # PTDF@node_g@(W@xi) + PTDF@node_w@(xi)
        # reformulation of robust constraints for line flows
        m.addConstr( lambda_f_up@H == PTDF@(-node_G@W + node_W)  )
        m.addConstr( h@lambda_f_up.T <= f_margin_up )

        m.addConstr( lambda_f_down@H == -PTDF@(-node_G@W + node_W) )
        m.addConstr( h@lambda_f_down.T <= f_margin_down)
                                   
    m.setObjective( Cost@p_G + C_r_up@r_up_G + C_r_down@r_down_G, gp.GRB.MINIMIZE)             
    m.optimize()
    try:       
        rob_sol = {}
        rob_sol['da_cost'] = m.objVal
        rob_sol['p_da'] = p_G.X
        rob_sol['r_up'] = r_up_G.X
        rob_sol['r_down'] = r_down_G.X
        rob_sol['A'] = W.X
        rob_sol['f_margin_up'] = f_margin_up.X
        rob_sol['f_margin_down'] = f_margin_down.X
        
        return rob_sol

    except:
        print('Infeasible solution')
        # scale cost back to aggregate    
        return 1e10, [], []
            

def robust_dcopf_scenarios(error_scenarios, w_expected, grid, network = True, slack_var = False, rho = 1, verbose = -1, horizon = 1):
    ''' 
    Solves robust DCOPF for polyhedral uncertainty set (covers box uncertainty set as a special case, but it's less efficient that the dual norm)
    Linear decision rules for a recourse policy of generator dispatch decisions    
    Forecast errors take values in uncertainty set: H@x <= h (box is a special case, but the dual norm is more efficient)
    Scenarios: H@x == h
    '''
    
    num_scen = len(error_scenarios)
    
    # Grid Parameters
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
    
    # number of robust constraints to be reformulated
        
    m = gp.Model()
    if verbose == -1:
        m.setParam('OutputFlag', 0)
            
    ### variables    
    # DA Variables
    p_G = m.addMVar((grid['n_unit']), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'p_G')
    r_up_G = m.addMVar((grid['n_unit']), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'upward reserve')
    r_down_G = m.addMVar((grid['n_unit']), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'downward reserve')
    
    # recourse per scenarios
    p_recourse = m.addMVar((grid['n_unit'], num_scen), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)
    #p_recourse_down = m.addMVar((grid['n_unit'], num_scenarios), vtype = gp.GRB.CONTINUOUS, lb = 0)
    
    f_margin_up = m.addMVar((grid['n_lines']), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'upward reserve')
    f_margin_down = m.addMVar((grid['n_lines']), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'downward reserve')    

    ### DA constraints 
    m.addConstr( p_G + r_up_G <= Pmax.reshape(-1))
    m.addConstr( p_G - r_down_G >= 0)

    # system balancing constraints (could relax this)
    m.addConstr( p_G.sum() + w_expected.sum() == grid['Pd'].sum())

    # network constraints (redundant if robust formulation is applied)        
    m.addConstr(PTDF@(node_G@p_G + node_W@w_expected - node_L@grid['Pd'] ) <= grid['Line_Capacity'].reshape(-1) )
    m.addConstr(PTDF@(node_G@p_G + node_W@w_expected - node_L@grid['Pd'] ) >= -grid['Line_Capacity'].reshape(-1) )

    m.addConstr(grid['Line_Capacity'].reshape(-1) - f_margin_up == PTDF@(node_G@p_G + node_W@w_expected - node_L@grid['Pd'] ) )
    m.addConstr(grid['Line_Capacity'].reshape(-1) - f_margin_down == -PTDF@(node_G@p_G + node_W@w_expected - node_L@grid['Pd'] ) )
    
    #### Constraints per scenario
    
    # aggregate error
    m.addConstrs( p_recourse[:,s].sum() + error_scenarios[s].sum() == 0 for s in range(num_scen))
    
    m.addConstrs( p_recourse[:,s] <= r_up_G for s in range(num_scen))
    m.addConstrs( -p_recourse[:,s] <= r_down_G for s in range(num_scen))
            
    if network:
        
        m.addConstrs( PTDF@(node_G@p_recourse[:,s] + node_W@error_scenarios[s]) <= f_margin_up for s in range(num_scen))
        m.addConstrs( -PTDF@(node_G@p_recourse[:,s] + node_W@error_scenarios[s]) <= f_margin_down for s in range(num_scen))
                                   
    m.setObjective( Cost@p_G + C_r_up@r_up_G + C_r_down@r_down_G, gp.GRB.MINIMIZE)             
    m.optimize()
    
    try:       
        rob_sol = {}
        rob_sol['da_cost'] = m.objVal
        rob_sol['p_da'] = p_G.X
        rob_sol['r_up'] = r_up_G.X
        rob_sol['r_down'] = r_down_G.X
        rob_sol['p_recourse'] = p_recourse.X
        rob_sol['f_margin_up'] = f_margin_up.X
        rob_sol['f_margin_down'] = f_margin_down.X
        
        return rob_sol

    except:
        print('Infeasible solution')
        # scale cost back to aggregate    
        return 1e10, [], []
    
def robust_dcopf_box(UB, LB, H, h, w_expected, grid, network = True, slack_var = False, verbose = -1, horizon = 1):
    ''' 
    Solves robust DCOPF for box uncertainty set, reformulates using the dual norm
    Linear decision rules for a recourse policy of generator dispatch decisions    
    Uncertainty set: l_inf: || xi ||_inf <= rho
    '''
        
    # Grid Parameters
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
    
    n_feat = grid['n_wind']
    # number of robust constraints to be reformulated
    
    num_constr = len(h)
    
    m = gp.Model()
    if verbose == -1:
        m.setParam('OutputFlag', 0)
    
            
    #formulation with primitive factors
    u_nom = (UB+LB)/2
    P = np.diag((UB-LB)/2)
    rho = 1
        
    #m.setParam('NumericFocus', 1)
    #m.setParam('Method', 0)
    #m.setParam('Crossover', 0)
    #m.setParam('Threads', 8)
        
    ### variables    
    # DA Variables
    p_G = m.addMVar((grid['n_unit']), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'p_G')
    r_up_G = m.addMVar((grid['n_unit']), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'upward reserve')
    r_down_G = m.addMVar((grid['n_unit']), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'downward reserve')
    
    # linear decision rules: p(xi) = p_da + W@xi
    W = m.addMVar((grid['n_unit'], grid['n_wind']), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'linear decision rules')
    
    slack_u = m.addMVar((grid['n_loads']), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'slack_up')
    slack_d = m.addMVar((grid['n_loads']), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'slack_down')

    f_margin_up = m.addMVar((grid['n_lines']), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'upward reserve')
    f_margin_down = m.addMVar((grid['n_lines']), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'downward reserve')    
    flow_da = m.addMVar((grid['n_lines']), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)    

    
    ### DA constraints 
    m.addConstr( p_G + r_up_G <= Pmax.reshape(-1))
    m.addConstr( p_G - r_down_G >= 0)
    
    # balancing all errors
    m.addConstr( W.sum(0) == np.ones(grid['n_wind']))
    
    # system balancing constraints (could relax this)
    m.addConstr( p_G.sum() + w_expected.sum() == grid['Pd'].sum())
    
    # network constraints (redundant if robust formulation is applied)        
    m.addConstr(PTDF@(node_G@p_G + node_W@w_expected - node_L@grid['Pd'] ) <= grid['Line_Capacity'].reshape(-1) )
    m.addConstr(PTDF@(node_G@p_G + node_W@w_expected - node_L@grid['Pd'] ) >= -grid['Line_Capacity'].reshape(-1) )

    #m.addConstr(flow_da == PTDF@(node_G@p_G + node_W@w_expected - node_L@grid['Pd'] ) )
    #m.addConstr(flow_da <= grid['Line_Capacity'].reshape(-1) )
    #m.addConstr(flow_da >= -grid['Line_Capacity'].reshape(-1) )

    m.addConstr(grid['Line_Capacity'].reshape(-1) - f_margin_up == PTDF@(node_G@p_G + node_W@w_expected - node_L@grid['Pd'] ) )
    m.addConstr(grid['Line_Capacity'].reshape(-1) - f_margin_down == -PTDF@(node_G@p_G + node_W@w_expected - node_L@grid['Pd'] ) )

    # robust constraints for all xi 
    # aux variables
    aux_g_up = m.addMVar((grid['n_unit'], n_feat) , vtype = gp.GRB.CONTINUOUS, lb = 0)
    aux_g_down = m.addMVar((grid['n_unit'], n_feat) , vtype = gp.GRB.CONTINUOUS, lb = 0)
            
    # upper/lower bound with l-inf norm and primitive uncertainty
    for g in range(grid['n_unit']):
        m.addConstr(aux_g_up[g] >= -P@W[g])
        m.addConstr(aux_g_up[g] >= P@W[g])
        m.addConstr(aux_g_down[g] >= -P@W[g])
        m.addConstr(aux_g_down[g] >= P@W[g])
    
        m.addConstr( -W[g]@u_nom + rho*aux_g_up[g]@np.ones(n_feat) <= r_up_G[g] )
        m.addConstr( W[g]@(u_nom) + rho*aux_g_down[g]@np.ones(n_feat) <= r_down_G[g] )
    
    #node_inj = m.addMVar((grid['n_nodes'], n_samples), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)        
    if network:
        # dual variables: per line and per features, for upper/lower bound
        #lambda_f_up = m.addMVar((grid['n_lines'], num_constr) , vtype = gp.GRB.CONTINUOUS, lb = 0)
        #lambda_f_down = m.addMVar((grid['n_lines'], num_constr) , vtype = gp.GRB.CONTINUOUS, lb = 0)

        #!!!!!! Error here
        #  Upper: PTDF@node_g@(-W@xi) + PTDF@node_w@(xi) <= f_up - flow_da, H@xi <= h
        #  Lower: -PTDF@node_g@(-W@xi) + PTDF@node_w@(xi) <= f_up + flow_da, H@xi <= h
        # PTDF@node_g@(W@xi) + PTDF@node_w@(xi)
        # reformulation of robust constraints for line flows
        #m.addConstr( lambda_f_up@H == PTDF@(-node_G@W + node_W)  )
        #m.addConstr( h@lambda_f_up.T <= grid['Line_Capacity'] - flow_da )

        #m.addConstr( lambda_f_down@H == -PTDF@(-node_G@W + node_W) )
        #m.addConstr( h@lambda_f_down.T <= grid['Line_Capacity'] + flow_da)
                    
        aux_f_up = m.addMVar((grid['n_lines'], n_feat) , vtype = gp.GRB.CONTINUOUS, lb = 0)
        aux_f_down = m.addMVar((grid['n_lines'], n_feat) , vtype = gp.GRB.CONTINUOUS, lb = 0)
                
        for l in range(grid['n_lines']):

            m.addConstr(aux_f_up[l] >= -(PTDF[l]@(-node_G@W + node_W))@P)
            m.addConstr(aux_f_up[l] >= (PTDF[l]@(-node_G@W + node_W))@P)
            m.addConstr(aux_f_down[l] >= -(PTDF[l]@(-node_G@W + node_W))@P)
            m.addConstr(aux_f_down[l] >= (PTDF[l]@(-node_G@W + node_W))@P)
        
            m.addConstr( PTDF[l]@((-node_G@W + node_W)@u_nom) + rho*aux_f_up[g]@np.ones(n_feat) <= grid['Line_Capacity'][l] - flow_da[l] )
            m.addConstr( -(PTDF[l]@(-node_G@W + node_W)@u_nom) + rho*aux_f_down[g]@np.ones(n_feat) <= grid['Line_Capacity'][l] + flow_da[l] )

        
                                   
    m.setObjective( Cost@p_G + C_r_up@r_up_G + C_r_down@r_down_G, gp.GRB.MINIMIZE)             
    m.optimize()
    try:       
        return m.objVal, p_G.X, r_up_G.X, r_down_G.X, W.X

    except:
        print('Infeasible solution')
        # scale cost back to aggregate    
        return 1e10, [], []

def create_conditional_errors(N, capacity, expected, distr = 'normal', plot = True, std = 0.25, seed = 0):
    m = len(capacity)
    errors = np.zeros((N, m))

    # error = actual - expected
    # max positive error: capacity - expected
    # min negative error: -expected
        
    if distr == 'normal':
        # multivariate normal
        std_Pd = std*expected      
        ub_Pd = capacity
        lb_Pd = np.zeros(m)
        
        mean_Pd = ((ub_Pd-lb_Pd)/2+lb_Pd).reshape(-1)
        
        # generate correlation matrix *must be symmetric    
        np.random.seed(0)
        a = np.random.rand(m, m)        
        R = np.tril(a) + np.tril(a, -1).T
        for i in range(grid['n_wind']): 
            R[i,i] = 1
        # estimate covariance matrix
        S_cov = np.diag(std_Pd)@R@np.diag(std_Pd)        
        # sample demands, project them into support
        np.random.seed(seed)
        samples = np.random.multivariate_normal(np.zeros(mean_Pd.shape[0]), S_cov, size = N_samples).round(2)
        errors = samples

    # Project back to feasible set
    for u in range(m):
        # upper bound/ positive errors == actual - exp >=0
        errors[:,u][errors[:,u] > capacity[u] - expected[u]] = capacity[u] - expected[u]
        #lower bound/ negative error == actual - exp <= 0
        errors[:,u][errors[:,u] < -expected[u]] = -expected[u]
        
    # error scatterplot
    if plot:
        plt.scatter(errors[:,0], errors[:,1])
        plt.show()
    return errors

def create_wind_errors(N, capacity, expected, distr = 'normal', plot = True, std = 0.25, seed = 0):
    m = len(capacity)
    errors = np.zeros((N, m))

    # error = actual - expected
    # max positive error: capacity - expected
    # min negative error: -expected
        
    if distr == 'normal':
        # multivariate normal
        std_Pd = std*expected      
        ub_Pd = capacity
        lb_Pd = np.zeros(m)
        
        mean_Pd = ((ub_Pd-lb_Pd)/2+lb_Pd).reshape(-1)
        
        # generate correlation matrix *must be symmetric    
        np.random.seed(0)
        a = np.random.rand(m, m)        
        R = np.tril(a) + np.tril(a, -1).T
        for i in range(grid['n_wind']): 
            R[i,i] = 1
        # estimate covariance matrix
        S_cov = np.diag(std_Pd)@R@np.diag(std_Pd)        
        # sample demands, project them into support
        np.random.seed(seed)
        samples = np.random.multivariate_normal(np.zeros(mean_Pd.shape[0]), S_cov, size = N_samples).round(2)
        errors = samples

    # Project back to feasible set
    for u in range(m):
        # upper bound/ positive errors == actual - exp >=0
        errors[:,u][errors[:,u] > capacity[u] - expected[u]] = capacity[u] - expected[u]
        #lower bound/ negative error == actual - exp <= 0
        errors[:,u][errors[:,u] < -expected[u]] = -expected[u]
        
    # error scatterplot
    if plot:
        plt.scatter(errors[:,0], errors[:,1])
        plt.show()
    return errors
         
#%%
# parameters of matpower case 5

# nominal demand (pu)
d = np.array([0.0, 3.0, 3.0, 4.0, 0.0])
# nominal capacity (pu)
pmax = np.array([0.4, 1.7, 5.2, 2.0, 6.0])
pmin = np.zeros(len(pmax))
cE = np.array([14.0, 15.0, 30.0, 40.0, 10.0]) # linear cost
cE_quad = np.sqrt(cE * 0.1) # quadratic cost
cR = np.array([80., 80., 15., 30., 80.]) # reserve cost

smax = np.array([4.0, 1.9, 2.2, 1.0, 1.0, 2.4])
ptdf_str  = '-0.193917 0.475895   0.348989  0.0  -0.159538;'
ptdf_str += '-0.437588  -0.258343  -0.189451  0.0  -0.36001;'
ptdf_str += '-0.368495  -0.217552  -0.159538  0.0   0.519548;'
ptdf_str += '-0.193917  -0.524105   0.348989  0.0  -0.159538;'
ptdf_str += '-0.193917  -0.524105  -0.651011  0.0  -0.159538;'
ptdf_str += '0.368495   0.217552   0.159538  0.0   0.48045'
ptdf = np.matrix(ptdf_str)

basemva = 100
genloc = np.array([1, 1, 3, 4, 5]) -1
windloc = np.array([3, 5]) - 1  # standard wind farm location
# windloc = np.array([3, 2]) - 1  # configuration B

smax = np.array([4.0, 1.9, 2.2, 1.0, 1.0, 2.4])


grid = {}
grid['Pmax'] = np.array([0.4, 1.7, 5.2, 2.0, 6.0])
grid['Pmin'] = np.zeros(len(pmax))
grid['Pd'] = d
grid['Line_Capacity'] = smax

grid['Cost'] = cE
grid['C_r_up'] = cR
grid['C_r_down'] = cR
grid['PTDF'] = ptdf

w_exp = np.array([1.0, 1.5])    # nominal wind forecasts/ expected values
w_cap = np.array([2.0, 3.0])

grid['n_unit'] = len(grid['Pmax'])
grid['n_loads'] = len(grid['Pd'])
grid['n_wind'] = len(w_exp)
grid['n_lines'] = grid['PTDF'].shape[0]
grid['n_nodes'] = grid['PTDF'].shape[1]

grid['w_exp'] = w_exp
grid['w_cap'] = w_cap

grid['VOLL'] = 2*1e5
grid['VOWS'] = 2*1e5

grid['node_G'] = np.zeros((grid['n_nodes'], grid['n_unit']))
grid['node_W'] = np.zeros((grid['n_nodes'], grid['n_wind']))

for g, bus in enumerate(genloc):
    grid['node_G'][bus,g] = 1
    
for u, bus in enumerate(windloc):
    grid['node_W'][bus,u] = 1

grid['node_L'] = np.eye((grid['n_nodes']))
#%%
c_viol = 1e4
n_wind = len(grid['w_exp'])
# Create samples of demand + wind + sample of forecast errors (e.g., sampling probabilistic distribution)
N_samples = 4000
N_train_samples = 2000
N_test_samples = N_samples - N_train_samples

# store results
Output_df = pd.DataFrame(data = [], columns = ['DA_cost', 'RT_cost', 'Total_cost'])
solution_dictionaries = {}

demand_samples = np.random.uniform(0.5, 1.1, size = (N_samples, len(grid['Pd'])))*grid['Pd']
# expected values
wind_samples = np.random.uniform(0.5, 1.1, size = (N_samples, len(grid['w_exp'])))*grid['w_exp']

corr = 0.5
wind_error = []
wind_error_samples = [] # only used for plotting
quantiles = [0.025, 0.975]

cond_UB = []
cond_LB = []

for i in range(N_samples):
    # conditional variance
    cond_S_cov = np.array([[(0.15*wind_samples[i,0])**2, 0.15**2*(corr*wind_samples[i,0]*wind_samples[i,1])], 
                  [0.15**2*(corr*wind_samples[i,0]*wind_samples[i,1]), (0.15*wind_samples[i,1])**2 ] ])

    # sample demands, project them into support
    np.random.seed(1234)
    samples = np.random.multivariate_normal(np.zeros(n_wind), cond_S_cov, 1000)
    
    temp_LB, temp_UB = np.quantile(samples, quantiles, 0)
    
    # prediction intervals, we assumed are produced from probabilistic forecasts
    cond_UB.append(temp_UB)
    cond_LB.append(temp_LB)
    
    # realized errors (used for testing)
    wind_error.append(samples[0].reshape(-1))
    wind_error_samples.append(samples)
    
wind_error = np.array(wind_error)    
cond_UB = np.array(cond_UB)
cond_LB = np.array(cond_LB)

# project to the support
wind_error = np.maximum(np.minimum(wind_error, grid['w_cap'] - wind_samples), - wind_samples)

train_wind_error = wind_error[:N_train_samples]
test_wind_error = wind_error[N_train_samples:]

train_demand_samples = demand_samples[:N_train_samples]
test_demand_samples = demand_samples[N_train_samples:]

train_wind_samples = wind_samples[:N_train_samples]
test_wind_samples = wind_samples[N_train_samples:]



#%% Conditional robust solution: solve a robust problem at each day

from torch_layers import *

da_clearing_layer = DA_RobustScen_Clearing(grid, 2, 4)

box_robust_sol = []

for i in range(N_test_samples):
    if i%1000==0: print(f'Observation:{i}')
    # create problem vertices from conditional intervals
    conditional_vertices = [r for r in itertools.product([cond_LB[i,0], cond_UB[i,0]], 
                                             [cond_LB[i,1], cond_UB[i,1]])]
    
    conditional_vertices = np.array(conditional_vertices)

    temp_sol = da_clearing_layer(test_demand_samples[i], test_wind_samples[i], conditional_vertices)
    
    box_robust_sol.append(temp_sol)


solution_dictionaries['Box_95'] = box_robust_sol

    
#%% Learning contextual, cost-driven scenarios
from torch_layers import *

tensor_train_demand = torch.FloatTensor(train_demand_samples)
tensor_train_wind = torch.FloatTensor(train_wind_samples)
tensor_train_wind_error = torch.FloatTensor(train_wind_error)

patience = 20
batch_size = 2000
num_epoch = 1000

Num_scen_list = [3, 4, 5]

train_data_loader = create_data_loader([tensor_train_demand, tensor_train_wind, tensor_train_wind_error], batch_size = batch_size)
valid_data_loader = create_data_loader([tensor_train_demand, tensor_train_wind, tensor_train_wind_error], batch_size = batch_size)

num_uncertainties = len(grid['w_exp'])

mlp_param = {}
mlp_param['input_size'] = grid['n_loads'] + grid['n_wind']
mlp_param['hidden_sizes'] = []


contextual_models_list = []
Predicted_scen_list = []
for num_scen in Num_scen_list:    
    
    # initialize (not sure we need this)
    w_init_error_scen = np.random.normal(loc = 0, scale = 0.5, size = (num_scen, num_uncertainties))
    
    # Train model
    contextual_robust_opf_model = Contextual_Scenario_Robust_OPF(mlp_param, num_uncertainties, num_scen, grid, w_init_error_scen, c_viol = c_viol)
    optimizer = torch.optim.Adam(contextual_robust_opf_model.parameters(), lr = 1e-2, weight_decay=1e-4)
    contextual_robust_opf_model.train_model(train_data_loader, valid_data_loader, optimizer, epochs = num_epoch, patience = patience, validation = False)
    contextual_models_list.append(contextual_robust_opf_model)
    
    # Generate prescriptive scenarios for test set
    predicted_scenarios = contextual_robust_opf_model.predict(torch.FloatTensor(test_demand_samples), torch.FloatTensor(test_wind_samples) )
    Predicted_scen_list.append(predicted_scenarios)
    
    # For each scenario, solve the DA market clearing problem
    cost_driven_sol = []
    
    for i in range(N_test_samples):
        if i == 0: 
            # initialize DA clearing problem for specific number of scenarios
            da_clearing_layer = DA_RobustScen_Clearing(grid, 2, num_scen)
    
        if i%1000==0: print(f'Observation:{i}')
        temp_scen = to_np(predicted_scenarios[i])
        
        #temp_box_scen_solution = robust_dcopf_scenarios(conditional_vertices, train_wind_samples, grid)
        temp_sol = da_clearing_layer(test_demand_samples[i], test_wind_samples[i], temp_scen)
        
        cost_driven_sol.append(temp_sol)
        
    # store the solutions for each test observation
    solution_dictionaries[f'Cost_driven_{num_scen}'] = cost_driven_sol

#%%

# Visualization
colors = ['tab:red', 'tab:orange', 'tab:green']

for i in range(0, N_test_samples, 100):
        
    fig, ax = plt.subplots(figsize = (6,4))        
    # scatter plot of some samples
    plt.scatter(wind_error_samples[i][:100,0], wind_error_samples[i][:100,1], color = 'tab:blue', label = 'Sampled Scenarios', alpha = 0.5)
    
    # box uncertainty, vertices derived from conditional intervals    
    box_scen = [r for r in itertools.product([cond_LB[i,0], cond_UB[i,0]], 
                                             [cond_LB[i,1], cond_UB[i,1]])]
    box_scen = np.array(box_scen)
    
    hull = ConvexHull(box_scen)
    plt.plot(box_scen[:,0], box_scen[:,1], 's', color = 'black', label = '90% Prediction Intervals')
    for j, simplex in enumerate(hull.simplices):
        plt.plot(box_scen[simplex, 0], box_scen[simplex, 1], color = 'black', linestyle = '--', lw = 2)    
    
    for k, num_scen in enumerate(Num_scen_list):
        
        temp_cost_driven_scen = Predicted_scen_list[k][i]
        hull = ConvexHull(temp_cost_driven_scen)
        plt.plot(temp_cost_driven_scen[:,0], temp_cost_driven_scen[:,1], 's', color = colors[k], label = f'Cost-driven_{num_scen} scenarios')
        for j, simplex in enumerate(hull.simplices):
            plt.plot(temp_cost_driven_scen[simplex, 0], temp_cost_driven_scen[simplex, 1], color = colors[k], linestyle = '--', lw = 2)    
                
    plt.ylim([-1, 1])
    plt.xlim([-1, 1])
    plt.title(f'C_viol = {c_viol} \$/MWh', fontsize = 12)
    plt.xlabel('Error 1')
    plt.ylabel('Error 2')
    plt.legend(fontsize = 12, loc = 'upper right')
    plt.show()

    
#%%
# Out-of-sample test
print('Out-of-sample test')
for m, method in enumerate(solution_dictionaries.keys()):
    print(method)    
    
    list_sol_dict = solution_dictionaries[method]
    
    temp_output = pd.DataFrame(data = np.zeros((1,3)), columns = ['DA_cost', 'RT_cost', 'Total_cost'], index = [method])
    
    for i in range(N_test_samples):
        temp_da_cost = list_sol_dict[i]['da_cost'].detach().numpy()
        temp_rt_cost = RT_Clearing(grid, 2, c_viol).forward(list_sol_dict[i], torch.FloatTensor(test_wind_error[i])).mean()
        
        temp_output.loc[method]['DA_cost'] += temp_da_cost/N_test_samples
        temp_output.loc[method]['RT_cost'] += temp_rt_cost/N_test_samples
        temp_output.loc[method]['Total_cost'] += (temp_da_cost + temp_rt_cost)/N_test_samples

    Output_df = pd.concat([Output_df, temp_output])
    
fig, ax = plt.subplots()
Output_df.T.plot(kind = 'bar', ax = ax)
plt.show()







