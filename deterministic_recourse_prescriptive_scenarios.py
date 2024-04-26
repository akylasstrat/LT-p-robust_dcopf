# -*- coding: utf-8 -*-
"""
Learning prescriptive scenarios/ deterministic formulation with redispatch modeled with recourse actions

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


def oos_cost_estimation(realizations, sol_dict, grid, c_viol = 2*1e3):
    
    recourse_actions = -(sol_dict['A']@realizations.T).T
    

    # exceeding reserves
    aggr_rup_violations = np.maximum( recourse_actions - sol_dict['r_up'], 0).sum()
    aggr_rdown_violations = np.maximum( -recourse_actions - sol_dict['r_down'], 0).sum()
                             
                                                                                 
    # exceeding line rating
    rt_injections = (grid['PTDF']@(grid['node_G']@recourse_actions.T + grid['node_W']@realizations.T)).T
    
    aggr_f_margin_up_violations = np.maximum( rt_injections - sol_dict['f_margin_up'], 0).sum()
    aggr_f_margin_down_violations = np.maximum( -rt_injections - sol_dict['f_margin_down'], 0).sum()
    
    print('Times of violations')    
    print('Upward reserve margin:', ((recourse_actions - sol_dict['r_up'])>0).sum() )
    print('Downward reserve margin:', ((recourse_actions - sol_dict['r_down'])>0).sum() )
    print('Upward line capacity margin:', ((rt_injections - sol_dict['f_margin_up'])>0).sum() )
    print('Downward line capacity margin:', ((-rt_injections - sol_dict['f_margin_down'])>0).sum() )

    print('Max exceedence')    
    print('Upward reserve margin:',  np.maximum( recourse_actions - sol_dict['r_up'], 0).max() )
    print('Downward reserve margin:', np.maximum( -recourse_actions - sol_dict['r_down'], 0).max() )
    print('Upward line capacity margin:', np.maximum( rt_injections - sol_dict['f_margin_up'], 0).max() )
    print('Downward line capacity margin:', np.maximum( -rt_injections - sol_dict['f_margin_down'], 0).max() )
    
    rt_cost = c_viol*(aggr_rup_violations + aggr_rdown_violations + aggr_f_margin_up_violations + aggr_f_margin_down_violations)
    return(rt_cost)

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
# windloc = np.array([3, 5]) - 1  # standard wind farm location, configuration A
windloc = np.array([3, 2]) - 1  # configuration B

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
# create a large set of forecast errors from different wind scenarios

N_samples = 2000
c_viol = 1e4
train_errors = create_wind_errors(N_samples, grid['w_cap'], grid['w_exp'], distr = 'normal', plot = True, std = 0.25, seed = 0)


# Upper and lower bounds for errors

UB = grid['w_cap'] - grid['w_exp']
LB = - grid['w_exp']

#%% 

# Data-driven uncertainty box
H_bound = []
h_bound = []

quantiles = [0.05, 0.95]

h_lb = np.quantile(train_errors, quantiles[0], axis = 0).reshape(-1)
h_ub = np.quantile(train_errors, quantiles[1], axis = 0).reshape(-1)

#h_ub = np.array([1, 4])

H_bound = np.row_stack((-np.eye(grid['n_wind']), np.eye(grid['n_wind'])))
h_dd_box = np.row_stack((-h_lb, h_ub)).reshape(-1)

h_bound = np.row_stack((-LB, UB)).reshape(-1)


fig, ax = plt.subplots(figsize = (6,4))

plt.scatter(train_errors[:,0], train_errors[:,1], alpha = 0.5)

# data-driven box
patches = []    
dd_box_vert = [r for r in itertools.product([h_lb[0], h_ub[0]], 
                                         [h_lb[1], h_ub[1]])]
#%%
dd_box_vert.insert(2, dd_box_vert[-1])
dd_box_vert = dd_box_vert[:-1]

dd_box = Polygon(np.array(dd_box_vert), fill = False, edgecolor = 'black', linewidth = 2, label = 'DD Box$_{98\%}$')
patches.append(dd_box)
ax.add_patch(dd_box)

# full box
patches = []    
box_vert = [r for r in itertools.product([LB[0], UB[0]], [LB[1], UB[1]])]
box_vert.insert(2, box_vert[-1])
box_vert = box_vert[:-1]

box = Polygon(np.array(box_vert), fill = False, 
                  edgecolor = 'black', linestyle = '--', linewidth = 2, label = 'Full Box')
patches.append(box)
ax.add_patch(box)

plt.ylim([-2, 2])
plt.xlim([-2, 2])
plt.legend(fontsize = 12, ncol = 2)
plt.xlabel('Error 1')
plt.ylabel('Error 2')
plt.show()

# solve robust OPF
dd_box_solutions = robust_dcopf_polyhedral(H_bound, h_bound, w_exp, grid, loss = 'cost', verbose = -1)

#tt = robust_dcopf_scenarios(np.array(dd_box_vert), w_expected, grid)

#%% Robust with recourse and fixed number of scenarios
w_scenarios = np.array(dd_box_vert)
w_expected = grid['w_exp']

# test different box intervals/ coverage
alpha = [1, .95, .90]

solution_dictionaries = {}

print('Optimizing for differnt intervals')
for a in np.array(alpha).round(2):
    # find respective quantiles
    if a == 1:        
        quantiles = [1e-5, 1-1e-5]
    else:
        quantiles = [0 + (1-a)/2, 1 - (1-a)/2]
    
    print(f'Quantiles: {quantiles}')
    
    temp_h_lb = np.quantile(train_errors, quantiles[0], axis = 0).reshape(-1)
    temp_h_ub = np.quantile(train_errors, quantiles[1], axis = 0).reshape(-1)
    
    temp_vertices = [r for r in itertools.product([temp_h_lb[0], temp_h_ub[0]], 
                                                  [temp_h_lb[1], temp_h_ub[1]])]
    temp_vertices = np.array(temp_vertices)
    #temp_vertices.insert(2, temp_vertices[-1])
    #temp_vertices = temp_vertices[:-1]
    
    
    temp_box_scen_solution = robust_dcopf_scenarios(temp_vertices, w_expected, grid)
    solution_dictionaries[f'Box_{int(100*a)}'] = temp_box_scen_solution


#box_scen_solution = robust_dcopf_scenarios(w_scenarios, w_expected, grid)
#box_scen_solution = robust_dcopf_scenarios(w_scenarios, w_expected, grid)
#box_scen_solution = robust_dcopf_scenarios(w_scenarios, w_expected, grid)

#%% CAISO's approach

aggr_error = train_errors.sum(1)
# histogram-based aggregated error bounds (system-wide requirement)
aggr_lb, aggr_ub = np.quantile(aggr_error, [0.025, 0.975])
# distribution of to nodes based on 

caiso_h_lb = aggr_lb*(train_errors.std(0)/train_errors.std(0).sum())
caiso_h_ub = aggr_ub*(train_errors.std(0)/train_errors.std(0).sum())

caiso_scen = np.vstack((caiso_h_lb, caiso_h_ub))

temp_box_scen_solution = robust_dcopf_scenarios(caiso_scen, w_expected, grid)
solution_dictionaries[f'CAISO_{int(90)}'] = temp_box_scen_solution

#%%

UB_init = grid['w_cap'] - grid['w_exp'] - 1
LB_init = - grid['w_exp'] + 1

from torch_layers import *

patience = 30
batch_size = 2000
num_epoch = 1000

num_scen = 3
w_scenarios = np.array(box_vert)

# support of uncertain parameters (used in projection step, avoid infeasibilities)
# 2*number of uncertainties: first row is always Upper Bound
support = np.vstack((UB, LB))
w_expected = grid['w_exp']
num_uncertainties = len(grid['w_exp'])

w_scenarios = np.random.normal(loc = 0, scale = 0.5, size = (num_scen, num_uncertainties))

tensor_trainY = torch.FloatTensor(train_errors)

train_data_loader = create_data_loader([tensor_trainY], batch_size = batch_size)
valid_data_loader = create_data_loader([tensor_trainY], batch_size = batch_size)

#tt = torch.FloatTensor(np.array([[0.2, 0.2], [0.1, .0]]))
#scen_robust_opf_model = Scenario_Robust_OPF(2, 2,  tt, grid)
#test = scen_robust_opf_model.forward( tt )

scen_robust_opf_model = Scenario_Robust_OPF(num_uncertainties, num_scen, w_scenarios, support, grid, c_viol = c_viol)
optimizer = torch.optim.Adam(scen_robust_opf_model.parameters(), lr = 1e-1)
scen_robust_opf_model.train_model(train_data_loader, valid_data_loader, optimizer, epochs = num_epoch, patience = patience, validation = False)

# solve problem with learned scenarios
prescriptive_scenarios = scen_robust_opf_model.w_scenarios_param.detach().numpy()
cost_driven_solution = robust_dcopf_scenarios(prescriptive_scenarios, w_expected, grid)

solution_dictionaries['Cost-driven'] = cost_driven_solution
#%% Visualize learned convex hull

points = to_np(scen_robust_opf_model.w_scenarios_param)
hull = ConvexHull(points)


fig, ax = plt.subplots(figsize = (6,4))

plt.scatter(train_errors[:,0], train_errors[:,1], alpha = 0.5)

# data-driven box
patches = []    
box_vert = [r for r in itertools.product([h_lb[0], h_ub[0]], 
                                         [h_lb[1], h_ub[1]])]
box_vert.insert(2, box_vert[-1])
box_vert = box_vert[:-1]

box = Polygon(np.array(box_vert), fill = False, edgecolor = 'black', linewidth = 2, label = 'DD Box$_{98\%}$')
patches.append(box)
ax.add_patch(box)

# full box
patches = []    
box_vert = [r for r in itertools.product([LB[0], UB[0]], [LB[1], UB[1]])]
box_vert.insert(2, box_vert[-1])
box_vert = box_vert[:-1]

box = Polygon(np.array(box_vert), fill = False, 
                  edgecolor = 'black', linestyle = '--', linewidth = 2, label = 'Full Box')
patches.append(box)
ax.add_patch(box)

# CAISO scen
plt.scatter(caiso_scen[:,0], caiso_scen[:,1], color = 'yellow', marker = '+', label = 'CAISO', s = 200, linewidth = 3)


# Cost-driven convex hull
points = to_np(scen_robust_opf_model.w_scenarios_param)
plt.plot(points[:,0], points[:,1], 's', color = 'tab:red', label = 'Cost-driven Scen')
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-', color = 'tab:red', linestyle = '--', lw = 2)    
#plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)

plt.ylim([-2, 2])
plt.xlim([-2, 2])
plt.legend(fontsize = 12)
plt.xlabel('Error 1')
plt.ylabel('Error 2')

plt.show()

#%% Visualize congestions for learned policy

# estimate a grid of optimal dispatches
w1_error_grid = np.arange(LB[0], UB[0], 0.1)
w2_error_grid = np.arange(LB[1], UB[1], 0.1)

xv, yv = np.meshgrid(w1_error_grid, w2_error_grid)
w_joint_grid = np.array([xv.ravel(), yv.ravel()]).T
#%%
#w_error_grid = np.ones((len(w_joint),3))

grid_cost = RT_Clearing(grid, 2, c_viol).forward(solution_dictionaries['Cost-driven'], torch.FloatTensor(w_joint_grid))

cost_surface = grid_cost.reshape(len(xv),-1)
error1_surface = w_joint_grid[:,0].reshape(len(xv),-1)
error2_surface = w_joint_grid[:,1].reshape(len(xv),-1)

# 3-d surface plot
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize = (5,5))

# Plot the surface.
surf = ax.plot_surface(xv, yv, cost_surface, cmap=cm.coolwarm, alpha = 0.75,
                       linewidth=0, antialiased=False)


# Customize the z axis.
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

ax.set_xlabel('Error 1' )
ax.set_ylabel('Error 2' )
#ax.set_ylabel('$d_'+str(ind_var_d[1])+'$ (MW)' )
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink = 0.25, aspect = 7)
plt.show()

cs = plt.contourf(xv, yv, cost_surface,
    colors=['#808080', '#A0A0A0', '#C0C0C0'], extend='both')
#cs.cmap.set_over('red')
#cs.cmap.set_under('blue')
cs.changed()

#%%
# Plot system regions with the same sets of binding constraints
sr1 = plt.contourf(xv, yv, cost_surface, alpha = 1,
    colors=['#808080', '#A0A0A0', '#C0C0C0'], extend='both')
sr1.cmap.set_over('white')
sr1.cmap.set_under('tab:blue')
sr1.changed()

sr2 = plt.contourf(xv, yv, cost_surface, alpha = .5,
    colors=['#808080', '#A0A0A0', '#C0C0C0'], extend='both')
sr2.cmap.set_over('white')
sr2.cmap.set_under('tab:green')
sr2.changed()

#%% Solve for all the grid, find when slacks are activated

opf_model = dc_opf_model(grid, demands_grid, horizon = 1, network = True, plot = False)

g_opt_path = []
Slacks = []
for d_i in demands_grid:
    opf_model.setParam('OutputFlag', 0)
    c1 = opf_model.addConstr(opf_model._vars['node_d_i'] == d_i.reshape(-1))
    opf_model.optimize()
    print(opf_model.CBasis)
    # 0 (basic), -1 non-basic
    Slacks.append([np.abs(c.Slack) > 0 for c in opf_model.getConstrs()])
    g_opt_path.append(opf_model._vars['p_G'].X)
    for c in [c1]:
        opf_model.remove(c)            
Slacks = 1*np.array(Slacks)
g_opt_path = np.array(g_opt_path)
cost_opt = g_opt_path@grid['Cost']


#%% Out-of-sample test
from torch_layers import *

N_test = 2000
distr = 'normal'

test_errors = create_wind_errors(N_test, grid['w_cap'], grid['w_exp'], std = 0.25, seed = 1)


#xx = rt_clearing(cost_driven_solution, grid, c_viol, test_errors)
#yy = rt_clearing(box_scen_solution, grid, c_viol, test_errors)

output = pd.DataFrame(data = [], columns = ['DA_cost', 'RT_cost', 'Total_cost'])

for i, method in enumerate(solution_dictionaries.keys()):
    temp_sol = solution_dictionaries[method]
    temp_output = pd.DataFrame(data = [], columns = ['DA_cost', 'RT_cost', 'Total_cost'], index = [method])
    
    temp_output.loc[method]['DA_cost'] = temp_sol['da_cost']
    temp_output.loc[method]['RT_cost'] = RT_Clearing(grid, 2, c_viol).forward(temp_sol, torch.FloatTensor(test_errors)).mean()
    temp_output.loc[method]['Total_cost'] = temp_output.loc[method]['DA_cost'] + temp_output.loc[method]['RT_cost']

    output = pd.concat([output, temp_output])
    
fig, ax = plt.subplots()
output.T.plot(kind = 'bar', ax = ax)
plt.show()







