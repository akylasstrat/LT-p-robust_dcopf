# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 20:34:16 2024

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
# create a large set of forecast errors from different wind scenarios

N_samples = 2000
c_viol = 100
train_errors = create_wind_errors(N_samples, grid['w_cap'], grid['w_exp'], distr = 'normal', plot = True, std = 0.25, seed = 0)


# Upper and lower bounds for errors

UB = grid['w_cap'] - grid['w_exp']
LB = - grid['w_exp']

#%% 

# Data-driven uncertainty box
H_bound = []
h_bound = []

quantiles = [0.01, 0.99]

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

plt.ylim([-2, 2])
plt.xlim([-2, 2])
plt.legend(fontsize = 12, ncol = 2)
plt.xlabel('Error 1')
plt.ylabel('Error 2')
plt.show()
#%%
# solve robust OPF
dd_box_solutions = robust_dcopf_polyhedral(H_bound, h_bound, w_exp, grid, loss = 'cost', verbose = -1)

#%% Scenario approach with redispatch 

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
num_scen = len(box_vert)
w_scenarios = np.array(box_vert).T
w_expected = grid['w_exp']

m = gp.Model()
    
### variables    
# DA Variables
p_G = m.addMVar((grid['n_unit']), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'p_G')
r_up_G = m.addMVar((grid['n_unit']), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'upward reserve')
r_down_G = m.addMVar((grid['n_unit']), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'downward reserve')

rd_g = m.addMVar((grid['n_unit'], num_scen), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY, name = 'redispatch')

f_margin_up = m.addMVar((grid['n_lines']), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'upward reserve')
f_margin_down = m.addMVar((grid['n_lines']), vtype = gp.GRB.CONTINUOUS, lb = 0, name = 'downward reserve')    

### DA constraints 
m.addConstr( p_G + r_up_G <= Pmax.reshape(-1))
m.addConstr( p_G - r_down_G >= 0)
m.addConstr( p_G.sum() + w_expected.sum() == grid['Pd'].sum())


m.addConstr(grid['Line_Capacity'].reshape(-1) - f_margin_up == PTDF@(node_G@p_G + node_W@w_expected - node_L@grid['Pd'] ) )
m.addConstr(grid['Line_Capacity'].reshape(-1) - f_margin_down == -PTDF@(node_G@p_G + node_W@w_expected - node_L@grid['Pd'] ) )

### scenario constraints 

# system balancing constraints over all scenarios
m.addConstrs( w_scenarios[:,s].sum() + rd_g[:,s].sum() == 0 for s in range(num_scen))

m.addConstrs( rd_g[:,s] <= r_up_G for s in range(num_scen))
m.addConstrs( -rd_g[:,s] <= r_down_G for s in range(num_scen))
        
m.addConstrs( PTDF@(node_G@(rd_g[:,s]) + node_W@w_scenarios[:,s]) <= f_margin_up for s in range(num_scen))
m.addConstrs( -PTDF@(node_G@(rd_g[:,s]) + node_W@w_scenarios[:,s]) <= f_margin_down for s in range(num_scen))
                               
m.setObjective( Cost@p_G + C_r_up@r_up_G + C_r_down@r_down_G, gp.GRB.MINIMIZE)             
m.optimize()

#%% Box uncertainty with budget/ dual norm

# !!!! need to fix something on the flow constraints to make it equivalent to the above formulation
#box_cost_DA, box_p_DA, box_r_up, box_r_down, box_A = robust_dcopf_box(h_ub, h_lb, H_bound, h_bound, w_exp, grid, verbose = -1)
        
#%% L1 uncertainty set with dual norm and budget

#%% Data-driven L1 uncertainty including correlations

from sklearn.decomposition import PCA

### create transformed features w PCA
pca = PCA(n_components = train_errors.shape[1]).fit(train_errors)
pca_features = pca.fit_transform(train_errors)

fig, ax = plt.subplots()
plt.scatter(pca_features[:,0], pca_features[:,1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Decorrelated Errors')
plt.show()
#%%

# If x is a column vector, then the linear transformations are:
# x = A_pca@x_pca + b_pca and x_pca = A_pca_inv@(x-b)
# if whiten = False at PCA, then the inverse is just the transpose
# !!!!! if A_pca is scaled, you need to properly invert it (once)
A_pca = np.zeros((pca_features.shape[1], pca_features.shape[1]))
A_pca = pca.components_.T
A_pca_inv = np.linalg.inv(A_pca)
b_pca = pca.mean_

n_feat = grid['n_wind']

# upper and lower bound of PCs

quantiles = [0.01, 0.99]

pc_lb = np.quantile(pca_features, quantiles[0], axis = 0).reshape(-1)
pc_ub = np.quantile(pca_features, quantiles[1], axis = 0).reshape(-1)
        
    ### Uncertainty set: H@u <= h
    # If there is no transformation, use upper/lower bound (equivalent to box)
    # Else, translate the box into polyhedral in the original feature space

pca_map = [A_pca, b_pca, A_pca_inv]

# define polyhedron
# PC = A.T@(u-b)
H_id = np.row_stack((np.identity(n_feat), -np.identity(n_feat) ))        
H_poly = H_id@pca_map[2]
h_poly = np.array([pc_ub, -pc_lb]).reshape(-1) + H_id@pca_map[2]@pca_map[1]        


# solve robust OPF
dd_poly_solutions = robust_dcopf_polyhedral(H_poly, h_poly, w_exp, grid, verbose = -1)

#%%

#d = np.linspace(-2,16,300)
d = np.linspace(-w_cap,w_cap,300)
x,y = np.meshgrid(d,d)


fig, ax = plt.subplots(figsize = (6,4))

plt.scatter(train_errors[:,0], train_errors[:,1], alpha = 0.5)

patches = []

# data-driven box    
dd_box_vert = [r for r in itertools.product([h_lb[0], h_ub[0]], [h_lb[1], h_ub[1]])]
dd_box_vert.insert(2, dd_box_vert[-1])
dd_box_vert = dd_box_vert[:-1]


dd_box = Polygon(np.array(dd_box_vert), fill = False, edgecolor = 'black', linewidth = 3, label = 'DD Box$_{98\%}$')
patches.append(dd_box)
ax.add_patch(dd_box)

# full box support   
box_vert = [r for r in itertools.product([LB[0], UB[0]], [LB[1], UB[1]])]
box_vert.insert(2, box_vert[-1])
box_vert = box_vert[:-1]

box = Polygon(np.array(box_vert), fill = False, edgecolor = 'black', linestyle = '--', linewidth = 3, label = 'Full support')
patches.append(box)
ax.add_patch(box)

# polyhedron
x = np.linspace(-1, 1, 2000)
y = [(h_poly[i] - H_poly[i,0]*x)/H_poly[i,1] for i in range(len(H_poly))]

for i in range(len(H_poly)):
    if i ==0:
        plt.plot(x, y[i], color = 'tab:orange', label = 'DD Polyhedron')
    else:
        plt.plot(x, y[i], color = 'tab:orange', label='_nolegend_')
        
plt.ylim([-2, 2])
plt.xlim([-2, 2])
plt.legend(fontsize = 12, ncol = 3)
plt.show()

#%%
x,y = np.meshgrid(d,d)

plt.imshow( ( (H_poly[0,0]*x + H_poly[0,1]*y <= h_poly[0]) & (H_poly[1,0]*x + H_poly[1,1]*y <= h_poly[1]) &
             (H_poly[2,0]*x + H_poly[2,1]*y <= h_poly[2]) & (H_poly[3,0]*x + H_poly[3,1]*y <= h_poly[3]) ).astype(int) , 
                extent=(x.min(),x.max(),y.min(),y.max()), origin="lower", cmap="Greys", alpha = 0.3)
plt.ylim([-2, 2])
plt.show()

#%% Out-of-sample test

N_test = 2000
distr = 'normal'

test_errors = create_wind_errors(N_test, grid['w_cap'], grid['w_exp'], std = 0.25, seed = 1)

output = pd.DataFrame(data = [], columns = ['DA_cost', 'RT_cost'], index = ['DD_box', 'DD_poly'])

dd_box_rtcost = oos_cost_estimation(test_errors, dd_box_solutions, grid, c_viol = 2*1e2)

output.loc['DD_box']['DA_cost'] = dd_box_solutions['da_cost']
output.loc['DD_box']['RT_cost'] = dd_box_rtcost

output.loc['DD_poly']['DA_cost'] = dd_poly_solutions['da_cost']
output.loc['DD_poly']['RT_cost'] = oos_cost_estimation(test_errors, dd_poly_solutions, grid, c_viol = c_viol)

fig, ax = plt.subplots()
output.T.plot(kind = 'bar', ax = ax)
plt.show()

#%% Cost-based learning of a polytope

UB_init = grid['w_cap'] - grid['w_exp'] - 1
LB_init = - grid['w_exp'] + 1

from torch_layers import *

patience = 25
batch_size = 2000
num_epoch = 1000

tensor_trainY = torch.FloatTensor(train_errors)

train_data_loader = create_data_loader([tensor_trainY], batch_size = batch_size)
valid_data_loader = create_data_loader([tensor_trainY], batch_size = batch_size)

robust_opf_model = Robust_OPF(grid['n_wind'], 4, grid, UB_init, LB_init, c_viol = c_viol, add_fixed_box = False)
optimizer = torch.optim.Adam(robust_opf_model.parameters(), lr = 1e-2)
robust_opf_model.train_model(train_data_loader, valid_data_loader, optimizer, epochs = num_epoch, patience = patience, validation = False)

#%% resolve using learned polyhedral

# learned parameters 

H_cost = robust_opf_model.H.detach().numpy()
h_cost = robust_opf_model.h.detach().numpy()

dd_cost_poly_solutions = robust_dcopf_polyhedral(H_cost, h_cost, w_exp, grid, verbose = -1)

temp_cost = pd.DataFrame(data = [], columns = ['DA_cost', 'RT_cost'], index = ['CostDriven_poly'])
cost_driven_rtcost = oos_cost_estimation(test_errors, dd_cost_poly_solutions, grid, c_viol = c_viol)

temp_cost.loc['CostDriven_poly']['DA_cost'] = dd_cost_poly_solutions['da_cost']
temp_cost.loc['CostDriven_poly']['RT_cost'] = cost_driven_rtcost
output = pd.concat([output, temp_cost])

# Plot everything
fig, ax = plt.subplots(figsize = (6,4))

plt.scatter(train_errors[:,0], train_errors[:,1], alpha = 0.5)

patches = []

# data-driven box    
dd_box_vert = [r for r in itertools.product([h_lb[0], h_ub[0]], [h_lb[1], h_ub[1]])]
dd_box_vert.insert(2, dd_box_vert[-1])
dd_box_vert = dd_box_vert[:-1]


dd_box = Polygon(np.array(dd_box_vert), fill = False, edgecolor = 'black', linewidth = 3, label = 'DD Box$_{98\%}$')
patches.append(dd_box)
ax.add_patch(dd_box)

# full box support   
box_vert = [r for r in itertools.product([LB[0], UB[0]], [LB[1], UB[1]])]
box_vert.insert(2, box_vert[-1])
box_vert = box_vert[:-1]

box = Polygon(np.array(box_vert), fill = False, edgecolor = 'black', linestyle = '--', linewidth = 3, label = 'Full Support')
patches.append(box)
ax.add_patch(box)

# polyhedron
x = np.linspace(-1.5, 1.5, 2000)
y = [(h_poly[i] - H_poly[i,0]*x)/H_poly[i,1] for i in range(len(H_poly))]

for i in range(len(H_poly)):
    if i ==0:
        plt.plot(x, y[i], color = 'tab:orange', label = 'DD Poly')
    else:
        plt.plot(x, y[i], color = 'tab:orange', label='_nolegend_')
        

# cost-driven polyhedron
y = [(h_cost[i] - H_cost[i,0]*x)/H_cost[i,1] for i in range(robust_opf_model.num_constr)]
for i in range(robust_opf_model.num_constr):
    if i== 0:
        plt.plot(x, y[i], color = 'tab:green', label = 'Cost-driven Poly')
    else:
        plt.plot(x, y[i], color = 'tab:green')
    
plt.ylim([-2, 2])
plt.xlim([-2, 2])
plt.legend(fontsize = 10, ncol = 4)
plt.show()


#%%
H_tensor = nn.Parameter(torch.FloatTensor(H_bound).requires_grad_())
h_tensor = nn.Parameter(torch.FloatTensor(h_bound).requires_grad_())
# forward pass just for a check
results = robust_opf_model.forward(H_tensor, h_tensor)

#%%
