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


def box_robust_dcopf_problem_param(mu_init, sigma_init, demand, wind, allow_slack=False, quadratic_cost=False, gamma=0):

    # some settings
    A_base = 10
    slack_base = 10
    obj_base = basemva/10
    
    FR = 0.8 # reduction of line capacity
    
    # define mean and uncertainty of wind power injections as parameters
    mu = cp.Parameter(D, value=mu_init, name="mu")
    sigma = cp.Parameter(D, value=sigma_init, name="sigma")
    
     # define load as a parameter
    d = cp.Parameter(B, value=demand, nonneg=True, name="demand")
    w = cp.Parameter(D, value=wind, nonneg=True, name="wind")
        
    # main variables
    p  = cp.Variable(G, nonneg=True, name="p")
    rp = cp.Variable(G, nonneg=True, name="rp") # reserve up
    rm = cp.Variable(G, nonneg=True, name="rm") # reserve down
    A  = cp.Variable((G,D), nonneg=True, name="A") # Linear decision rules, maps error realizations to generator setpoints
    fRAMp = cp.Variable(L, nonneg=True, name="fRAMp")
    fRAMm = cp.Variable(L, nonneg=True, name="fRAMm")

    # aux. variables for robust constraints
    z = cp.Variable((2*G + 2*L,D), name="z")
    
    # aux. variables to ensure feasibility
    if allow_slack:
        slack = cp.Variable(2*G + 2*L, nonneg=True, name="slack")
    
    # basic det constraints
    flow = ptdf @ ((gen2bus @ p) + (wind2bus @ w) - d)
    consts = [
        cp.sum(p) + cp.sum(w) == cp.sum(d),
        p + rp <= pmax,
        p - rm >= pmin, 
        A.T @ np.ones(G) == np.ones(D)*A_base,
         flow + fRAMp == smax * FR,
        -flow + fRAMm == smax * FR
    ]

    # upper/lower bound with l-inf norm and primitive uncertainty
    #m.addConstr(aux_g_up[g] >= -P@coef[g])
    #m.addConstr(aux_g_up[g] >= P@coef[g])
    #m.addConstr(aux_g_down[g] >= -P@coef[g])
    #m.addConstr(aux_g_down[g] >= P@coef[g])

    #m.addConstr( coef[g]@u_nom + aux_g_up[g]@np.ones(n_feat) <= Pmax[g] )
    #m.addConstr( coef[g]@(-u_nom) + aux_g_down[g]@np.ones(n_feat) <= 0 )

    # box support constraints
    for g in range(G):
        if allow_slack:
            consts.append((mu.T @ (-A[g,:]/A_base)) + (sigma.T @ A[g,:]/A_base) <= rp[g] + slack[g]/slack_base)
        else:
            consts.append((mu.T @ (-A[g,:]/A_base)) + (sigma.T @ A[g,:]/A_base) <= rp[g])
        if allow_slack:
            consts.append((mu.T @ (A[g,:]/A_base)) + (sigma.T @  A[g,:]/A_base) <= rm[g] + slack[g+G]/slack_base)
        else:
            consts.append((mu.T @ (A[g,:]/A_base)) + (sigma.T @  A[g,:]/A_base) <= rm[g])
    for l in range(L):
        Bl = cp.reshape(ptdf[l,:] @ (wind2bus - (gen2bus @ A/A_base)), D)
        # Bl = (ptdf[l,:] @ (wind2bus - (gen2bus @ A))).T
        if allow_slack:
            consts.append(mu.T @ Bl + (sigma.T @ z[l,:]) <= fRAMp[l] + slack[2*G+l]/slack_base)
        else:
            consts.append(mu.T @ Bl + (sigma.T @ z[l,:]) <= fRAMp[l])
        consts.append(z[l,:] >= Bl)
        consts.append(z[l,:] >= -Bl)
        if allow_slack:
            consts.append(mu.T @ -Bl + (sigma.T @ z[L+l,:]) <= fRAMm[l] + slack[2*G+L+l]/slack_base)   
        else:
            consts.append(mu.T @ -Bl + (sigma.T @ z[L+l,:]) <= fRAMm[l])
        consts.append(z[L+l,:] >= -Bl)
        consts.append(z[L+l,:] >= Bl)

    # objective
    cost_E = (cE.T @ p)
    if quadratic_cost:
        cost_E_quad = cp.sum_squares(cp.multiply(cE_quad, p))
    else:
        cost_E_quad = 0                         
    cost_R = (cR.T @ (rp + rm))
    objective = cost_E + cost_E_quad + cost_R
    
    if allow_slack:
        thevars = [p, rp, rm, A, fRAMp, fRAMm, z, slack]
    else:
        thevars = [p, rp, rm, A, fRAMp, fRAMm, z]
    x = cp.hstack([v.flatten() for v in thevars])
    regularization = gamma * cp.sum_squares(x)
    objective += regularization
    
    if allow_slack:
        penalty_slack = cp.sum(slack) * obj_base * 1e3
        objective += penalty_slack
    
    theprob = cp.Problem(cp.Minimize(objective), consts)
    
    return theprob, thevars, [d, w, mu, sigma], consts

def create_historical_data(w_fcst, N=1000, SEED=42, metadata=False, corr=0.1, rel_sigma=[0.15, 0.15]):
    mu = np.zeros(D)
    rel_sigma = np.array(rel_sigma)
    correlation = np.matrix([[1.0, corr],[corr, 1.0]])
    sigma = w_fcst * rel_sigma
    Sigma = np.diag(sigma)*correlation*np.diag(sigma)
    # sample
    # np.random.seed(seed=SEED)
    hist_data = np.random.multivariate_normal(mu, Sigma, size=N)
    # truncate
    for j in range(D):
        hist_data[(hist_data[:,j] >= w_cap[j] - w_fcst[j]),j] = w_cap[j] - w_fcst[j]
        hist_data[(hist_data[:,j] <= -w_fcst[j]),j] = -w_fcst[j]
    if metadata:
        return hist_data, mu, Sigma
    else:
        return hist_data

def expected_cost(var_values, hist_data_tch, gamma=0):
    
    # some settings
    A_base = 10
    slack_base = 10
    obj_base = basemva/10
    
    p = var_values[0]
    rp = var_values[1]
    rm = var_values[2]
    A = var_values[3]
    fRAMp = var_values[4]
    fRAMm = var_values[5]
    if len(var_values) == 8:
        slack = var_values[-1]
    else:
        slack = torch.tensor(0)
    varlist = [p, rp, rm, A, fRAMp, fRAMm, slack]
    x = torch.hstack([v.flatten() for v in varlist])
        
    # expected first stage cost
    opf_cost = (torch.dot(p, cE_tch) + torch.dot(rp + rm, cR_tch)) 
    opf_cost += torch.sum(slack) * obj_base * 1e3

    # expected reserve violation cost
    reaction_gen = torch.matmul(A/A_base, hist_data_tch.T)
    expected_rp_viol_cost = torch.sum(nonneg(-reaction_gen.T - rp[None, :]).mean(axis=0) * cM_tch)
    expected_rm_viol_cost = torch.sum(nonneg(reaction_gen.T - rm[None, :]).mean(axis=0) * cM_tch)
    reaction_branch = torch.matmul(torch.matmul(ptdf_tch,(wind2bus_tch - torch.matmul(gen2bus_tch, A/A_base))), hist_data_tch.T)
    expected_framp_viol_cost = torch.sum(nonneg(reaction_branch.T - fRAMp[None, :]).mean(axis=0) * cM_tch)
    expected_framm_viol_cost = torch.sum(nonneg(-reaction_branch.T - fRAMm[None, :]).mean(axis=0) * cM_tch)
    
    regularization = gamma*torch.sum(torch.square(x))
    
    return opf_cost + expected_rp_viol_cost + expected_rm_viol_cost + expected_framp_viol_cost + expected_framm_viol_cost + regularization
#%%

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


def robust_dcopf(H, h, w_expected, grid, policy='affine', network = True, slack_var = False, rho = 1, loss = 'cost', verbose = -1, horizon = 1):
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

    m.addConstr(flow_da == PTDF@(node_G@p_G + node_W@w_expected - node_L@grid['Pd'] ) )
    m.addConstr(flow_da <= grid['Line_Capacity'].reshape(-1) )
    m.addConstr(flow_da >= -grid['Line_Capacity'].reshape(-1) )

    # robust constraints for all xi in H@xi <= h
    # W@xi <= r_down_G
    # -W@xi <= r_up_G
    
    # Reformulation of robust constraints
    # downward reserve bound
    lambda_up = m.addMVar((grid['n_unit'], num_constr) , vtype = gp.GRB.CONTINUOUS, lb = 0)
    lambda_down = m.addMVar((grid['n_unit'], num_constr) , vtype = gp.GRB.CONTINUOUS, lb = 0)
    
    m.addConstr( lambda_down@H == W )
    m.addConstr( h@lambda_down.T <= r_down_G)

    m.addConstr( lambda_up@H == -W )
    m.addConstr( h@lambda_down.T <= r_down_G)
            
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
        m.addConstr( h@lambda_f_up.T <= grid['Line_Capacity'] - flow_da )

        m.addConstr( lambda_f_down@H == -PTDF@(-node_G@W + node_W) )
        m.addConstr( h@lambda_f_down.T <= grid['Line_Capacity'] + flow_da)
                                   
        m.setObjective( Cost@p_G + C_r_up@r_up_G + C_r_down@r_down_G, gp.GRB.MINIMIZE)             
        m.optimize()
    try:       
        return m.objVal, p_G.X, r_up_G.X, r_down_G.X, W.X

    except:
        print('Infeasible solution')
        # scale cost back to aggregate    
        return 1e10, [], []
            
            
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
distr = 'normal'

all_errors = np.zeros((N_samples, grid['n_wind']))

if distr == 'normal':
    # multivariate normal
    std_Pd = .25*grid['w_exp']        
    ub_Pd = grid['w_cap']
    lb_Pd = np.zeros(len(grid['w_cap']))
    
    mean_Pd = ((ub_Pd-lb_Pd)/2+lb_Pd).reshape(-1)
    
    # generate correlation matrix *must be symmetric    
    np.random.seed(0)
    a = np.random.rand(grid['n_wind'], grid['n_wind'])
    
    R = np.tril(a) + np.tril(a, -1).T
    for i in range(grid['n_wind']): R[i,i] = 1
    # estimate covariance matrix
    S_cov = np.diag(std_Pd)@R@np.diag(std_Pd)        
    # sample demands, project them into support
    samples = np.random.multivariate_normal(np.zeros(mean_Pd.shape[0]), S_cov, size = N_samples).round(2)
    all_errors = samples
    
# Project back to feasible set
for u in range(grid['n_wind']):
    all_errors[:,u][grid['w_exp'][u] - all_errors[:,u] <= 0] = grid['w_exp'][u]
    all_errors[:,u][grid['w_exp'][u] + all_errors[:,u] >= grid['w_cap'][u]] = grid['w_cap'][u]
    
# error scatterplot
plt.scatter(all_errors[:,0], all_errors[:,1])
plt.show()

#%% 
import itertools

# Data-driven uncertainty box

H_bound = []
h_bound = []

quantiles = [0.00001, 0.9999999]

h_lb = np.quantile(all_errors, quantiles[0], axis = 0).reshape(-1)
h_ub = np.quantile(all_errors, quantiles[1], axis = 0).reshape(-1)

#h_ub = np.array([1, 4])

H_bound = np.row_stack((-np.eye(grid['n_wind']), np.eye(grid['n_wind'])))
h_bound = np.row_stack((-h_lb, h_ub)).reshape(-1)

fig, ax = plt.subplots(figsize = (6,4))

plt.scatter(all_errors[:,0], all_errors[:,1], alpha = 0.5)

patches = []
    
box_vert = [r for r in itertools.product([h_lb[0], h_ub[0]], 
                                         [h_lb[1], h_ub[1]])]
box_vert.insert(2, box_vert[-1])
box_vert = box_vert[:-1]

box = Polygon(np.array(box_vert), fill = False, 
                  edgecolor = 'black', linewidth = 3, label = '$\mathcal{U}$')
patches.append(box)
ax.add_patch(box)

plt.show()

# solve robust OPF
cost_DA, p_DA, r_up, r_down, A = robust_dcopf(H_bound, h_bound, w_exp, grid, loss = 'cost', verbose = -1)

#%% Box uncertainty with budget/ dual norm

#%% L1 uncertainty set with dual norm and budget

#%% Data-driven polyhedral (accounting for correlation)

#%%

# init based on stdv
mu_init = np.mean(train, axis=0)
sigma_init =np.std(train, axis=0)/2

# percentile-based set paramters/ data-driven intervals
perc= 10 # in percent

# !!!! equivalent to 10%, 90% quantile of errors
percupper = np.percentile(train, 100-perc, axis=0)
perclower = np.percentile(train, perc, axis=0)

# center and intervals of box (see my previous code)
# see also reformulation with primitive uncertainty factors from Bertsimas, den Hertog
mu_base_perc = (percupper + perclower) / 2
sigma_base_perc = mu_init + ((percupper - perclower) / 2)

#%%
prob, _, theparams, _ = box_robust_dcopf_problem_param(mu_init, sigma_init, d, w, allow_slack=False, quadratic_cost=True)
prob.solve(solver="ECOS")

# test feasibility for a few random scenarios
d_range = [0.5, 1.1]
w_range = [0.5, 1.1]
d_scenario = np.random.uniform(*d_range, B) * d
w_scenario = np.random.uniform(*w_range, D) * w

theparams[0].value = d_scenario
theparams[1].value = w_scenario
prob.solve(solver='ECOS', warm_start=True)
print(prob.status)
print(f'Objective value:  {prob.value:.4f}') 

#%% COST-BASED LOSS
# loss function

# use relu to discard negative values
nonneg = torch.nn.ReLU(inplace=False)

# some additonal settings
LR = 5e-6
LMOM = 0.3
GAMMA = 0.
cM = 2000
BATCHSIZE = 64

# reset randomness
np.random.seed(seed=SEED)

#### SINGLE

# prepare parameters 
cE_tch = torch.tensor(cE, dtype=DTYPE)
cR_tch = torch.tensor(cR, dtype=DTYPE)
cM_tch = torch.tensor(cM, dtype=DTYPE)
ptdf_tch = torch.tensor(ptdf, dtype=DTYPE)
gen2bus_tch = torch.tensor(gen2bus, dtype=DTYPE)
wind2bus_tch = torch.tensor(wind2bus, dtype=DTYPE)
zero_tch = torch.tensor(0, dtype=DTYPE)

# set up the layer
inner, vs, params, consts = box_robust_dcopf_problem_param(mu_init, sigma_init, d, w, gamma=GAMMA, allow_slack=True, quadratic_cost=False)
inner_cvxpylayer = CvxpyLayer(inner, parameters=inner.parameters(), variables=inner.variables())

# set up base model for comparison
base_prob, _, _, _ = box_robust_dcopf_problem_param(mu_base_perc, sigma_base_perc, d, w, gamma=GAMMA, allow_slack=True, quadratic_cost=False)

# set up the prescriptor
sigma_tch = torch.tensor(sigma_init, dtype=DTYPE, requires_grad=True)
mu_tch = torch.tensor(mu_init, dtype=DTYPE, requires_grad=True)

# set up SGD 
opt = torch.optim.SGD([mu_tch, sigma_tch], lr=LR, momentum=LMOM) 
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)

#train
loss_during_training = []
training_data_df = pd.DataFrame()
with trange(MAX_EPOCH) as ep_looper:
    for epoch in ep_looper:
        ep_looper.set_description(f'Epoch {epoch}')
        
        # reset loss
        loss = torch.tensor(0., dtype=DTYPE)
        
        for batch in range(BATCHSIZE):
            # create net demand scenario 
            d_scenario_np = np.random.uniform(*d_range, B) * d + 1e-7
            d_scenario = torch.tensor(d_scenario_np, dtype=DTYPE)
            w_scenario_np = np.random.uniform(*w_range, D) * w + 1e-7
            w_scenario = torch.tensor(w_scenario_np, dtype=DTYPE)
            scenario_vector = torch.cat((d_scenario, w_scenario))
        
            # compute current inner solution
            opf_params = [w_scenario, d_scenario, mu_tch, sigma_tch]
            try:
                var_values = inner_cvxpylayer(*opf_params,  solver_args={'solve_method': "ECOS", 'max_iters':20_000})
            except:
                print('infeasibility')
                continue
            # calculate loss
            temploss = expected_cost(var_values, train_data, gamma=GAMMA)
            loss = loss + temploss/BATCHSIZE
            
        # backpropagate
        loss.backward()

        # step the SGD
        opt.step()
        opt.zero_grad()
        scheduler.step(loss)
        
        # some analysis and reporting
        current_results = pd.Series({
            "epoch": epoch,
            "loss": loss.item(),
            "mu": mu_tch.detach().numpy(),
            "sigma": sigma_tch.detach().numpy(),
        })
        training_data_df = pd.concat([training_data_df, current_results.to_frame().T], ignore_index=True)
        loss_during_training.append(loss.item())
        ep_looper.set_postfix(loss=loss.item())
        
results_without_prescription = training_data_df.copy()
            
# some final reporting    
fig, ax = plt.subplots(1,1)
ax.plot(loss_during_training, label='train')
ax.set_ylabel('loss')
ax.set_xlabel('step')
ax.legend()


#%% P-ALL

# some additonal settings
LR = 1e-6
LMOM = 0.3
GAMMA = 0.1
cM = 2000

# reset randomness
np.random.seed(seed=SEED)

# prepare parameters 
cE_tch = torch.tensor(cE, dtype=DTYPE)
cR_tch = torch.tensor(cR, dtype=DTYPE)
cM_tch = torch.tensor(cM, dtype=DTYPE)
ptdf_tch = torch.tensor(ptdf, dtype=DTYPE)
gen2bus_tch = torch.tensor(gen2bus, dtype=DTYPE)
wind2bus_tch = torch.tensor(wind2bus, dtype=DTYPE)
zero_tch = torch.tensor(0, dtype=DTYPE)

# set up the layer
inner, vs, params, consts = box_robust_dcopf_problem_param(mu_init, sigma_init, d, w, gamma=GAMMA, allow_slack=True, quadratic_cost=False)
inner_cvxpylayer = CvxpyLayer(inner, parameters=inner.parameters(), variables=inner.variables())

# set up base model for comparison
base_prob, _, _, _ = box_robust_dcopf_problem_param(mu_base_perc, sigma_base_perc, d, w, gamma=GAMMA, allow_slack=True, quadratic_cost=False)

# set up the prescriptor
sigma_prescriptor = torch.nn.Linear(B+D, D)
sigma_prescriptor.weight.data = torch.zeros((D, B+D))
sigma_prescriptor.bias.data = torch.tensor(sigma_init, dtype=DTYPE)
mu_prescriptor = torch.nn.Linear(B+D, D)
mu_prescriptor.weight.data = torch.zeros((D, B+D))
mu_prescriptor.bias.data = torch.tensor(mu_init, dtype=DTYPE)

# set up SGD 
parameters = list(mu_prescriptor.parameters()) + list(sigma_prescriptor.parameters())
opt = torch.optim.SGD(parameters, lr=LR, momentum=LMOM) 
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)

#train
loss_during_training = []
training_data_df = pd.DataFrame()
with trange(MAX_EPOCH) as ep_looper:
    for epoch in ep_looper:
        ep_looper.set_description(f'Epoch {epoch}')
        
        # reset loss
        loss = torch.tensor(0., dtype=DTYPE)
        oosloss = torch.tensor(0., dtype=DTYPE)
        baseloss = torch.tensor(0., dtype=DTYPE)
    
        for batch in range(BATCHSIZE):
            # create net demand scenario 
            d_scenario_np = np.random.uniform(*d_range, B) * d
            d_scenario = torch.tensor(d_scenario_np, dtype=DTYPE)
            w_scenario_np = np.random.uniform(*w_range, D) * w
            w_scenario = torch.tensor(w_scenario_np, dtype=DTYPE)
            scenario_vector = torch.cat((d_scenario, w_scenario))
            
            # prescribe the the set size
            mu = mu_prescriptor(scenario_vector.float())
            sigma = sigma_prescriptor(scenario_vector.float())
        
            # compute current inner solution
            opf_params = [w_scenario, d_scenario, mu, sigma]
            var_values = inner_cvxpylayer(*opf_params,  solver_args={'solve_method': "ECOS"})
            
            # calculate loss
            temploss = expected_cost(var_values, train_data, gamma=GAMMA)
            loss = loss + temploss/BATCHSIZE
              
        # backpropagate
        loss.backward()

        # step the SGD
        opt.step()
        opt.zero_grad()
        scheduler.step(loss)
        
        # some analysis and reporting
        current_results = pd.Series({
            "epoch": epoch,
            "loss": loss.item(),
            "mu_pres_weight": mu_prescriptor.weight.data.detach(),
            "mu_pres_bias": mu_prescriptor.bias.data.detach(),
            "sigma_pres_weight": sigma_prescriptor.weight.data.detach(),
            "sigma_pres_bias": sigma_prescriptor.bias.data.detach(),
        })
        training_data_df = pd.concat([training_data_df, current_results.to_frame().T], ignore_index=True)
        loss_during_training.append(loss.item())
        ep_looper.set_postfix(loss=loss.item())

results_with_prescription = training_data_df.copy()
            
# some final reporting    
fig, ax = plt.subplots(1,1)
ax.plot(loss_during_training, label='train')
ax.set_ylabel('loss')
ax.set_xlabel('step')
ax.legend()

#%% OOS testing
# in oos gamma is always zero
GAMMA = 0.
cM = 2000

# get final training
def get_prescriptors(training_results):
    final_training_epoch = training_results.iloc[-1]
    sigma_prescriptor = torch.nn.Linear(B+D, D)
    sigma_prescriptor.weight.data = final_training_epoch.sigma_pres_weight
    sigma_prescriptor.bias.data = final_training_epoch.sigma_pres_bias
    mu_prescriptor = torch.nn.Linear(B+D, D)
    mu_prescriptor.weight.data = final_training_epoch.mu_pres_weight
    mu_prescriptor.bias.data = final_training_epoch.mu_pres_bias
    return mu_prescriptor, sigma_prescriptor

# create model
prob, thevars, theparams, _ = box_robust_dcopf_problem_param(mu_init, sigma_init, d, w, gamma=GAMMA, allow_slack=True, quadratic_cost=False)

pu_scale = 100
N_OOS = 500
oos_loss_base = []
oos_loss_full = []
oos_loss_single = []
oos_loss_presc = []
oos_loss_presc_cond = []
oos_loss_presc_imp_cond = []
for oos in trange(N_OOS):

    # create a scenario
    d_scenario_np = np.random.uniform(*d_range, B) * d
    w_scenario_np = np.random.uniform(*w_range, D) * w
    d_scenario = torch.tensor(d_scenario_np, dtype=DTYPE)
    w_scenario = torch.tensor(w_scenario_np, dtype=DTYPE)
    scenario_vector = torch.cat((d_scenario, w_scenario))
    # create a single error occurence
    cur_error = create_historical_data(w_scenario_np, N=12, corr=CORR)
    cur_data = torch.tensor(cur_error, dtype=DTYPE)

    ## base model
    theparams[0].value = d_scenario_np
    theparams[1].value = w_scenario_np
    theparams[2].value = mu_base_perc
    theparams[3].value = sigma_base_perc
    prob.solve(solver='ECOS', warm_start=True)
    var_values = [torch.tensor(v.value, dtype=DTYPE) for v in thevars]
    # compute loss for the prescribed model
    loss_base = expected_cost(var_values, cur_data, gamma=GAMMA)
    oos_loss_base.append(loss_base.item() * pu_scale)
    
    ## full robust
    mine = np.min(train, axis=0)
    maxe = np.min(train, axis=0)
    theparams[2].value = (maxe + mine) / 2
    theparams[3].value = mu_init + ((maxe - mine) / 2)
    prob.solve(solver='ECOS', warm_start=True)
    var_values = [torch.tensor(v.value, dtype=DTYPE) for v in thevars]
    # compute loss for the prescribed model
    loss_full = expected_cost(var_values, cur_data, gamma=GAMMA)
    oos_loss_full.append(loss_full.item() * pu_scale)
    
    ## one size fits all
    final_epoch = results_without_prescription.iloc[-1]
    theparams[2].value = final_epoch.mu
    theparams[3].value = final_epoch.sigma
    prob.solve(solver='ECOS', warm_start=True)
    var_values = [torch.tensor(v.value, dtype=DTYPE) for v in thevars]
    # compute loss for the prescribed model
    loss_single = expected_cost(var_values, cur_data, gamma=GAMMA)
    oos_loss_single.append(loss_single.item() * pu_scale)
    
    ## prescribed model
    # parametrize prescribed model
    mu_presc, sigma_presc = get_prescriptors(results_with_prescription)
    # mu_presc, sigma_presc = get_prescriptors(results_with_prescription_outer_sample)
    theparams[2].value = mu_presc(scenario_vector).detach().numpy()
    theparams[3].value = sigma_presc(scenario_vector).detach().numpy()
    prob.solve(solver='ECOS', warm_start=True)
    var_values = [torch.tensor(v.value, dtype=DTYPE) for v in thevars]
    # compute loss for the prescribed model
    loss_presc = expected_cost(var_values, cur_data, gamma=GAMMA)
    oos_loss_presc.append(loss_presc.item() * pu_scale)
    
    ## prescribed model with perfect knowledge of conditional distributon
    mu_presc_cond, sigma_presc_cond = get_prescriptors(results_with_prescription_and_cond_error)
    theparams[2].value = mu_presc_cond(scenario_vector).detach().numpy()
    theparams[3].value = sigma_presc_cond(scenario_vector).detach().numpy()
    prob.solve(solver='ECOS', warm_start=True)
    var_values = [torch.tensor(v.value, dtype=DTYPE) for v in thevars]
    # compute loss for the prescribed model
    loss_presc_cond = expected_cost(var_values, cur_data, gamma=GAMMA)
    oos_loss_presc_cond.append(loss_presc_cond.item() * pu_scale)
    
    ## prescribed model with IMperfect knowledge of conditional distributon
    mu_presc_imp_cond, sigma_presc_imp_cond = get_prescriptors(results_with_prescription_and_imp_cond_error)
    theparams[2].value = mu_presc_imp_cond(scenario_vector).detach().numpy()
    theparams[3].value = sigma_presc_imp_cond(scenario_vector).detach().numpy()
    prob.solve(solver='ECOS', warm_start=True)
    var_values = [torch.tensor(v.value, dtype=DTYPE) for v in thevars]
    # compute loss for the prescribed model
    loss_presc_imp_cond = expected_cost(var_values, cur_data, gamma=GAMMA)
    oos_loss_presc_imp_cond.append(loss_presc_imp_cond.item() * pu_scale)

    
for ci,cn in enumerate(['B', 'Full', 'OSFA', 'P', 'PC', 'PIPC']):
    curec = np.mean([oos_loss_base, oos_loss_full, oos_loss_single, oos_loss_presc, oos_loss_presc_cond, oos_loss_presc_imp_cond][ci])
    print(f'Expected cost case {cn}: {curec:.3f}')








