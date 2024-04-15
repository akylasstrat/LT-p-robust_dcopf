# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:54:33 2024

@author: astratig
"""

import numpy as np
import pandas as pd
from math import ceil
from itertools import product

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch

# from tqdm.notebook import tqdm, trange
from tqdm import tqdm, trange

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

from settings5 import *

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
# parameters of matpower case 5

# nominal demand (pu)
d = np.array([0.0, 3.0, 3.0, 4.0, 0.0])
# nominal capacity (pu)
pmax = np.array([0.4, 1.7, 5.2, 2.0, 6.0])
pmin = np.zeros(len(pmax))

smax = np.array([4.0, 1.9, 2.2, 1.0, 1.0, 2.4])
ptdf_str  = '-0.193917 0.475895   0.348989  0.0  -0.159538;'
ptdf_str += '-0.437588  -0.258343  -0.189451  0.0  -0.36001;'
ptdf_str += '-0.368495  -0.217552  -0.159538  0.0   0.519548;'
ptdf_str += '-0.193917  -0.524105   0.348989  0.0  -0.159538;'
ptdf_str += '-0.193917  -0.524105  -0.651011  0.0  -0.159538;'
ptdf_str += '0.368495   0.217552   0.159538  0.0   0.48045'
ptdf = np.matrix(ptdf_str)
#%%

cE = np.array([14.0, 15.0, 30.0, 40.0, 10.0]) # linear cost
cE_quad = np.sqrt(cE * 0.1) # quadratic cost
cR = np.array([80., 80., 15., 30., 80.]) # reserve cost
basemva = 100
genloc = np.array([1, 1, 3, 4, 5]) -1
windloc = np.array([3, 5]) - 1  # standard wind farm location
# windloc = np.array([3, 2]) - 1  # configuration B

w = np.array([1.0, 1.5])    # nominal wind forecasts/ expected values
w_cap = np.array([2.0, 3.0])

# number of gens, loads, lines, nodes
G = len(genloc)
D = len(windloc)
L = ptdf.shape[0]
B = ptdf.shape[1]

# incidence matrices (maps generators and wind to nodes)
gen2bus = np.zeros((B,G))
for g, bus in enumerate(genloc):
    gen2bus[bus, g] = 1
wind2bus = np.zeros((B,D))
for u, bus in enumerate(windloc):
    wind2bus[bus, u] = 1
#%%
TEST_PERC = 0.25
CORR = 0.5

# reset randomness
np.random.seed(seed=10)

# some other settings
d_range = [0.5, 1.1]
w_range = [0.5, 1.1]

# define bins
nbins = 10
bins = [np.linspace(w_range[0]*w[i], w_range[1]*w[i], nbins+1) for i in range(D)]
#%%
# create a large set of forecast errors from different wind scenarios
N_samples = 2000
train_errors_in_bins = [[[] for bi in range(nbins+3)] for i in range(D)]
all_errors = []
for i in trange(N_samples):
    d_scenario_np = np.random.uniform(*d_range, B) * d
    w_scenario_np = np.random.uniform(*w_range, D) * w
    cur_data = create_historical_data(w_scenario_np, N=1, corr=CORR)[0]
    for i in range(D):
        cur_bin = np.digitize(w_scenario_np[i], bins[i])
        train_errors_in_bins[i][cur_bin].append(cur_data[i])
    all_errors.append(cur_data)
all_errors = np.vstack(all_errors)

# error scatterplot
plt.scatter(all_errors[:,0], all_errors[:,1])
plt.show()

#%%
train, test = train_test_split(all_errors, test_size=int(all_errors.shape[0]*TEST_PERC), random_state=SEED)

train_data = torch.tensor(train, dtype=DTYPE)
test_data = torch.tensor(test, dtype=DTYPE)
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








