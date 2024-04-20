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
                 add_fixed_box = True):
        super(Robust_OPF, self).__init__()
        
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
        print(H_init.shape)
        print(h_init.shape)
        
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
        
        ###### DA variables and linear decision rules
        
        ### variables    
        # DA Variables
                        
        #flow_da = m.addMVar((grid['n_lines']), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)    


        p_G = cp.Variable((grid['n_unit']), nonneg = True)
        r_up_G = cp.Variable((grid['n_unit']), nonneg = True)
        r_down_G = cp.Variable((grid['n_unit']), nonneg = True)

        W = cp.Variable((grid['n_unit'], num_uncertainties))

        f_margin_up = cp.Variable((grid['n_lines']), nonneg = True)
        f_margin_down = cp.Variable((grid['n_lines']), nonneg = True)

        #### DA constraints
        
        ### DA constraints 
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

        
        DA_cost_expr = Cost@p_G + C_r_up@r_up_G + C_r_down@r_down_G
        objective_funct = cp.Minimize( DA_cost_expr ) 

        if add_fixed_box:
            # fixed robust constraints
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
                                        
        
        ###### Additional layer to estimate real-time cost
        
    def forward(self, H_value, h_value):
        """
        Forward pass of the newvendor layer.

        Args:
            list_inputs: A list of of input tensors/ PDFs.

        Returns:
            torch.Tensor: The convex combination of input tensors/ combination of PDFs.
        """
        # Ensure that the weights are in the range [0, 1] using softmax activation
        #H_param = self.H
        #h_param = self.h

        # Pass the combined output to the CVXPY layer
        cvxpy_output = self.robust_opf_layer(H_value, h_value, solver_args={'max_iters':20_000})

        return cvxpy_output
    
    def train_model(self, train_loader, val_loader, optimizer, epochs = 20, patience=5, projection = True, validation = False, 
                    relative_tolerance = 0):
                
        #L_t = []
        best_train_loss = 1e7
        best_val_loss = 1e7 
        early_stopping_counter = 0
        best_weights = copy.deepcopy(self.state_dict())

        Pmax_tensor = torch.FloatTensor(self.grid['Pmax'].reshape(-1))
        Pmin_tensor = torch.FloatTensor(self.grid['Pmin'].reshape(-1))
        
        # estimate initial loss
        print('Estimate Initial Loss...')
        '''
        initial_train_loss = 0
        with torch.no_grad():
            for batch_data in train_loader:
                y_batch = batch_data[-1]
                # clear gradients
                optimizer.zero_grad()
                output_hat = self.forward(batch_data[:-1])
                
                pdf_comb_hat = output_hat[0]
                cdf_comb_hat = pdf_comb_hat.cumsum(1)
                
                decisions_hat = output_hat[1][0]                
                p_hat = decisions_hat[0]
                
                # Project p_hat to feasible set
                p_hat_proj = torch.maximum(torch.minimum(p_hat, Pmax_tensor), Pmin_tensor)
                cost_DA_hat = decisions_hat[-1]

                # solve RT layer, find redispatch cost                
                rt_output = self.rt_layer(p_hat_proj, y_batch.reshape(-1,1), solver_args={'max_iters':50_000})                
                # CRPS of combination
                crps_i = sum([torch.square( cdf_comb_hat[i] - 1*(self.support >= y_batch[i]) ).sum() for i in range(len(y_batch))])
                loss = cost_DA_hat.mean() + rt_output[-1].mean() + self.gamma*crps_i
 
                initial_train_loss += loss.item()
                
        initial_train_loss = initial_train_loss/len(train_loader)
        best_train_loss = initial_train_loss
        print(f'Initial Estimate: {best_train_loss}')
        '''
        for epoch in range(epochs):
            # activate train functionality
            self.train()
            running_loss = 0.0                
            

            # sample batch data
            for batch_data in train_loader:
                # error realizations

                y_batch = batch_data[0]
                
                # clear gradients
                optimizer.zero_grad()

                # Forward pass: solve robust OPF
                #start = time.time()
                #start = time.time()
                decisions_hat = self.forward(self.H, self.h)
                
                p_hat = decisions_hat[0]
                r_up_hat = decisions_hat[1]
                r_down_hat = decisions_hat[2]
                W_hat = decisions_hat[3]
                f_margin_up_hat = decisions_hat[4]
                f_margin_down_hat = decisions_hat[5]
                
                # Evaluate cost-driven error
                # DA cost                
                cost_DA_hat = self.Cost@p_hat + self.C_r_up@r_up_hat + self.C_r_down@r_down_hat

                # RT dispatch cost (penalize violations)
                recourse_actions = -(W_hat@y_batch.T).T
                
                # exceeding reserves
                aggr_rup_violations = torch.maximum(recourse_actions - r_up_hat, torch.zeros(self.grid['n_unit'])).sum()
                aggr_rdown_violations = torch.maximum(-recourse_actions - r_down_hat, torch.zeros(self.grid['n_unit'])).sum()
                                                                                                                                      
                # exceeding line rating
                rt_injections = (self.PTDF@(self.node_G@recourse_actions.T + self.node_W@y_batch.T)).T
                
                aggr_f_margin_up_violations = torch.maximum( rt_injections - f_margin_up_hat, torch.zeros(self.grid['n_lines']) ).sum()
                aggr_f_margin_down_violations = torch.maximum( -rt_injections - f_margin_down_hat, torch.zeros(self.grid['n_lines'])).sum()

                rt_cost = self.c_viol*(aggr_rup_violations + aggr_rdown_violations + aggr_f_margin_up_violations + aggr_f_margin_down_violations)                
                
                loss = cost_DA_hat.mean() + rt_cost                
                #loss = cost_DA_hat.mean()
                    
                # backward pass
                # forward pass: combine forecasts and each stochastic ED problem
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            if epoch%10 == 0:
                
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
                
            average_train_loss = running_loss / len(train_loader)
                                        
            if validation == True:
                # evaluate performance on stand-out validation set
                val_loss = self.evaluate(val_loader)
                print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} - Val Loss: {val_loss:.4f}")
                
                if (val_loss < best_val_loss) and ( (best_val_loss-val_loss)/best_val_loss > relative_tolerance):
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
            else:
                # only evaluate on training data set
                print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} ")
    
                if (average_train_loss < best_train_loss) and ( (best_train_loss-average_train_loss)/best_train_loss > relative_tolerance):
                    best_train_loss = average_train_loss
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

    def evaluate(self, data_loader):
        # evaluate loss criterion/ used for estimating validation loss
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_data in data_loader:
                y_batch = batch_data[-1]

                # forward pass: combine forecasts and each stochastic ED problem
                output_hat = self.forward(batch_data[:-1])

                pdf_comb_hat = output_hat[0]
                cdf_comb_hat = pdf_comb_hat.cumsum(1)
                
                decisions_hat = output_hat[1][0]
                
                p_hat = decisions_hat[0]
                
                # solve RT layer, find redispatch cost
                rt_output = self.rt_layer(p_hat, y_batch.reshape(-1,1), solver_args={'max_iters':50000})                

                # CRPS of combination
                crps_i = sum([torch.square( cdf_comb_hat[i] - 1*(self.support >= y_batch[i]) ).sum() for i in range(len(y_batch))])

                # total loss
                loss = rt_output[-1].mean() + self.gamma*crps_i

                total_loss += loss.item()
                
        average_loss = total_loss / len(data_loader)
        return average_loss

    def __init__(self, input_size, hidden_sizes, output_size, support, activation=nn.ReLU(), apply_softmax = True):
        super(AdaptiveLinearPoolCRPSLayer, self).__init__()
        """
        Adaptive forecast combination, predicts weights for linear pool
        Args:
            input_size, hidden_sizes, output_size: standard arguments for declaring an MLP
            
            output_size: equal to the number of combination weights, i.e., number of experts we want to combine
            
        """
        # Initialize learnable weight parameters
        self.weights = nn.Parameter(torch.FloatTensor((1/output_size)*np.ones(output_size)).requires_grad_())
        self.num_features = input_size
        self.num_experts = output_size
        self.support = support
        self.apply_softmax = apply_softmax
            
        # create sequential MLP model to predict combination weights
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation)
                    
        self.model = nn.Sequential(*layers)
        if self.apply_softmax:
            self.model.add_module('softmax', nn.Softmax())
                          
    def forward(self, x, list_inputs):
        """
        Forward pass of linear pool.

        Args:
            x: input tensors/ features
            list_inputs: A list of of input tensors/ probability vectors.

        Returns:
            torch.Tensor: The convex combination of input tensors.
        """

        # Forwatd pass of the MLP to predict the combination weights (use softmax activation)
        weights = self.model(x)

        # Apply the weights element-wise to each input tensor !!!! CDFs
        
        #weighted_inputs = [weights[k,i] * input_tensor for k in range(weights.shape[0]) for i, input_tensor in enumerate(list_inputs)]
       # weighted_inputs = [weights[:,i] * input_tensor[:,k] for k in range(len(self.support)) 
       #                    for i, input_tensor in enumerate(list_inputs)]

        weighted_inputs = [torch.tile(weights[:,i:i+1], (1, input_tensor.shape[1])) * input_tensor for i, input_tensor in enumerate(list_inputs)]
        
        # Perform the convex combination across input vectors
        
        combined_PDF = sum(weighted_inputs)
        
        return combined_PDF
    
    def train_model(self, train_loader, val_loader, 
                    optimizer, epochs = 20, patience=5, projection = False):
        
        if (projection)and(self.apply_softmax != True):     
            lambda_proj = cp.Variable(self.num_inputs)
            lambda_hat = cp.Parameter(self.num_inputs)
            proj_problem = cp.Problem(cp.Minimize(0.5*cp.sum_squares(lambda_proj-lambda_hat)), [lambda_proj >= 0, lambda_proj.sum()==1])
        
        L_t = []
        best_train_loss = float('inf')
        best_val_loss = float('inf')
        early_stopping_counter = 0
        best_weights = copy.deepcopy(self.state_dict())

        for epoch in range(epochs):
            # activate train functionality
            self.train()
            running_loss = 0.0
            # sample batch data
            for batch_data in train_loader:
                
                y_batch = batch_data[-1]
                x_batch = batch_data[-2]
                p_list_batch = batch_data[0:-2]
                
                # clear gradients
                optimizer.zero_grad()
                
                # forward pass: predict weights and combine forecasts
                comb_PDF = self.forward(x_batch, p_list_batch)
                comb_CDF = comb_PDF.cumsum(1)
                
                # estimate CRPS (heavyside function)
                loss_i = [torch.square( comb_CDF[i] - 1*(self.support >= y_batch[i]) ).sum() for i in range(len(y_batch))]
                loss = sum(loss_i)/len(loss_i)
                
                # backward pass
                loss.backward()
                optimizer.step()                
                
                # Apply projection
                if (projection)and(self.apply_softmax != True):     
                    lambda_hat.value = to_np(self.weights)
                    proj_problem.solve(solver = 'GUROBI')
                    # update parameter values
                    with torch.no_grad():
                        self.weights.copy_(torch.FloatTensor(lambda_proj.value))
                
                running_loss += loss.item()
            

            L_t.append(to_np(self.weights).copy())
                
            average_train_loss = running_loss / len(train_loader)
            val_loss = self.evaluate(val_loader)
            
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

    def evaluate(self, data_loader):
        # evaluate loss criterion/ used for estimating validation loss
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_data in data_loader:
                y_batch = batch_data[-1]
                x_batch = batch_data[-2]
                p_list_batch = batch_data[0:-2]

                # forward pass: combine forecasts and solve each newsvendor problem
                comb_PDF_hat = self.forward(x_batch, p_list_batch)
                comb_CDF_hat = comb_PDF_hat.cumsum(1)

                # estimate CRPS (heavyside function)
                loss_i = [torch.square( comb_CDF_hat[i] - 1*(self.support >= y_batch[i]) ).sum() for i in range(len(y_batch))]
                loss = sum(loss_i)/len(loss_i)
                total_loss += loss.item()
                
        average_loss = total_loss / len(data_loader)
        return average_loss
                
    def predict_weights(self, x):
        'Forecast combination weights, inference only'
        with torch.no_grad():            
            return self.model(x).detach().numpy()
        
class Scenario_Robust_OPF(nn.Module):        
    
    def __init__(self, num_uncertainties, num_scen, initial_scenarios, support, grid, c_viol = 2*1e4, regularization = 0, include_network = True, 
                 add_fixed_box = True):
        super(Scenario_Robust_OPF, self).__init__()
        
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
        
        # project to feasible set        
        w_proj = torch.maximum(torch.minimum( self.w_scenarios_param, self.support_UB), self.support_LB)
        # update parameter values
        with torch.no_grad():
            self.w_scenarios_param.copy_(w_proj)
                    
        self.grid = grid
        self.regularization = regularization
        self.include_network = include_network
                
        #### Robust DCOPF Layer
        w_scen_param = cp.Parameter((num_scen, num_uncertainties))
        
        ###### DA variables and linear decision rules
        
        ### variables    
        # DA Variables
                        
        #flow_da = m.addMVar((grid['n_lines']), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)    
        p_G = cp.Variable((grid['n_unit']), nonneg = True)
        r_up_G = cp.Variable((grid['n_unit']), nonneg = True)
        r_down_G = cp.Variable((grid['n_unit']), nonneg = True)

        p_rt_up = cp.Variable((grid['n_unit'], num_scen), nonneg = True)
        p_rt_down = cp.Variable((grid['n_unit'], num_scen), nonneg = True)

        f_margin_up = cp.Variable((grid['n_lines']), nonneg = True)
        f_margin_down = cp.Variable((grid['n_lines']), nonneg = True)

        #### DA constraints
        
        ### DA constraints 
        DA_constraints = [p_G + r_up_G<= grid['Pmax'].reshape(-1), p_G - r_down_G >= 0, 
                          p_G.sum() + grid['w_exp'].sum() == grid['Pd'].sum(), 
                          grid['Line_Capacity'].reshape(-1) - f_margin_up == PTDF@(node_G@p_G + node_W@grid['w_exp'] - node_L@grid['Pd']),
                          grid['Line_Capacity'].reshape(-1) - f_margin_down == -PTDF@(node_G@p_G + node_W@grid['w_exp'] - node_L@grid['Pd'] ),] 
                          #PTDF@(node_G@p_G + node_W@grid['w_exp'] - node_L@grid['Pd'] ) <= grid['Line_Capacity'].reshape(-1), 
                          #PTDF@(node_G@p_G + node_W@grid['w_exp'] - node_L@grid['Pd'] ) >= -grid['Line_Capacity'].reshape(-1)]
        

        ##### Robust constraints per scenario
        
        # Reformulation of robust constraints
        # downward reserve bound/ each row are the duals to reformulate each constraints
        
        Robust_constr = [ p_rt_up[:,s] <= r_up_G for s in range(num_scen)] \
                        + [p_rt_down[:,s] <= r_down_G for s in range(num_scen)]\
                        + [p_rt_up[:,s].sum() - p_rt_down[:,s].sum() + w_scen_param[s].sum() == 0 for s in range(num_scen)]\
                        + [PTDF@(node_G@(p_rt_up[:,s] - p_rt_down[:,s]) + node_W@w_scen_param[s]) <= f_margin_up for s in range(num_scen)]\
                        + [-PTDF@(node_G@(p_rt_up[:,s] - p_rt_down[:,s]) + node_W@w_scen_param[s]) <= f_margin_down for s in range(num_scen)]\
        
        DA_cost_expr = Cost@p_G + C_r_up@r_up_G + C_r_down@r_down_G
        objective_funct = cp.Minimize( DA_cost_expr ) 
                
        robust_opf_problem = cp.Problem(objective_funct, DA_constraints + Robust_constr)
         
        self.robust_opf_layer = CvxpyLayer(robust_opf_problem, parameters=[w_scen_param],
                                           variables = [p_G, r_up_G, r_down_G, p_rt_up, p_rt_down, f_margin_up, f_margin_down])
                                        
        
        ###### RT market layer
        #p_g_param = cp.Parameter(grid['n_unit'], nonneg = True)
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
        RT_sched_constraints += [cost_RT == self.c_viol*(slack_up.sum() + slack_down.sum()) ]
            
        objective_funct = cp.Minimize( cost_RT ) 
        rt_problem = cp.Problem(objective_funct, RT_sched_constraints)
         
        self.rt_layer = CvxpyLayer(rt_problem, parameters=[r_up_param, r_down_param, f_margin_up_param, f_margin_down_param, w_realized_error],
                                           variables = [recourse_up, recourse_down, slack_up, slack_down, cost_RT] )


    def forward(self, w_scen):
        """
        Forward pass of the newvendor layer.

        Args:
            list_inputs: A list of of input tensors/ PDFs.

        Returns:
            torch.Tensor: The convex combination of input tensors/ combination of PDFs.
        """
        # Ensure that the weights are in the range [0, 1] using softmax activation
        #H_param = self.H
        #h_param = self.h

        # Pass the combined output to the CVXPY layer
        cvxpy_output = self.robust_opf_layer(w_scen, solver_args={'max_iters':500_000})

        return cvxpy_output
    
    def train_model(self, train_loader, val_loader, optimizer, epochs = 20, patience=5, projection = True, validation = False, 
                    relative_tolerance = 0):
                
        #L_t = []
        best_train_loss = 1e7
        best_val_loss = 1e7 
        early_stopping_counter = 0
        best_weights = copy.deepcopy(self.state_dict())

        Pmax_tensor = torch.FloatTensor(self.grid['Pmax'].reshape(-1))
        Pmin_tensor = torch.FloatTensor(self.grid['Pmin'].reshape(-1))
        
        # estimate initial loss
        print('Estimate Initial Loss...')
        '''
        initial_train_loss = 0
        with torch.no_grad():
            for batch_data in train_loader:
                y_batch = batch_data[-1]
                # clear gradients
                optimizer.zero_grad()
                output_hat = self.forward(batch_data[:-1])
                
                pdf_comb_hat = output_hat[0]
                cdf_comb_hat = pdf_comb_hat.cumsum(1)
                
                decisions_hat = output_hat[1][0]                
                p_hat = decisions_hat[0]
                
                # Project p_hat to feasible set
                p_hat_proj = torch.maximum(torch.minimum(p_hat, Pmax_tensor), Pmin_tensor)
                cost_DA_hat = decisions_hat[-1]

                # solve RT layer, find redispatch cost                
                rt_output = self.rt_layer(p_hat_proj, y_batch.reshape(-1,1), solver_args={'max_iters':50_000})                
                # CRPS of combination
                crps_i = sum([torch.square( cdf_comb_hat[i] - 1*(self.support >= y_batch[i]) ).sum() for i in range(len(y_batch))])
                loss = cost_DA_hat.mean() + rt_output[-1].mean() + self.gamma*crps_i
 
                initial_train_loss += loss.item()
                
        initial_train_loss = initial_train_loss/len(train_loader)
        best_train_loss = initial_train_loss
        print(f'Initial Estimate: {best_train_loss}')
        '''                
        for epoch in range(epochs):
            # activate train functionality
            self.train()
            running_loss = 0.0                
            

            # sample batch data
            for batch_iter, batch_data in enumerate(train_loader):
                # error realizations
                y_batch = batch_data[0]

                # visualize current box
                if ((epoch)%5 == 0) and (batch_iter == 0):
                                    
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
                    plt.title(f'C_viol = {self.c_viol} \$/MWh, Iteration: {epoch}', fontsize = 12)
                    plt.xlabel('Error 1')
                    plt.xlabel('Error 2')
                    plt.legend(fontsize = 12, loc = 'upper right')
                    plt.show()

                
                # clear gradients
                optimizer.zero_grad()

                # Forward pass: solve robust OPF
                #start = time.time()
                #start = time.time()
                decisions_hat = self.forward(self.w_scenarios_param)
                
                p_hat = decisions_hat[0]
                r_up_hat = decisions_hat[1]
                r_down_hat = decisions_hat[2]
                f_margin_up_hat = decisions_hat[5]
                f_margin_down_hat = decisions_hat[6]
                
                # Project to feasible set (might incur numerical errors)
                p_hat_proj = torch.maximum(torch.minimum(p_hat, self.Pmax), self.Pmin)
                r_up_hat_proj = torch.maximum(torch.minimum(r_up_hat, self.Pmax - p_hat_proj), torch.zeros(self.grid['n_unit']))
                r_down_hat_proj = torch.maximum(torch.minimum(r_down_hat, p_hat_proj), torch.zeros(self.grid['n_unit']))
                
                f_margin_up_hat_proj = torch.maximum(f_margin_up_hat, torch.zeros(self.grid['n_lines']) ) 
                f_margin_down_hat_proj = torch.maximum(f_margin_down_hat, torch.zeros(self.grid['n_lines']) ) 
                
                # Evaluate cost-driven error
                # DA cost                
                cost_DA_hat = self.Cost@p_hat + self.C_r_up@r_up_hat + self.C_r_down@r_down_hat

                # RT dispatch cost (penalize violations)
                rt_output = self.rt_layer(r_up_hat_proj, r_down_hat_proj, f_margin_up_hat_proj, f_margin_down_hat_proj, 
                                          y_batch, solver_args={'max_iters':500_000})                
                cost_RT_hat = rt_output[-1]
                
                loss = cost_DA_hat.mean() + cost_RT_hat.mean()
                    
                # backward pass
                # forward pass: combine forecasts and each stochastic ED problem
                loss.backward()
                optimizer.step()
                
                # Project back to box of support 
                w_proj = torch.maximum(torch.minimum( self.w_scenarios_param, self.support_UB), self.support_LB)
                # update parameter values
                with torch.no_grad():
                    self.w_scenarios_param.copy_(w_proj)
                    
                running_loss += loss.item()
                
                
            average_train_loss = running_loss / len(train_loader)
                                        
            if validation == True:
                # evaluate performance on stand-out validation set
                val_loss = self.evaluate(val_loader)
                print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} - Val Loss: {val_loss:.4f}")
                
                if (val_loss < best_val_loss) and ( (best_val_loss-val_loss)/best_val_loss > relative_tolerance):
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
            else:
                # only evaluate on training data set
                print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} ")
    
                if (average_train_loss < best_train_loss) and ( (best_train_loss-average_train_loss)/best_train_loss > relative_tolerance):
                    best_train_loss = average_train_loss
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

    def evaluate(self, data_loader):
        # evaluate loss criterion/ used for estimating validation loss
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_data in data_loader:
                y_batch = batch_data[-1]

                # forward pass: combine forecasts and each stochastic ED problem
                output_hat = self.forward(batch_data[:-1])

                pdf_comb_hat = output_hat[0]
                cdf_comb_hat = pdf_comb_hat.cumsum(1)
                
                decisions_hat = output_hat[1][0]
                
                p_hat = decisions_hat[0]
                
                # solve RT layer, find redispatch cost
                rt_output = self.rt_layer(p_hat, y_batch.reshape(-1,1), solver_args={'max_iters':50000})                

                # CRPS of combination
                crps_i = sum([torch.square( cdf_comb_hat[i] - 1*(self.support >= y_batch[i]) ).sum() for i in range(len(y_batch))])

                # total loss
                loss = rt_output[-1].mean() + self.gamma*crps_i

                total_loss += loss.item()
                
        average_loss = total_loss / len(data_loader)
        return average_loss
    

class Contextual_Scenario_Robust_OPF(nn.Module):        
    
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
        self.num_uncertainties = num_uncertainties # number of uncertain parameters (only winds here)
        self.num_constr = num_scen # number of scenarios to learn
        
        # Parameters to be estimated (initialize at the extremes of the box)
        self.w_scenarios_param = nn.Parameter(torch.FloatTensor(initial_scenarios).requires_grad_())
        
        
        # projection step (set as separate function)        
        #w_proj = torch.maximum(torch.minimum( self.w_scenarios_param, self.support_UB), self.support_LB)
        # update parameter values
        #with torch.no_grad():
        #    self.w_scenarios_param.copy_(w_proj)
        
        ### Initialize MLP layer
        input_size = mlp_param_dict['input_size']
        hidden_sizes = mlp_param_dict['hidden_sizes']
        output_size = num_uncertainties*num_scen
        
        # create sequential MLP model to predict combination weights
        #!!! *** To avoid outputting a matrix, create a model for each scenario ***
        
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation)

        self.model = nn.Sequential(*layers)
        
        #### Robust DCOPF Layer
        # Parameters: nominal wind production, nominal demand (expected values), scenarios for errors
        w_nominal = cp.Parameter((grid['n_wind']), nonneg = True)
        d_nominal = cp.Parameter((grid['n_loads']), nonneg = True)
        
        w_error_scen = cp.Parameter((num_scen, num_uncertainties))
        
        ###### DA variables and linear decision rules
        
        ### variables    
        # DA Variables
                        
        #flow_da = m.addMVar((grid['n_lines']), vtype = gp.GRB.CONTINUOUS, lb = -gp.GRB.INFINITY)    
        p_G = cp.Variable((grid['n_unit']), nonneg = True)
        r_up_G = cp.Variable((grid['n_unit']), nonneg = True)
        r_down_G = cp.Variable((grid['n_unit']), nonneg = True)
        slack = cp.Variable((grid['n_nodes']), nonneg = True)
        
        p_rt_up = cp.Variable((grid['n_unit'], num_scen), nonneg = True)
        p_rt_down = cp.Variable((grid['n_unit'], num_scen), nonneg = True)

        f_margin_up = cp.Variable((grid['n_lines']), nonneg = True)
        f_margin_down = cp.Variable((grid['n_lines']), nonneg = True)

        #### DA constraints
        
        ### DA constraints 
        DA_constraints = [p_G + r_up_G<= self.Pmax.reshape(-1), p_G - r_down_G >= 0, 
                          p_G.sum() + w_nominal.sum() == d_nominal.sum(), 
                          grid['Line_Capacity'].reshape(-1) - f_margin_up == self.PTDF@(self.node_G@p_G + self.node_W@w_nominal - self.node_L@d_nominal),
                          grid['Line_Capacity'].reshape(-1) - f_margin_down == -self.PTDF@(self.node_G@p_G + self.node_W@w_nominal - self.node_L@d_nominal ),]         

        ##### Robust constraints per scenario
        
        # Reformulation of robust constraints
        # downward reserve bound/ each row are the duals to reformulate each constraints
        Robust_constr = [ p_rt_up[:,s] <= r_up_G for s in range(num_scen)] \
                        + [p_rt_down[:,s] <= r_down_G for s in range(num_scen)]\
                        + [p_rt_up[:,s].sum() - p_rt_down[:,s].sum() + slack.sum() + w_error_scen[s].sum() == 0 for s in range(num_scen)]\
                        + [self.PTDF@(self.node_G@(p_rt_up[:,s] - p_rt_down[:,s]) + slack + self.node_W@w_error_scen[s]) <= f_margin_up for s in range(num_scen)]\
                        + [-self.PTDF@(self.node_G@(p_rt_up[:,s] - p_rt_down[:,s]) + slack + self.node_W@w_error_scen[s]) <= f_margin_down for s in range(num_scen)]\
        

        DA_cost_expr = self.Cost@p_G + self.C_r_up@r_up_G + self.C_r_down@r_down_G + self.c_viol*cp.abs(slack).sum()
        objective_funct = cp.Minimize( DA_cost_expr ) 
                
        robust_opf_problem = cp.Problem(objective_funct, DA_constraints + Robust_constr)
         
        self.robust_opf_layer = CvxpyLayer(robust_opf_problem, parameters=[d_nominal, w_nominal, w_error_scen],
                                           variables = [p_G, r_up_G, r_down_G, p_rt_up, p_rt_down, f_margin_up, f_margin_down])
                                        
        
        ###### RT market layer
        #p_g_param = cp.Parameter(grid['n_unit'], nonneg = True)
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
        RT_sched_constraints += [self.PTDF@(self.node_G@(recourse_up - recourse_down) + (slack_up -  slack_down) + self.node_W@(w_realized_error)) <= f_margin_up_param]            
        RT_sched_constraints += [-self.PTDF@(self.node_G@(recourse_up - recourse_down) + (slack_up -  slack_down) + self.node_W@(w_realized_error)) <= f_margin_down_param]            
                
        # balancing
        RT_sched_constraints += [ recourse_up.sum() - recourse_down.sum() + w_realized_error.sum() + (slack_up -  slack_down).sum() == 0]
        RT_sched_constraints += [cost_RT == self.c_viol*(slack_up.sum() + slack_down.sum()) ]
            
        objective_funct = cp.Minimize( cost_RT ) 
        rt_problem = cp.Problem(objective_funct, RT_sched_constraints)
         
        self.rt_layer = CvxpyLayer(rt_problem, parameters=[r_up_param, r_down_param, f_margin_up_param, f_margin_down_param, w_realized_error],
                                           variables = [recourse_up, recourse_down, slack_up, slack_down, cost_RT] )


    def scenario_projection(self, w_nominal, w_error_scen):
        """
        Projects scenarios to box constraints defined by support
            Support depends on problem parameterization (e.g., expected wind production)
        """            
        temp_UB = self.w_cap - w_nominal
        temp_LB = - w_nominal
        
        w_proj = w_error_scen
        
        #print(torch.maximum(torch.minimum( w_error_scen[:,0,:], temp_UB), temp_LB).shape)
        for j in range(self.num_constr):
            with torch.no_grad():
                w_proj[:,j,:] = torch.maximum(torch.minimum( w_error_scen[:,j,:], temp_UB), temp_LB)
        
        # update parameter values
        #with torch.no_grad():
        #    self.w_scenarios_param.copy_(w_proj)
        return  w_proj  
    
    def forward(self, demand_batch, wind_batch):
        """
        Forward pass of the newvendor layer.

        Args:
            list_inputs: A list of of input tensors/ PDFs.

        Returns:
            torch.Tensor: The convex combination of input tensors/ combination of PDFs.
        """
        
        # predict the scenarios based on realization of features
        #demand_batch = list_batch[0]
        #wind_batch = list_batch[1]
        #wind_error_batch = list_batch[2]
        batch_size = len(demand_batch)
        feat_hat = torch.cat([demand_batch, wind_batch], axis = 1)
        
        #w_scen_hat = torch.FloatTensor(np.array([self.model_scen[i](feat_hat) for i in range(self.num_constr)]))
        
        # !!!! Check that it transpoes everything correct
        w_scen_hat = self.model(feat_hat).reshape(batch_size, self.num_constr, self.num_uncertainties)
        #standard_output = self.model(feat_hat)
        
        # project back to feasible set
        w_scen_hat_proj = self.scenario_projection(wind_batch, w_scen_hat)
        
        # Pass the combined output to the CVXPY layer
        cvxpy_output = self.robust_opf_layer(demand_batch, wind_batch, w_scen_hat_proj, solver_args={'max_iters':500_000})

        return cvxpy_output
    
    def predict(self, demand_batch, wind_batch):
        """
        Predicts the robust scenarios
        """
        
        # predict the scenarios based on realization of features
        batch_size = len(demand_batch)
        feat_hat = torch.cat([demand_batch, wind_batch], axis = 1)
                
        # !!!! Check that it transpoes everything correct
        with torch.no_grad():
            w_scen_hat = self.model(feat_hat).reshape(batch_size, self.num_constr, self.num_uncertainties)
            #standard_output = self.model(feat_hat)
            
            # project back to feasible set
            w_scen_hat_proj = self.scenario_projection(wind_batch, w_scen_hat)

        return w_scen_hat_proj
    
    def train_model(self, train_loader, val_loader, optimizer, epochs = 20, patience=5, projection = True, validation = False, 
                    relative_tolerance = 0):
                
        #L_t = []
        best_train_loss = 1e7
        best_val_loss = 1e7 
        early_stopping_counter = 0
        best_weights = copy.deepcopy(self.state_dict())

        Pmax_tensor = torch.FloatTensor(self.grid['Pmax'].reshape(-1))
        Pmin_tensor = torch.FloatTensor(self.grid['Pmin'].reshape(-1))
        
        # estimate initial loss
        print('Estimate Initial Loss...')
        '''
        initial_train_loss = 0
        with torch.no_grad():
            for batch_data in train_loader:
                y_batch = batch_data[-1]
                # clear gradients
                optimizer.zero_grad()
                output_hat = self.forward(batch_data[:-1])
                
                pdf_comb_hat = output_hat[0]
                cdf_comb_hat = pdf_comb_hat.cumsum(1)
                
                decisions_hat = output_hat[1][0]                
                p_hat = decisions_hat[0]
                
                # Project p_hat to feasible set
                p_hat_proj = torch.maximum(torch.minimum(p_hat, Pmax_tensor), Pmin_tensor)
                cost_DA_hat = decisions_hat[-1]

                # solve RT layer, find redispatch cost                
                rt_output = self.rt_layer(p_hat_proj, y_batch.reshape(-1,1), solver_args={'max_iters':50_000})                
                # CRPS of combination
                crps_i = sum([torch.square( cdf_comb_hat[i] - 1*(self.support >= y_batch[i]) ).sum() for i in range(len(y_batch))])
                loss = cost_DA_hat.mean() + rt_output[-1].mean() + self.gamma*crps_i
 
                initial_train_loss += loss.item()
                
        initial_train_loss = initial_train_loss/len(train_loader)
        best_train_loss = initial_train_loss
        print(f'Initial Estimate: {best_train_loss}')
        '''                
        for epoch in range(epochs):
            # activate train functionality
            self.train()
            running_loss = 0.0                
            

            # sample batch data
            for batch_iter, batch_data in enumerate(train_loader):

                d_hat = batch_data[0]
                w_hat = batch_data[1]
                w_error_hat = batch_data[2]
                
                # clear gradients
                optimizer.zero_grad()

                # Forward pass: solve robust OPF
                #start = time.time()
                #start = time.time()
                
                
                # Forward pass: map contextual information to scenarios, solve robust problem
                decisions_hat = self.forward(d_hat, w_hat)
                
                # visualize some scenarios
                '''
                if ((epoch)%5 == 0) and (batch_iter == 0):
                                    
                    fig, ax = plt.subplots(figsize = (6,4))                
                    w_scen = to_np(self.w_scenarios_param)
                    #plt.scatter(y_batch[:,0], y_batch[:,1], label = 'Data')                    
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
                    plt.title(f'C_viol = {self.c_viol} \$/MWh, Iteration: {epoch}', fontsize = 12)
                    plt.xlabel('Error 1')
                    plt.xlabel('Error 2')
                    plt.legend(fontsize = 12, loc = 'upper right')
                    plt.show()
                '''
                
                p_hat = decisions_hat[0]
                r_up_hat = decisions_hat[1]
                r_down_hat = decisions_hat[2]
                f_margin_up_hat = decisions_hat[5]
                f_margin_down_hat = decisions_hat[6]
                
                # Project to feasible set (might incur numerical errors)
                p_hat_proj = torch.maximum(torch.minimum(p_hat, self.Pmax), self.Pmin)
                r_up_hat_proj = torch.maximum(torch.minimum(r_up_hat, self.Pmax - p_hat_proj), torch.zeros(self.grid['n_unit']))
                r_down_hat_proj = torch.maximum(torch.minimum(r_down_hat, p_hat_proj), torch.zeros(self.grid['n_unit']))
                
                f_margin_up_hat_proj = torch.maximum(f_margin_up_hat, torch.zeros(self.grid['n_lines']) ) 
                f_margin_down_hat_proj = torch.maximum(f_margin_down_hat, torch.zeros(self.grid['n_lines']) ) 
                
                # Evaluate cost-driven error
                # DA cost                
                
                cost_DA_hat = p_hat@self.Cost + r_up_hat@self.C_r_up + r_down_hat@self.C_r_down

                # RT dispatch cost (penalize violations)
                rt_output = self.rt_layer(r_up_hat_proj, r_down_hat_proj, f_margin_up_hat_proj, f_margin_down_hat_proj, 
                                          w_error_hat, solver_args={'max_iters':500_000})                
                cost_RT_hat = rt_output[-1]
                
                loss = cost_DA_hat.mean() + cost_RT_hat.mean()
                    
                # backward pass
                # forward pass: combine forecasts and each stochastic ED problem
                loss.backward()
                optimizer.step()
                
                # Project back to box of support 
                #w_proj = torch.maximum(torch.minimum( self.w_scenarios_param, self.support_UB), self.support_LB)
                # update parameter values
                #with torch.no_grad():
                #    self.w_scenarios_param.copy_(w_proj)
                    
                running_loss += loss.item()
                
                
            average_train_loss = running_loss / len(train_loader)
                                        
            if validation == True:
                # evaluate performance on stand-out validation set
                val_loss = self.evaluate(val_loader)
                print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} - Val Loss: {val_loss:.4f}")
                
                if (val_loss < best_val_loss) and ( (best_val_loss-val_loss)/best_val_loss > relative_tolerance):
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
            else:
                # only evaluate on training data set
                print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} ")
    
                if (average_train_loss < best_train_loss) and ( (best_train_loss-average_train_loss)/best_train_loss > relative_tolerance):
                    best_train_loss = average_train_loss
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

    def evaluate(self, data_loader):
        # evaluate loss criterion/ used for estimating validation loss
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_data in data_loader:
                y_batch = batch_data[-1]

                # forward pass: combine forecasts and each stochastic ED problem
                output_hat = self.forward(batch_data[:-1])

                pdf_comb_hat = output_hat[0]
                cdf_comb_hat = pdf_comb_hat.cumsum(1)
                
                decisions_hat = output_hat[1][0]
                
                p_hat = decisions_hat[0]
                
                # solve RT layer, find redispatch cost
                rt_output = self.rt_layer(p_hat, y_batch.reshape(-1,1), solver_args={'max_iters':50000})                

                # CRPS of combination
                crps_i = sum([torch.square( cdf_comb_hat[i] - 1*(self.support >= y_batch[i]) ).sum() for i in range(len(y_batch))])

                # total loss
                loss = rt_output[-1].mean() + self.gamma*crps_i

                total_loss += loss.item()
                
        average_loss = total_loss / len(data_loader)
        return average_loss



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


    def forward(self, sol_dict, w_error_scen):
        
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
        
        rt_output = self.rt_layer(r_up_hat_proj, r_down_hat_proj, f_margin_up_hat_proj, f_margin_down_hat_proj, 
                                  w_error_scen, solver_args={'max_iters':100_000})                
        cost_RT_hat = rt_output[-1]
        
        rt_loss = cost_RT_hat.mean()
            
        return to_np(cost_RT_hat)
