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
    
    def __init__(self, num_uncertainties, num_constr, grid, UB, LB, c_viol = 2*1e4, regularization = 0, include_network = True):
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
                
        H_init = np.row_stack((-np.eye(num_uncertainties), np.eye(num_uncertainties)))
        h_init = np.row_stack((-LB, UB)).reshape(-1)
        
        for i in range(num_constr - len(h_init)):
            h_init = np.row_stack((h_init.reshape(-1,1), np.zeros((1,1)))).reshape(-1)
            H_init = np.row_stack((H_init, np.zeros((1, num_uncertainties)))) 

        #H_init = np.zeros((num_constr, num_uncertainties))
        #h_init = np.zeros(num_constr)
        print(H_init.shape)
        print(h_init.shape)
        
        self.H = nn.Parameter(torch.FloatTensor(H_init).requires_grad_())
        self.h = nn.Parameter(torch.FloatTensor(h_init).requires_grad_())

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
                
        robust_opf_problem = cp.Problem(objective_funct, DA_constraints + Robust_constr)
         
        self.robust_opf_layer = CvxpyLayer(robust_opf_problem, parameters=[H_param, h_param],
                                           variables = [p_G, r_up_G, r_down_G, W, f_margin_up, f_margin_down, 
                                                        lambda_up, lambda_down, lambda_f_up, lambda_f_down] )
                                        
        
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
                
            if epoch%5 == 0:
                
                fig, ax = plt.subplots(figsize = (6,4))
                x = np.linspace(-3, 3, 1000)
                
                H_np = to_np(self.H)
                h_np = to_np(self.h)
                
                y = [(h_np[i] - H_np[i,0]*x)/H_np[i,1] for i in range(len(H_np))]
                

                plt.scatter(y_batch[:,0], y_batch[:,1])
                for i in range(len(H_np)):
                    plt.plot(x, y[i], color = 'black')
                plt.ylim([-2, 2])
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
    
class LinearPoolCRPSLayer(nn.Module):        
    def __init__(self, num_inputs, support, apply_softmax = False):
        super(LinearPoolCRPSLayer, self).__init__()

        # Initialize learnable weight parameters
        #self.weights = nn.Parameter(torch.rand(num_inputs), requires_grad=True)
        self.weights = nn.Parameter(torch.FloatTensor((1/num_inputs)*np.ones(num_inputs)).requires_grad_())
        self.num_inputs = num_inputs
        self.support = support
        self.apply_softmax = apply_softmax
        
    def forward(self, list_inputs):
        """
        Forward pass of the linear pool minimizing CRPS.

        Args:
            list_inputs: A list of of input tensors (discrete PDFs).

        Returns:
            torch.Tensor: The convex combination of input tensors.
        """
        # Ensure that the weights are in the range [0, 1] using sigmoid activation
        #weights = torch.nn.functional.softmax(self.weights)

        # Ensure that the weights are in the range [0, 1] using sigmoid activation
        if self.apply_softmax:
            weights = torch.nn.functional.softmax(self.weights, dim = 0)
        else:
            weights = self.weights
        
        # Apply the weights element-wise to each input tensor !!!! CDFs
        weighted_inputs = [weights[i] * input_tensor.cumsum(1) for i, input_tensor in enumerate(list_inputs)]

        # Perform the convex combination across input vectors
        combined_CDF = sum(weighted_inputs)

        return combined_CDF
    
    def train_model(self, train_loader, optimizer, epochs = 20, patience=5, projection = True):
        # define projection problem for backward pass
        lambda_proj = cp.Variable(self.num_inputs)
        lambda_hat = cp.Parameter(self.num_inputs)
        proj_problem = cp.Problem(cp.Minimize(0.5*cp.sum_squares(lambda_proj-lambda_hat)), [lambda_proj >= 0, lambda_proj.sum()==1])
        
        
        L_t = []
        best_train_loss = float('inf')

        for epoch in range(epochs):
            # activate train functionality
            self.train()
            running_loss = 0.0
            # sample batch
            for batch_data in train_loader:
                
                y_batch = batch_data[-1]
                
                #cdf_batch = [batch_data[i] for i in range(self.num_inputs)]

                # clear gradients
                optimizer.zero_grad()
                
                # forward pass: combine forecasts and solve each newsvendor problem
                comb_CDF = self.forward(batch_data[:-1])
                
                # estimate CRPS (heavyside function)
                
                
                #loss_i = [torch.square( comb_CDF[i] - 1*(self.support >= y_batch[i]) ).sum() for i in range(len(y_batch))]
                loss_i = torch.sum(torch.square( comb_CDF - 1*(self.support >= y_batch.reshape(-1,1))), 1)
                
                loss = torch.mean(loss_i)

                # Decomposition (see Online learning with the Continuous Ranked Probability Score for ensemble forecasting) 
                #divergence_i = [(weights[j]*torch.norm(self.support - y_batch[i] )) for ]
                
                #loss_i = [weights[j]*torch.abs(self.support >= y_batch[i]) ).sum() for i in range(len(y_batch))]
                #loss = sum(loss_i)/len(loss_i)
                
                # backward pass
                loss.backward()
                optimizer.step()                
                
                if (projection)and(self.apply_softmax != True):     
                    lambda_hat.value = to_np(self.weights)
                    proj_problem.solve(solver = 'GUROBI')
                    # update parameter values
                    with torch.no_grad():
                        self.weights.copy_(torch.FloatTensor(lambda_proj.value))
                
                running_loss += loss.item()
            

            L_t.append(to_np(self.weights).copy())
            average_train_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {average_train_loss:.4f} ")

            if average_train_loss < best_train_loss:
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

class AdaptiveLinearPoolCRPSLayer(nn.Module):        
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
