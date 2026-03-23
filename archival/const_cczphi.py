# constants and fast functions for the ccz gate optimization
# in general, problem specific constants 

import torch 
import cst_n_fn as cfn 
import schsolve
import matplotlib.pyplot as plt 
import torch.nn as nn 
import numpy as np 

angle_batch = 1#80 
time_steps = 201
hilbert_space_dim = 3**3 # Now for three qubits
hilbert_space_dim_reduced = 2**3
device = 'cpu'

range_rabi, range_detuning = [0.0, cfn.rabi], [-2.0*cfn.rabi, 2.0*cfn.rabi]

system = cfn.Rydberg_Hamiltonian(nqubits = 3, rabi_f = cfn.zero_fn, \
                                detuning = cfn.zero_fn, 
                                addressing = 'global', rabi_max = cfn.rabi)

system.gg_initialize(rabi_f = [cfn.zero_fn]*3, detuning = [cfn.zero_fn]*3,\
                    target = ['0','1'], rabi_max_gg = 0.0, addressing = 'local') 

instance = schsolve.schsolver(system, angle_batch, mode = "family").to(device)

init_matrix = torch.zeros(angle_batch, hilbert_space_dim, hilbert_space_dim, dtype = torch.cfloat, device = device)

for i in range(angle_batch):

    init_matrix[i] = torch.eye(hilbert_space_dim, dtype = torch.cfloat, device = device)
    
init_matrix.requires_grad_(True)

def corr_1q_rotation_fast_vector(phi_01, angle_batch = angle_batch): 
    # this is what a Z rotation over three qubits looks like 
    identity = torch.eye(hilbert_space_dim_reduced, dtype = torch.cfloat, device = device)
    j = identity.unsqueeze(0).repeat(angle_batch, 1, 1)
    j1 = torch.exp(-1.0j*phi_01)
    j[:, 1, 1] = j[:, 2, 2] = j[:, 4, 4] = j1
    j[:, 3, 3] = j[:, 5, 5] = j[:, 6, 6] = j1**2 #pow_(2)
    j[:, 7, 7] = j1**3 #.pow_(3) 
    return j

identity = torch.eye(hilbert_space_dim_reduced, dtype = torch.cfloat, device = device)
identity = torch.flip(identity, [0]) # global x rotation 
global_x_rotation = identity.unsqueeze(0).repeat(angle_batch, 1, 1)

index_list = [0, 1, 3, 4, 9, 10, 12, 13]
a_to_keep = index_list*hilbert_space_dim_reduced
b_to_keep = [item for item in index_list \
            for _ in range(hilbert_space_dim_reduced)]

def reduce_r_dim_3q_vector(unitary, angle_batch = angle_batch): 
    
    # * should return a 8x8 matrix
    
    return unitary[:, a_to_keep, b_to_keep].view(angle_batch, \
        hilbert_space_dim_reduced, hilbert_space_dim_reduced).transpose_(1, 2) 

def correction_1q(sol_intrm, angle_batch = angle_batch):
    
    # torch.bmm: batch matrix multiplication 
    mat0 = corr_1q_rotation_fast_vector((torch.angle(sol_intrm[:, 1, 1])), angle_batch)
    return torch.bmm(mat0, sol_intrm)

def correction_1q_param(sol_intrm, phi_01, angle_batch = angle_batch):
    
    # torch.bmm: batch matrix multiplication 
    mat0 = corr_1q_rotation_fast_vector(phi_01, angle_batch)
    return torch.bmm(mat0, sol_intrm)

# ! Until here it seems I have adapted everything to three qubits


class neural_trainer_time_optimal_cz(nn.Module):
    
    def __init__(self, nl1 = 3, nu1 = 40, nl2 = 6, nu2 = 110,\
         beta_control = 1.8, time_bounds = [5/cfn.rabi, 10/cfn.rabi], \
            beta_time = 1.0, range_detuning = range_detuning,\
                mode = 'sig', scale_factor = 1.0):
        super(neural_trainer_time_optimal_cz, self).__init__()
        
        # * 6, 150 in the paper to
        
        self.ansatz_control = schsolve.linear_n_nu_ansatz(nl2, nu2, input_dim = 2, output_dim = 1, beta = beta_control).to(device) # takes in angle, time, outputs pulse
        # _aux_tensor below might not work 
        self._aux_tensor = torch.linspace(0, 1.0, time_steps, device = device)
        self.time_bounds = time_bounds
        self.debug_gradient_time = 0.0
        self.debug_gradient_control = 0.0
        self.scale_factor = scale_factor
        self.range_detuning = range_detuning
        self.flag = False
        if mode == 'sig':
            self.ansatz_time = schsolve.linear_n_nu_ansatz(nl1, nu1, input_dim = 1, output_dim = 1, beta = beta_time).to(device)
            self.scale = [0.0, 1.0]
        elif mode == 'tanh':
            self.ansatz_time = schsolve.linear_n_nu_ansatz_no_sig(nl1, nu1, input_dim = 1, output_dim = 1, beta = beta_time).to(device)
            self.scale = [1.0, 2.0]
            #self.time_bounds = [0.0, 1.0]

    def forward(self, x):
                
        with torch.enable_grad():

            init_prop = (self.ansatz_time(x) + self.scale[0])/self.scale[1]

            self.debug_gradient_time = init_prop
            
            self.gatetime_prediction = (init_prop*(self.time_bounds[1] - self.time_bounds[0]) + self.time_bounds[0])

            time_inpt_2 = (self._aux_tensor*(self.gatetime_prediction))

            x_interleave = torch.repeat_interleave(x, time_steps, dim = 0)
            time_inpt_proc = time_inpt_2.reshape(time_inpt_2.numel(), 1)
            input_tensor = torch.cat([x_interleave, time_inpt_proc], dim = -1)
            fpropval = self.ansatz_control(input_tensor) 
        
        return fpropval 

def list_to_fn_tensor_var_gatetime(aux_list:torch.Tensor, gatetime:torch.Tensor, time_steps = time_steps):

    """
    Create a multi-pulse function from a list of amplitudes. This one is specifically tailored for the czphi 
    optimization problem. 

    Parameters:
    aux_list (torch.Tensor): batch of amplitudes (shape m*n, see nn_tutorial.ipynb for more).
    gatetime (torch.Tensor): Total gate time: this is now a tensor too 
    time_steps (int): Number of time steps.

    Returns:
    function: A multi-pulse function that takes time t and returns the corresponding amplitude
              for each instance in the batch in shape k*1 for k samples.
    """
    
    batch_size = int(len(aux_list)/time_steps) 
    
    step_size = gatetime/time_steps 
    b = aux_list.reshape(batch_size, time_steps)

    def multi_pulse(t):

        pulse = torch.zeros(batch_size, 1, device = device) + 20*cfn.rabi 
        # above, large value can be added bc off-resonance (to ensure no state transfer after gatetime)  
        
        floor_index = torch.floor(t/step_size).long()
        bool_label = floor_index<time_steps
        current_step = floor_index[bool_label] #returns col vals. (for given rows below) to be retained 
        index_arr = torch.nonzero(bool_label).select(1,0) #returns the rows to be retained  
        pulse[bool_label] = b[index_arr, current_step]
        
        return pulse

    return multi_pulse

trial_angle_batch = 1 
class arbitrary_1q_global_3(nn.Module):
    
    # * Performs a global single-qubit rotation on a system of three qubits 
    # * note that it only works for one qubit for the time being! 
    # fg = ExponentialModel()
    def __init__(self):

        super(arbitrary_1q_global_3, self).__init__()
        sigma_x = torch.flip(torch.eye(2, dtype = torch.cfloat), [0])
        sigma_y = torch.flip(torch.eye(2, dtype = torch.cfloat), [0])
        sigma_y[0][1] = -1.0j
        sigma_y[1][0] = 1.0j
        sigma_z = torch.eye(2, dtype = torch.cfloat)
        sigma_z[1][1] = -1.0
        self.sigvec = torch.stack([sigma_x, sigma_y, sigma_z])
        self.f = nn.Parameter(2*torch.pi*torch.rand(trial_angle_batch, 3).reshape(3,1,1))  # Initial guess for theta
        # print(self.sigvec)

    def forward(self):
    
        #print(self.sigvec.shape)
        dot_ = 0.5*self.f*self.sigvec
        U = torch.matrix_exp(-1.0j*torch.sum(dot_, dim = [0]))
        tensor_product = torch.kron(torch.kron(U, U), U) # this will give us a thing of three qubits
        return tensor_product 


class arbitrary_z_global(nn.Module):
    
    # * Performs a global single-qubit rotation on a system of three qubits 
    # * note that it only works for one qubit for the time being! 
    # fg = ExponentialModel()
    def __init__(self):

        super(arbitrary_z_global, self).__init__()
        #self.sigma_z = torch.eye(2, dtype = torch.cfloat)
        #self.sigma_z[1][1] = -1.0
        self.f = nn.Parameter(torch.pi*(2*torch.rand(trial_angle_batch, 1, dtype=torch.float64) - 1.0))  # Initial guess for theta
        # print(self.sigvec)

    def forward(self):
    
        #print(self.sigvec.shape)
        #dot_ = 0.5*self.f*self.sigma_z
        #U = torch.matrix_exp(-1.0j*dot_)
        rot_ = corr_1q_rotation_fast_vector(self.f, angle_batch)
        #tensor_product = torch.kron(torch.kron(U, U), U) # this will give us a thing of three qubits
        return rot_

