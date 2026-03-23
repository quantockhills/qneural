# constants and fast functions for the cz gate optimization
# in general, problem specific constants 

import torch 
import cst_n_fn as cfn 
import schsolve
import numpy as np 
import matplotlib.pyplot as plt 
import torch.nn as nn 

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

device = 'cpu'

angle_batch = 1#80 # 58  
time_steps = 201
range_rabi, range_detuning = [0, cfn.rabi], [-2*cfn.rabi, 2*cfn.rabi]
hilbert_space_dim = 3**2 # for two qubit case w 0, 1, r
hilbert_space_dim_reduced = 2**2 # 0, 1
system = cfn.Rydberg_Hamiltonian(nqubits = 2, rabi_f = cfn.zero_fn, \
                                detuning = cfn.zero_fn, 
                                addressing = 'global', rabi_max = cfn.rabi)

system.gg_initialize(rabi_f = [cfn.zero_fn]*2, detuning = [cfn.zero_fn]*2,\
                    target = ['0','1'], rabi_max_gg = 0.0, addressing = 'local') 

instance = schsolve.schsolver(system, angle_batch, mode = "family").to(device)

init_matrix = torch.zeros(angle_batch, hilbert_space_dim, hilbert_space_dim, dtype = torch.cfloat, device = device)

for i in range(angle_batch):

    init_matrix[i] = torch.eye(hilbert_space_dim, dtype = torch.cfloat, device = device)
    
init_matrix.requires_grad_(True)

a_to_keep = [0, 1, 3, 4]*4
b_to_keep = [0,0,0,0,1,1,1,1,3,3,3,3,4,4,4,4]

def corr_1q_rotation_fast(phi_01): 
    
    j = torch.eye(4, dtype = torch.cfloat).to('cpu')
    j1 = torch.exp(-1.0j*phi_01)
    j[1,1] = j[2, 2] = j1
    j[3, 3] = j1.square()
    return j

def corr_1q_rotation_fast_vector(phi_01, angle_batch = angle_batch): 

    identity = torch.eye(hilbert_space_dim_reduced, dtype = torch.cfloat, device = device)
    j = identity.unsqueeze(0).repeat(angle_batch, 1, 1)
    j1 = torch.exp(-1.0j*phi_01)
    j[:, 1, 1] = j[:, 2, 2] = j1
    j[:, 3, 3] = j1.square()
    return j

def reduce_r_dim_2q_vector(unitary, angle_batch = angle_batch): 

    return unitary[:, a_to_keep, b_to_keep].view(angle_batch, 4, 4).transpose_(1, 2) 

def correction_1q(sol_intrm, angle_batch = angle_batch):
    
    # torch.bmm: batch matrix multiplication 
    mat0 = corr_1q_rotation_fast_vector(torch.angle(sol_intrm[:, 1, 1]), angle_batch)
    return torch.bmm(mat0, sol_intrm)


def _widget_plot_pulse_3d(network, input_tensor, angles, control = 'rabi',\
                         range_rabi = range_rabi, range_detuning = range_detuning, save_flag = [False]):

    time_arr = torch.linspace(0, gatetime, time_steps).reshape((time_steps, 1)).to(device)
    time_arr = torch.cat([time_arr]*angle_batch)
    
    x = time_arr.reshape(angle_batch*time_steps).cpu().detach().numpy()*cfn.rabi # * should work for time-optimal case but more care needs to be put in 
    y = angles.reshape(angle_batch*time_steps).cpu().detach().numpy()/torch.pi

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel(r'$\Omega_{max}$T')
    ax.set_ylabel(r'$\alpha/\pi$')
    
    if control == 'rabi':
        pred_outputs_rabi = (network(input_tensor).select(1,0))*(range_rabi[1] - range_rabi[0]) + range_rabi[0]
        ax.plot_trisurf(x, y, pred_outputs_rabi.detach().numpy()/cfn.rabi)
        ax.set_zlabel(r'$\Omega/\Omega_{max}$') 

    elif control == 'detuning':
        pred_outputs_det = (network(input_tensor).select(1,1))*(range_detuning[1] - range_detuning[0]) + range_detuning[0]
        ax.plot_trisurf(x, y, pred_outputs_det.detach().numpy()/cfn.rabi)
        ax.set_zlabel(r'$\Delta/\Omega_{max}$') 
    if save_flag[0] == True:
        plt.savefig(save_flag[1], transparent=True)


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


