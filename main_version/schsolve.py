import torch
import torchdiffeq as tdf
import cst_n_fn as cfn
import numpy as np
import torch.nn as nn
import torch
import torch.jit as jit 

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

device = "cpu"
'''
def time_ansatz(nl, nu, input_dim, output_dim, beta = 0): 
    a = 1 
    c = nn.Sequential()
    first_layer = nn.Linear(input_dim, nu)
    first_layer.weight_data *= beta
    for i in torch.arange(a):
        a = 1
'''
def linear_n_nu_ansatz(nl = 2, nu = 4, input_dim = 2, output_dim = 2, beta = 'n/a'):

    """
    Creates a neural network ansatz with a variable amount of layers and neurons.

    Args:
        nl (int): Number of layers in the neural network.
        nu (int): Number of units (nodes) in each layer.
        input_dim (int): Dimension of input data (rotation angles).
        output_dim (int): Dimension of output data (phase, rabi frequency, detuning etc.).
        beta (str): Scaling factor (optional).

    Returns:
        nn.Sequential: Neural network model.
    """

    c = nn.Sequential()
    first_layer = nn.Linear(input_dim, nu)
    first_layer.weight.data *= beta
    c.append(first_layer) # * first layer
    c.append(nn.ReLU())
    
    for i in torch.arange(0, nl - 2): # * -2 bc we have already appended one layer above, and will append one input layer in the end 
        hidden_layer = nn.Linear(nu, nu)
        hidden_layer.weight.data *= beta
        c.append(hidden_layer)
        c.append(nn.ReLU())
    output_layer = nn.Linear(nu, output_dim)
    output_layer.weight.data *= beta
    c.append(output_layer)
    c.append(nn.Sigmoid()) # * ordering of output_dim shouldn't really matter 
    return c

def linear_n_nu_ansatz_no_sig(nl = 2, nu = 4, input_dim = 2, output_dim = 2, beta = 'n/a'):

    """
    Creates a neural network ansatz with a variable amount of layers and neurons.

    Args:
        nl (int): Number of layers in the neural network.
        nu (int): Number of units (nodes) in each layer.
        input_dim (int): Dimension of input data (rotation angles).
        output_dim (int): Dimension of output data (phase, rabi frequency, detuning etc.).
        beta (str): Scaling factor (optional).

    Returns:
        nn.Sequential: Neural network model.
    """

    c = nn.Sequential()
    first_layer = nn.Linear(input_dim, nu)
    first_layer.weight.data *= beta
    c.append(first_layer) # * first layer
    c.append(nn.ReLU())
    
    for i in torch.arange(0, nl - 2): # * -2 bc we have already appended one layer above, and will append one input layer in the end 
        hidden_layer = nn.Linear(nu, nu)
        hidden_layer.weight.data *= beta
        c.append(hidden_layer)
        c.append(nn.ReLU())
    output_layer = nn.Linear(nu, output_dim)
    output_layer.weight.data *= beta
    c.append(output_layer)
    c.append(nn.Tanh()) # * ordering of output_dim shouldn't really matter 
    return c

def linear_n_nu_ansatz_batchnorm(nl=2, nu=4, input_dim=2, output_dim=2, beta='n/a'):
    """
    Creates a neural network ansatz with a variable amount of layers and neurons.

    Args:
        nl (int): Number of layers in the neural network.
        nu (int): Number of units (nodes) in each layer.
        input_dim (int): Dimension of input data (rotation angles).
        output_dim (int): Dimension of output data (phase, rabi frequency, detuning etc.).
        beta (str): Scaling factor (optional).

    Returns:
        nn.Sequential: Neural network model.
    """

    c = nn.Sequential()
    first_layer = nn.Linear(input_dim, nu)
    first_layer.weight.data *= beta
    c.append(first_layer)  # * first layer
    c.append(nn.BatchNorm1d(nu))
    c.append(nn.ReLU())
    
    for i in torch.arange(0, nl - 2):  # * -2 bc we have already appended one layer above, and will append one input layer in the end 
        
        hidden_layer = nn.Linear(nu, nu)
        hidden_layer.weight.data *= beta
        c.append(hidden_layer)
        c.append(nn.BatchNorm1d(nu))
        c.append(nn.ReLU())
    
    output_layer = nn.Linear(nu, output_dim)
    output_layer.weight.data *= beta
    c.append(output_layer)
    c.append(nn.Sigmoid())  # * ordering of output_dim shouldn't really matter 
    return c

class neural_trainer(nn.Module):

    def __init__(self, input_dim = 2, output_dim = 2, nl = 2, nu = 40, beta = 1.8):
        super(neural_trainer, self).__init__()
        
        # * 6, 150 in the paper 
        
        self.ansatz = linear_n_nu_ansatz(nl, nu, input_dim = input_dim, output_dim = output_dim, beta = beta)
        
    def forward(self, x):
        # * x is a vector
        # ! x[0] = alpha, x[1] = time
        with torch.enable_grad():
    
            fpropval = self.ansatz(x)
    
        return fpropval 

class neural_trainer_batch_norm(nn.Module):

    def __init__(self, input_dim = 2, output_dim = 2, nl = 2, nu = 40, beta = 1.8):
        super(neural_trainer, self).__init__()
        
        # * 6, 150 in the paper 
        
        self.ansatz = linear_n_nu_ansatz_batchnorm(nl, nu, input_dim = input_dim, output_dim = output_dim, beta = beta)
        
    def forward(self, x):
        # * x is a vector
        # ! x[0] = alpha, x[1] = time
        with torch.enable_grad():
    
            fpropval = self.ansatz(x)
    
        return fpropval 


class schsolver(nn.Module):

    def __init__(self, hamiltonian, angle_batch, mode = "single"):
        # mode can be family or single angle 
        super(schsolver, self).__init__()
        self.hamiltonian = hamiltonian # * inputs can be a list of functions 
        self.matrix_size = self.hamiltonian.rabi_tensored["qubit 0"].shape[0] # square matrix  
        self.angle_batch = angle_batch  

        if mode == "family":
            g = self.hamiltonian.interaction_tensored["total"] 
            self.hamiltonian.interaction_tensored["total"] \
                = g.unsqueeze(0).expand(self.angle_batch, -1, -1)
        
        def par_hamiltonian(t):
                
                aux_rabi_det = 0.0 + 0.0j

                for i in torch.arange(0, self.hamiltonian.nqubits):
                    aux_rabi_det += \
                    (0.5*self.hamiltonian.rabi_tensored["pulse " + str(i.item())](t).unsqueeze(1) * \
                    self.hamiltonian.rabi_tensored["qubit " + str(i.item())] + \
                    self.hamiltonian.det_tensored["pulse " + str(i.item())](t).unsqueeze(1) * \
                    self.hamiltonian.\
                        det_tensored["qubit " + str(i.item())])#\
                            #.view(self.angle_batch*self.matrix_size, \
                    aux_rabi_det += -1.0j*self.hamiltonian.decay_width/2 * \
                        self.hamiltonian.det_tensored["qubit " + str(i.item())]

                if self.hamiltonian.nqubits >= 2:

                    aux_rabi_det += self.hamiltonian.interaction_tensored["total"]
                                     
                return aux_rabi_det 
                
        self.par_hamiltonian = par_hamiltonian
    
    def forward(self, t, propagator): 

        # * ih dU/dt = H(rabi(t), det(t)) U
        with torch.enable_grad():
            # self.par_hamiltonian now has shape (angle_batch, hspace dim, hspace dim)

            du_dt = -1.0j*torch.bmm(self.par_hamiltonian(t), propagator)  
        
        return du_dt
    
hmltn = cfn.Rydberg_Hamiltonian(1, rabi_f = cfn.zero_fn) 

class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, initial_lr, target_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr = self.initial_lr + (self.target_lr - self.initial_lr) * (self.current_step / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
