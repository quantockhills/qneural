import numpy as np
import torch 
import qutip as qt # * for the bloch plot 
import matplotlib.pyplot as plt 
import functools as ft 
from sympy.utilities.iterables import multiset_permutations
import torch
import matplotlib.pyplot as plt
import os 

device = 'cpu'

rabi = 2 * torch.pi * 4 #1
rabi_gg_default = 2 * np.pi * 4 * 10**(-3) # * 2pi * 4 kHz 
trial_wvfn = torch.zeros(3, 1) 
trial_wvfn[0] = 1.0 # * just one qubit in state 0, with three possible states 
# * recall that it's 0, 1, r. (so assuming gg-qubits atm) 

id_mat = torch.eye(3, dtype = torch.cfloat)
zero_fn = lambda t: torch.tensor(0.0, device = device)

zero_val = torch.tensor(0.0, device = device)

def zero_fn(t):    
    return zero_val
 
def const_fn(no):
    
    val = torch.tensor(no,  device = device)
    
    return (lambda t: val)

def const_then_zero(no, cutoff):
    
    val = torch.tensor(no, device = device)
    
    def rabi(t):
        
        if t <= cutoff:  
            return val
        else: return zero_val  
    
    return rabi

def const_then_zero_tensor(no, cutoff:torch.Tensor):
    
    val = torch.tensor(no, device = device)
    batch_ = len(cutoff)
    
    def rabi(t):
        
        f = torch.zeros(batch_, 1, device = device)
        f[t <= cutoff] = val 
        return f 
        #if t < cutoff: 
        
        #    return val 
        
        #else: return zero_val 
    return rabi     

def _optim_to_dict(network, soln, angles, input_tensor, solver, loss_val):
    # adds all the relevant objects from the optimization to a dictionary 
    # useful for saving  
    savenet = {'network':network, 'soln':soln, 'angle': angles,
            'inpt_tensor':input_tensor, 'solver':solver, 'loss_instance': loss_val}
    return savenet 

def numberToBase(n, b):
    
    # internal function 
    
    if n == torch.tensor(0):
        return '0'
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    str_aux = ''
    for i in digits[::-1]: 
        if i == 2: 
            str_aux += 'r'
        else: 
            str_aux += str(i)
    return  str_aux

def basis_tensor(state, dim = 3): # creates a basis tensor for a given string
    
    """
    Returns a torch.Tensor object corresponding to a input state string. 

    Parameters
    ----------
    state : str
        state in string form e.g. '002' is  a three-qubit state. we take convention 
        to be from low-lying to high-lying e.g. 0 and 1 here are hyperfine states and 
        2 is the Rydberg state.   
    
    dim : int
        local dimension. will be 2 for gr-qubits, 3 for gg-qubits  

    Returns
    -------
    torch.Tensor
        Basis state. 

    """

    tens = torch.zeros(dim**len(state), 1, dtype = torch.cfloat, device = device) # * odeint can (as of recent) handle complex ODEs so no need to double state space
    tens[int(state.replace('r','2'), 3)] = 1.0
    return tens 

def tens_prod(obj_list):
    
    """
    Carries out a tensor product of the wavefunctions (in torch.Tensor form) provided in 
    the list: effect on wavefunctions is equivalent to qt.tensor.

    Parameters
    ----------
    obj_list: list of torch.Tensor objects

    Returns
    -------
    torch.Tensor
        Composite state.  

    """

    return(ft.reduce(torch.kron, obj_list)) # * carries out tensor product

o_z = basis_tensor('1')*basis_tensor('0').H
o_o = basis_tensor('1')*basis_tensor('1').H
k1_br = basis_tensor('1')*basis_tensor('r').H # bra1ketr
kr_br = basis_tensor('r')*basis_tensor('r').H
krr_brr = basis_tensor('rr')*basis_tensor('rr').H

or_ro = (k1_br + k1_br.H)

def str_to_tensor(aux_list, mode = ['1r']): 
    
    # * helps construct the Hamiltonian in class Rydberg_Hamiltonian
    # internal function 
    
    if mode[0] == '1r':
        aux2 = [torch.eye(3, dtype = torch.cfloat, device = device) if i=='Id' else i for i in aux_list]
        aux2 = [or_ro if i=='or_ro' else i for i in aux2]
        aux2 = [kr_br if i=='kr_br' else i for i in aux2]
        aux2 = [krr_brr if i=='krr_brr' else i for i in aux2]
        
    elif mode[0] == 'gg':
        
        aux2 = [torch.eye(3, dtype = torch.cfloat, device = device) if i=='Id' else i for i in aux_list]
        aux2 = [(np.exp(-1.0j * mode[1]) * o_z + np.exp(1.0j * mode[1])* o_z.H) if i=='oz_zo' else i for i in aux2]
        aux2 = [o_o if i=='o_o' else i for i in aux2]

    return aux2

def basis_states_output(wvfn):
    
    # * Gives a readable output for the wavefunction
    main_list = {}
    main_list["main"], main_list["minor"] = [], []
    ctr_main = 0
    nqubits = np.emath.logn(3, len(wvfn)) # * base, number
    #print(nqubits)
    for basis in wvfn:
         
            obj = []
            obj.append(basis.item())
            basis_base = numberToBase(ctr_main, 3)
            while len(basis_base) < int(nqubits):
                basis_base += '0' 
            obj.append(basis_base)
            if abs(basis.item()) > 0.01:
                main_list["main"].append(obj)
                ctr_main += 1
            else: 
                main_list["minor"].append(obj)
                ctr_main += 1
    return main_list
 
class Rydberg_Hamiltonian: 

    """
    Represents n Hamiltonians for n systems of Rydberg atoms, each system corresponding to controls that 
    implement a different desired angle (of some parametrized unitary gate).

    This class constructs the Hamiltonian for a given number of qubits, with options for
    global or local addressing through Rabi frequency and detuning functions.

    This currently works for gg-qubits. 
    
    Attributes:
    nqubits (int): Number of qubits in the system.
    
    rabi_f (function): Time-dependent Rabi frequency function (vectorized i.e. returns rabi frequency at some time t
    for each angle).
    
    rabi_max (torch.Tensor): Maximum Rabi frequency.
    detuning (function): Detuning function (vectorized i.e. returns detuning at some time t for each angle).
    addressing (str): Type of addressing ('global' or 'local').
    
    vdd (torch.Tensor): Interaction strength.
    interaction_tensored (dict): Tensored interaction terms.
    
    det_tensored (dict): Dict consisting of the detuning functions at each qubit (in det_tensored["pulse <qubit no.>)"]) 
    and the corresponding operator (in det_tensored["qubit <qubit no.>"]) for transition between 1 and r.  
    
    rabi_tensored (dict): Dict consisting of the rabi frequency functions at each qubit (in rabi_tensored["pulse <qubit no.>)"]) 
    and the corresponding operator (in rabi_tensored["qubit <qubit no.>"]) for transition between 1 and r. 
    """

    def __init__(self, nqubits, rabi_f = lambda t: torch.tensor(rabi, device = device), detuning = zero_fn, addressing = 'global', rabi_max = rabi, decay_width = 'n/a'):
        
        self.nqubits = nqubits
        self.rabi_f = rabi_f 
        self.rabi_max = torch.tensor(rabi_max, device = device)
        self.detuning = detuning 
        self.addressing = addressing 
        self.vdd =  21.1 * torch.tensor(rabi, device = device)
        self.decay_width = decay_width
        aux_list = (nqubits - 1) * ['Id']
        aux_list.append('or_ro')
        aux_list = list(multiset_permutations(aux_list))
        rabi_tensored, det_tensored, interaction_tensored = {}, {}, {}
        
        for i in aux_list: 

            t1 = tens_prod(str_to_tensor(i))
            rabi_tensored['qubit ' + str(i.index('or_ro'))] = t1
            # rabi_tensored['conj. qubit ' + str(i.index('or_ro'))] = t1.mH?

            if self.addressing == 'global':
                rabi_tensored['pulse ' + str(i.index('or_ro'))] = self.rabi_f
            elif self.addressing == 'local': 
                rabi_tensored['pulse ' + str(i.index('or_ro'))] = self.rabi_f[(i.index('or_ro'))]
        
        aux_list = (nqubits - 1) * ['Id']
        aux_list.append('kr_br') # * det. term
        aux_list = list(multiset_permutations(aux_list))
        
        for i in aux_list:
        
            det_tensored['qubit ' + str(i.index('kr_br'))] = tens_prod(str_to_tensor(i))
            if self.addressing == 'global':
                det_tensored['pulse ' + str(i.index('kr_br'))] = self.detuning
            elif self.addressing == 'local':
                det_tensored['pulse ' + str(i.index('kr_br'))] = self.detuning[(i.index('kr_br'))]
        
        self.interaction_tensored = 'n/a'
        
        if nqubits >= 2: 

            # * new version below
            aux_list = (nqubits - 2)*['Id'] 
            aux_list.append('kr_br')
            aux_list.append('kr_br')  
            aux_list = list(multiset_permutations(aux_list))
            interaction_tensored['total'] = 0.0
            for i in aux_list: 

                #indices = [j for j in range(len(i))\
                #          if i[j] == 'kr_br']
                
                #interaction_tensored['qubits ' + \
                #    str(indices)] = \    
                interaction_tensored['total'] += self.vdd*tens_prod(str_to_tensor(i))
            
            self.interaction_tensored = interaction_tensored

        self.det_tensored = det_tensored
        self.rabi_tensored = rabi_tensored
        
    def gg_initialize(self, rabi_f = lambda t: torch.tensor(rabi_gg_default).to(device), detuning = zero_fn, target = ['0'], phase = 0.0, addressing = 'local', rabi_max_gg = rabi_gg_default):
        # ! This probably will not work at the moment
        # * lags behind new code, fix it ! 
        # probably not that important for our use cases but might be relevant later 
        # * target is (in general) an array 
        self.rabi_max_gg = torch.tensor(rabi_max_gg).to(device)
        
        if addressing == 'global': 
            self.detuning_gg = detuning
            self.rabi_f_gg = rabi_f
        
        elif addressing == 'local':
        
            self.detuning_gg = []
            self.rabi_f_gg = []
            
            for i in np.arange(0, self.nqubits): 
                
                if str(i) in target:
                    
                    self.rabi_f_gg.append(rabi_f[i])    
                    self.detuning_gg.append(detuning[i])
                
                else: 
                    self.detuning_gg.append(zero_fn)
                    self.rabi_f_gg.append(zero_fn)
        
        self.phase_gg = torch.tensor(phase).to(device)
        self.addressing_gg = addressing
        self.target_index = ['0'] 
        self.addressing_gg = addressing
        
        rabi_tensored, det_tensored = {}, {}
        aux_list = (self.nqubits - 1)*['Id']
        aux_list.append('oz_zo')
        aux_list = list(multiset_permutations(aux_list))
        
        for i in aux_list:
            rabi_tensored['qubit ' + str(i.index('oz_zo'))] = tens_prod(str_to_tensor(i, mode = ['gg', self.phase_gg.cpu()]))
            if self.addressing_gg == 'global':
                rabi_tensored['pulse ' + str(i.index('oz_zo'))] = self.rabi_f_gg
            elif self.addressing_gg == 'local':
                rabi_tensored['pulse ' + str(i.index('oz_zo'))] = self.rabi_f_gg[(i.index('oz_zo'))]
        
        aux_list = (self.nqubits - 1) * ['Id']
        aux_list.append('o_o') # * det. term
        aux_list = list(multiset_permutations(aux_list))
        
        for i in aux_list:
            det_tensored['qubit ' + str(i.index('o_o'))] = tens_prod(str_to_tensor(i, mode = ['gg', 0]))        

            if self.addressing_gg == 'global':
                det_tensored['pulse ' + str(i.index('o_o'))] = self.detuning_gg
        
            elif self.addressing_gg == 'local':
                det_tensored['pulse ' + str(i.index('o_o'))] = self.detuning_gg[(i.index('o_o'))]
        
        self.det_tensored_gg = det_tensored
        self.rabi_tensored_gg = rabi_tensored
    
sigmax = torch.tensor([[0, 1], [1, 0]], dtype = torch.cfloat).to(device)
sigmay = torch.tensor([[0, -1.0j], [1.0j, 0]], dtype = torch.cfloat).to(device)
sigmaz = torch.tensor([[1, 0], [0, -1]], dtype = torch.cfloat).to(device)

# * For alphas in [0, pi]

exclude_row = lambda mat, row_no: torch.cat((mat[:row_no],mat[row_no+1:]))

exclude_col = lambda mat, col_no: torch.cat([mat[:, :col_no],mat[:, col_no+1:]], dim=1)

def reduce_unitary_dim(unitary, dim_name = '2'): 
# ! go thru rest of code to confirm that 'r' does not matter too much     
    # * NOTE: Generally returns a non-unitary matrix (bc. of leakage to Rydberg state)
    ctr = 0
    for i in torch.arange(0, unitary.shape[0]):

        #index_no = numberToBase(i, 3)
        index_no = np.base_repr(i, 3)
        #print(index_no)
        if dim_name in index_no:     
            #index_no = index_no.replace('r','2')
            unitary = exclude_col(unitary, int(index_no, 3) - ctr)
            unitary = exclude_row(unitary, int(index_no, 3) - ctr)
            ctr += 1
            
    return unitary 

def unitary_infidelity_array(u1, u2, dim = 2, nqbits = 1):

    # * average across n instances 
    """
    Calculate average infidelity over two batches (of size m) of unitary matrices of shape n x n (see Phys. Rev. Lett. 129, 050507 for the metric used).

    Parameters:
    u1 (torch.Tensor): Batch of unitary matrices (shape: m x n x n).
    u2 (torch.Tensor): Another batch of unitary matrices (same shape as u1).
    dim (int, optional): Dimension of local hilbert space (default: 2). For instance 3 would correspond to 0, 1, r (rydberg state)
    nqbits (int, optional): Number of qubits (default: 1).

    Returns:
    torch.Tensor: Total loss rate (a float value).
    """
    
    c = torch.einsum('mij, nji -> mn',u1.mH, u2)
    g = torch.einsum('mm ->', c**2) # * contraction
    f_ = 1 - torch.abs(1/(u1.shape[0]*dim**(2*nqbits))*g)
    return f_

oneqrot = lambda alpha_1, alpha_2, alpha_3: torch.linalg.multi_dot([torch.matrix_exp(-1.0j*(alpha_1/2)*sigmaz),torch.matrix_exp(-1.0j*(alpha_2/2)*sigmay), torch.matrix_exp(-1.0j*(alpha_3/2)*sigmaz)])

def gg_bloch_plot(wvfn): # * only works with single qubit wvfn 
        
        b = qt.Bloch()
        
        for i in wvfn:

            b.add_states(qt.Qobj(np.array(i)))
        
        b.make_sphere()
        b.show()
        
def list_to_fn_tensor(aux_list:torch.Tensor, gatetime:torch.Tensor, time_steps):
    """
    Create a multi-pulse function from a list of amplitudes. See also 'list_to_fn_tensor_var_gatetime'
    in const_czphi.py . 

    Parameters:
    aux_list (torch.Tensor): batch of amplitudes (shape m*n, see nn_tutorial.ipynb for more).
    gatetime (torch.Tensor): Total gate time.
    time_steps (int): Number of time steps.

    Returns:
    function: A multi-pulse function that takes time t and returns the corresponding amplitude
              for each instance in the batch in shape k*1 for k samples.
    """
    
    # i note that this was done for a case w one gatetime 
    # For now we can set the time to zero after the gatetime prediction from the network 
    batch_size = int(len(aux_list)/time_steps)
    # * for when an input tensor consists of the function 
    step_size = gatetime/time_steps # step_size is now an array 
    b = aux_list.reshape(batch_size, time_steps)
    # each row is a different time step, each column a different angle 

    def multi_pulse(t):

        if t < gatetime: 
            j = b.select(1, torch.floor(t/step_size).long()).reshape(b.shape[0], 1)       
        else: return torch.zeros(b.shape[0], 1) 
        #? can the else statement lead to requires_grad issues? 
        # a. don't think so as it's not a NN output 
        # reshape too? 
 
        return j 

    return multi_pulse
    # aux_list is now just a horizontal stack of different angle hamiltonians 
    #aux_list = torch.tensor(aux_list)
    # the for loop should be outside i.e. not in the pulse 


# ******************************************************
# * Code for testing the neural networks 
# ******************************************************
# * the factor of 0.5 was missing, now changed

rx_gate = lambda theta: torch.matrix_exp(-0.5j*theta*sigmax) 
ry_gate = lambda theta: torch.matrix_exp(-0.5j*theta*sigmay)
rz_gate = lambda theta: torch.matrix_exp(-0.5j*theta*sigmaz) 

def ry_gate_stack(theta_tensor):
# Does not need to be differentiable, just a helper to compute the auxilary function
    aux_list = []
    for i in theta_tensor:
        aux_list.append(ry_gate(i))

    return torch.stack(aux_list)

def czp_gate_stack(theta_tensor:torch.Tensor):
    
    """
    Create a stack of controlled-Z_phi gates i.e. parametrized gates.

    Parameters:
    theta_tensor (torch.Tensor): Tensor containing phase angles.

    Returns:
    torch.Tensor: Stack of the associated unitaries.
    """

    aux_list = []
    for i in theta_tensor:
        czp = torch.eye(4, dtype = torch.cfloat, device = device)
        czp[3][3] = torch.exp(1.0j*i)
        aux_list.append(czp)
    return torch.stack(aux_list)
   
def cczp_gate_stack(theta_tensor:torch.Tensor):
    
    """
    Create a stack of controlled-Z_phi gates i.e. parametrized gates.

    Parameters:
    theta_tensor (torch.Tensor): Tensor containing phase angles.

    Returns:
    torch.Tensor: Stack of the associated unitaries.
    """

    aux_list = []
    for i in theta_tensor:
        cczp = torch.eye(8, dtype = torch.cfloat, device = device)
        cczp[-1][-1] = torch.exp(1.0j*i)
        aux_list.append(cczp)
    return torch.stack(aux_list)
   
def cczp_gate_stack_zzz(theta_tensor:torch.Tensor):
    
    """
    Create a stack of controlled-Z_phi gates i.e. parametrized gates.

    Parameters:
    theta_tensor (torch.Tensor): Tensor containing phase angles.

    Returns:
    torch.Tensor: Stack of the associated unitaries.
    """

    aux_list = []
    for i in theta_tensor:
        cczp = -1.0*torch.eye(8, dtype = torch.cfloat, device = device)
        cczp[0][0] = -torch.exp(1.0j*i)
        aux_list.append(cczp)
    return torch.stack(aux_list)
   


torch_cz = torch.zeros(9, 9, dtype = torch.cfloat, device = device) # cz in the 0, 1, r basis 
torch_cz[1, 1] = torch_cz[3, 3] = torch_cz[4, 4] = -1.0 
torch_cz[0, 0] = 1.0

torch_cz_dim_2 = torch.eye(4, dtype = torch.cfloat, device = device) # cz in the 0, 1 basis 
torch_cz_dim_2[3, 3] = -1.0
zer = qt.basis(2, 0)
one = qt.basis(2, 1)

# * See for instance Rev. Mod. Phys. 82, 2313 (2010) for the Jaksh definition of the CZ protocol as defined below

def rabi_jaksch_control(t):
    rabi_t = 0
    if t < torch.pi/rabi:
        rabi_t = rabi
    elif t >= 3*torch.pi/rabi and t < 4*torch.pi/rabi:
        rabi_t = rabi
    return rabi_t

def rabi_jaksch_target(t):
    rabi_t = 0
    if t >= torch.pi/rabi and t < 3 *torch.pi/rabi:
        rabi_t = rabi 
    return rabi_t 

def pulse_plot(pulse, time, rabi_max = rabi):
    
    plt.xlabel(r"Time ($\Omega_{\mathrm{max}}t$)")
    plt.ylabel(r"Rabi frequency ($\Omega/\Omega_{\mathrm{max}}$)")
    time_lin = np.linspace(0, time, 51)
    pulse_list = []
    
    for i in time_lin: 

        pulse_list.append(pulse(i)/rabi_max)
    plt.xlim(0, time*rabi_max)
    plt.plot(time_lin*rabi_max, np.array(pulse_list))
    plt.show()

def corr_1q_rotation(phi_01): 
    # * phi_01 should be a tensor 
    # * make this autodiff friendly:) 

    j = torch.eye(4, dtype = torch.cfloat, device = device)
    j1 = torch.exp(-1.0j*phi_01)
    j[1,1] = j[2, 2] = j1
    j[3, 3] = j1.square()
    return j

def save(data, file_path):
    """
    Save the data using torch.save only if the file does not already exist.

    Parameters:
    data (torch.Tensor or torch.nn.Module): The data to save.
    file_path (str): The path to save the file.
    """
    if not os.path.exists(file_path):
        torch.save(data, file_path)
        print(f"File saved to {file_path}")
    else:
        print(f"File {file_path} already exists. Save operation skipped.")

def normalize_tensor(tensor):
    # Step 1: Calculate the mean of the tensor
    mean = tensor.mean()
    
    # Step 2: Subtract the mean from the tensor to center it around zero
    centered_tensor = tensor - mean
    
    # Step 3: Calculate the standard deviation of the centered tensor
    std_dev = centered_tensor.std()
    
    # Step 4: Divide the centered tensor by its standard deviation to normalize the variance to 1
    normalized_tensor = centered_tensor / std_dev
    
    return normalized_tensor
