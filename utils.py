import numpy as np
from qiskit.quantum_info import SparsePauliOp

geometries = {
        'LiH' : 'Li 0 0 0; H 0 0 1.595',

        'H2' : 'H 0 0 0; H 0 0 0.7410102132613643;',
        
        'BeH2' : 
            'Be 0 0 0; H 0 0 1.339; H 0 0 -1.339;',
    }

def make_geometry(name_mol):
    return geometries[name_mol]
  
def get_exact_fci_energy(hamiltonian: SparsePauliOp):
    """
    Computes the exact ground state energy (FCI) by diagonalizing the Hamiltonian.
    """
    # 1. Convert SparsePauliOp to a dense NumPy matrix
    # This works well for small systems (H2, LiH, < 12 qubits)
    H_matrix = hamiltonian.to_matrix()
    
    # 2. Use NumPy to find eigenvalues (eigh is for Hermitian matrices)
    eigenvalues = np.linalg.eigvalsh(H_matrix)
    
    # 3. The smallest eigenvalue is the Ground State Energy
    fci_energy = eigenvalues[0]
    
    return fci_energy