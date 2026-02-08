import numpy as np
from qiskit.quantum_info import SparsePauliOp

geometries = {
        'LiH' : 'Li 0 0 0; H 0 0 1.595',

        'H2' : 'H 0 0 0; H 0 0 0.741;',
        
        'BeH2' : 
            'Be 0 0 0; H 0 0 1.339; H 0 0 -1.339;',
    }

def make_geometry(atomic_symbol):
    return geometries[atomic_symbol]
  
def get_exact_fci_energy(hamiltonian: SparsePauliOp):
    """
    Computes the exact ground state energy (FCI) by diagonalizing the Hamiltonian.
        Args:
            - atomic_symbol: str, atomic symbol of the molecule of interest. Only 'H2' and 'LiH'
            possible
        Returns:
            - fci_energy: FCI ground-state energy of the molecule
    """
    # Convert SparsePauliOp to a dense NumPy matrix
    H_matrix = hamiltonian.to_matrix()
    
    # Use NumPy to find eigenvalues (eigh is for Hermitian matrices)
    eigenvalues = np.linalg.eigvalsh(H_matrix)
    
    # The smallest eigenvalue is the Ground State Energy
    fci_energy = eigenvalues[0]
    
    return fci_energy