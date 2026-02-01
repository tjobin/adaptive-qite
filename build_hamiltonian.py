
from qiskit_nature.second_q.circuit.library.initial_states.hartree_fock import HartreeFock
from utils import make_geometry
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, BravyiKitaevSuperFastMapper, BravyiKitaevMapper
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

# From IBM "Hamiltonians for Quantum Chemistry,
# https://quantum.cloud.ibm.com/learning/en/courses/quantum-chem-with-vqe/hamiltonian-construction
# Modified by Timothé Jobin


def build_hamiltonian_and_state(
        geometry : str,
        basis_set : str,
        n_elec : int,
        active_orb : int,
        mapper,
        ):
    
    driver = PySCFDriver(
        atom=geometry,
        basis=basis_set,
        charge=0,
        spin=0
    )
    transformer = ActiveSpaceTransformer(
        num_electrons=n_elec,            # Keep 2 valence electrons
        num_spatial_orbitals=active_orb      # Keep 3 orbitals (e.g. HOMO, LUMO, LUMO+1)
    )

    # 2. Run the Driver to get the Electronic Problem
    problem = driver.run()
    reduced_problem = transformer.transform(problem)
    state = Statevector(HartreeFock(
        num_spatial_orbitals=reduced_problem.num_spatial_orbitals,
        num_particles=reduced_problem.num_particles,
        qubit_mapper=mapper
        ))

    # 3. Generate the Qubit Hamiltonian
    # We use Jordan-Wigner mapping, which results in 4 qubits for STO-3G
    hamiltonian_op = mapper.map(reduced_problem.hamiltonian.second_q_op())

    # Add Nuclear Repulsion Energy (constant offset usually stored separately)
    # QITE minimizes the electronic part, but to match -1.137, we add this constant.
    nuclear_repulsion = reduced_problem.hamiltonian.nuclear_repulsion_energy
    core_energy = reduced_problem.hamiltonian.constants['ActiveSpaceTransformer']

    hamiltonian_full = hamiltonian_op + SparsePauliOp(["I" * hamiltonian_op.num_qubits], coeffs=[nuclear_repulsion+core_energy])


    return hamiltonian_full, state
