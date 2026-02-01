import numpy as np
import scipy.linalg
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector, partial_trace
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator

# --- 1. Efficient Math Helpers (Cached & Vectorized) ---

_BASIS_CACHE = {}

def get_pauli_basis(num_qubits):
    if num_qubits in _BASIS_CACHE:
        return _BASIS_CACHE[num_qubits]
    labels = ['I', 'X', 'Y', 'Z']
    basis_strings = [''.join(p) for p in product(labels, repeat=num_qubits)]
    basis_ops = [SparsePauliOp(s) for s in basis_strings]
    basis_mats = np.array([op.to_matrix() for op in basis_ops])
    _BASIS_CACHE[num_qubits] = (basis_ops, basis_mats)
    return _BASIS_CACHE[num_qubits]

def get_active_and_domain_qubits(term, total_qubits, radius=1):
    p_str = term.paulis[0].to_label()
    active_indices = []
    # Qiskit string is reversed (q_{n-1}...q_0)
    for i, char in enumerate(reversed(p_str)):
        if char != 'I':
            active_indices.append(i)
    if not active_indices: return [], []
    domain_set = set(active_indices)
    for idx in active_indices:
        for r in range(1, radius + 1):
            if idx - r >= 0: domain_set.add(idx - r)
            if idx + r < total_qubits: domain_set.add(idx + r)
    return active_indices, sorted(list(domain_set))

def solve_vectorized(basis_mats, h_local_mat, rdm_data):
    """Solves S * a = b using vectorized tensor contractions."""
    B = basis_mats
    rho = rdm_data
    
    # S matrix: S_ij = Re[ Tr(rho * B[i] * B[j]) ]
    S_complex = np.einsum('pq,iqr,jrp->ij', rho, B, B, optimize=True)
    S = np.real(S_complex)
    
    # b vector: b_i = -Im[ Tr(rho * [h, B[i]]) ]
    term1 = np.einsum('pq,qr,irp->i', rho, h_local_mat, B, optimize=True)
    term2 = np.einsum('pq,iqr,rp->i', rho, B, h_local_mat, optimize=True)
    b = -(term1 - term2).imag
    
    # Solve (S + reg) * a = b
    S_reg = S + np.eye(len(b)) * 1e-8
    try:
        a_coeffs = scipy.linalg.solve(S_reg, b, assume_a='sym')
    except:
        a_coeffs = np.linalg.lstsq(S_reg, b, rcond=1e-6)[0]
    return a_coeffs

# --- 2. Adaptive Circuit Step ---

def qite_circuit_step_adaptive(qc, H, current_delta_tau, domain_radius=1):
    """
    Appends QITE gates for one time step to 'qc'.
    Returns updated circuit and max_a coefficient.
    """
    num_qubits = qc.num_qubits
    current_psi = Statevector.from_instruction(qc) # Simulator shortcut
    max_a_in_sweep = 0.0
    
    for term in H:
        if abs(term.coeffs[0]) < 1e-10: continue

        # A. Identify Domain
        active_idx, domain_idx = get_active_and_domain_qubits(term, num_qubits, domain_radius)
        if not domain_idx: continue
        domain_size = len(domain_idx)

        # B. Compute RDM
        trace_qubits = [q for q in range(num_qubits) if q not in domain_idx]
        rdm = partial_trace(current_psi, trace_qubits)
        rdm_data = rdm.data

        # C. Construct Local Hamiltonian
        full_p_str = term.paulis[0].to_label()
        op_list = [np.eye(2, dtype=complex)] * domain_size
        global_to_local = {g: l for l, g in enumerate(domain_idx)}
        
        for g_idx in active_idx:
            char = full_p_str[len(full_p_str) - 1 - g_idx]
            if char == 'X': mat = np.array([[0, 1], [1, 0]], dtype=complex)
            elif char == 'Y': mat = np.array([[0, -1j], [1j, 0]], dtype=complex)
            elif char == 'Z': mat = np.array([[1, 0], [0, -1]], dtype=complex)
            else: mat = np.eye(2, dtype=complex)
            op_list[domain_size - 1 - global_to_local[g_idx]] = mat

        h_local_mat = op_list[0]
        for mat in op_list[1:]: h_local_mat = np.kron(h_local_mat, mat)
        h_local_mat *= term.coeffs[0]

        # D. Solve Linear System
        basis_ops, basis_mats = get_pauli_basis(domain_size)
        
        if rdm_data.shape != h_local_mat.shape: continue 

        a_coeffs = solve_vectorized(basis_mats, h_local_mat, rdm_data)
        
        # Track max 'velocity' for adaptive logic
        current_max = np.max(np.abs(a_coeffs))
        if current_max > max_a_in_sweep:
            max_a_in_sweep = current_max

        # E. Construct Unitary
        A_matrix = np.einsum('i,ijk->jk', a_coeffs, basis_mats)
        U_matrix = scipy.linalg.expm(-1j * current_delta_tau * A_matrix)
        
        # F. Append Gate
        u_gate = UnitaryGate(U_matrix, label=f"QITE")
        qc.append(u_gate, qargs=domain_idx)
        
        # Update tracking state for next Trotter term
        current_psi = current_psi.evolve(u_gate, qargs=domain_idx)

    return qc, max_a_in_sweep

# --- 3. Main Adaptive Loop (For Loop Version) ---

def run_circuit_qite_adaptive(H, initial_state, total_time, max_steps=1000,
                              initial_dt=0.1, target_rotation=0.1, max_dt=0.5, min_dt=1e-4,
                              domain_radius=1, filename='default_filename'):
    

    # Initialize Circuit
    qc = QuantumCircuit(initial_state.num_qubits)
    qc.prepare_state(initial_state)
        
    current_dt = initial_dt
    time_elapsed = 0.0
    
    # Store history
    energies = []
    times = []
    dts = []
    max_as = []
    
    # Initial Energy Check
    psi_init = Statevector.from_instruction(qc)
    psi_init = psi_init / np.linalg.norm(psi_init.data)
    current_energy = psi_init.expectation_value(H).real

    energies.append(current_energy)
    times.append(0.0)
    dts.append(initial_dt)

    if filename is not None:
        fout = open(f'out/{filename}.out','w')
        fout.write("QITE simulation \n")
        fout.write(f"{'Iteration':>10} {'dtau':>10} {'E':>10} {'∆E':>10} {'max_a':>10}\n")
        fout.write(f"{'0':>10} {current_dt:>10.4f} {current_energy:>10.6f} {'-':>10} {'-':>10}\n")

    
    print(f"Starting Adaptive Circuit QITE. Target Rot={target_rotation} rad")
    
    # --- The Requested For-Loop ---
    # We iterate up to max_steps, but break internally when total_time is reached.
    pbar = tqdm(range(max_steps), desc="QITE Progress", unit="step")
    
    for step in pbar:
        # 1. Termination Check
        if time_elapsed >= total_time:
            pbar.write("Total evolution time reached.")
            break

        step_accepted = False

        while not step_accepted:
            qc_trial = qc.copy()
            # 2. Cap dt to hit total_time exactly
            if time_elapsed + current_dt > total_time:
                current_dt = total_time - time_elapsed
            
            # 3. Perform QITE Step
            qc_trial, max_a = qite_circuit_step_adaptive(qc_trial, H, current_dt, domain_radius)
        
            # 4. Measure Data
            psi_trial = Statevector.from_instruction(qc_trial)
            psi_trial = psi_trial / np.linalg.norm(psi_trial.data)
            new_energy = psi_trial.expectation_value(H).real
            energy_change = new_energy - current_energy

        
            if energy_change > -1e-5:
                print(f'Energy stalling or increasing after {step} iterations. Halving the time step.')
                current_dt *= 0.5
                if current_dt < min_dt:
                    pbar.write(f"Converged: Step size {current_dt:.2e} below threshold.")
                    return qc, energies, times, dts, max_as
            else:
                step_accepted = True
                qc = qc_trial
                current_energy = new_energy
                time_elapsed += current_dt
                energies.append(current_energy)
                times.append(time_elapsed)
                dts.append(current_dt)
                max_as.append(max_a)

                if filename is not None:
                    fout.write(f'{step+1:>10.0f} {current_dt:>10.4f} {current_energy:>10.6f} {energy_change:>10.6f} {max_a:>10.6f}\n')

                
                # 5. Adaptive Logic for Next dt
                if max_a > 1e-9:
                    suggested_dt = target_rotation / max_a
                else:
                    suggested_dt = max_dt * 1.5 
                    
                # Smoothing & Limits
                suggested_dt = max(min(suggested_dt, current_dt * 2.0), current_dt * 0.5)
                current_dt = min(suggested_dt, max_dt)

                # 6. Update Progress Bar Info
        pbar.set_description(f"T={time_elapsed:.2f}/{total_time} | E={current_energy:.4f} | dt={current_dt:.3f}")
        
    return qc, energies, times, dts, max_as

