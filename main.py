from adaptive_circuit_qite import run_circuit_qite_adaptive
from normal_circuit_qite import run_circuit_qite
from _plots import (
    make_energy_vs_depth_plot,
    make_convergence_plot,
    make_pes_plot,
    make_entropy_plot,
    make_energy_dt_vs_iter_plot_general
)
from utils import get_exact_fci_energy, make_geometry
from qiskit_nature.second_q.mappers import JordanWignerMapper, BravyiKitaevSuperFastMapper, BravyiKitaevMapper
from build_hamiltonian import build_hamiltonian_and_state
import numpy as np

molecule = 'H2'
geometry= make_geometry(molecule)
basis_set =  'sto-3g'
charge = 0
spin = 0
active_orb = 2
n_elec = 2
domain_size = 4
mapper = JordanWignerMapper()



## Fixed params for the optimization
total_time = 50
initial_dt = 0.05
max_theta = 0.05
min_dt = 0.0001
max_dt = 0.5
nsteps = int(total_time // initial_dt)



## TIME, ENERGY VS ITER

ham, state = build_hamiltonian_and_state(
        geometry,
        basis_set,
        n_elec,
        active_orb,
        mapper
    )
qc, energies_aqite, times, _, _ = run_circuit_qite_adaptive(
    ham,
    state,
    total_time,
    max_steps=nsteps,
    initial_dt=initial_dt,
    target_rotation=max_theta,
    max_dt=max_dt,
    min_dt=min_dt,
    domain_radius=domain_size,
    filename=f'{molecule}_out_init-dt={initial_dt}_A'
    )

# BOND-DISTANCE
no_type = 2
distances = np.arange(0.4, 3.5,0.1)
energies_per_type = [[] for _ in range(no_type)]
timesteps_required = []
qcs = []
for distance in distances:
    geometry = f'H 0 0 0; H 0 0 {distance}'
    ham, state = build_hamiltonian_and_state(
        geometry,
        basis_set,
        n_elec,
        active_orb,
        mapper
    )
    qc, energies_aqite, times, _, _ = run_circuit_qite_adaptive(
        ham,
        state,
        total_time,
        max_steps=nsteps,
        initial_dt=initial_dt,
        target_rotation=max_theta,
        max_dt=max_dt,
        min_dt=min_dt,
        domain_radius=domain_size,
        filename=f'{molecule}_out_init-dt={initial_dt}_A'
        )
    timesteps_required.append(len(energies_aqite))
    energy_aqite = energies_aqite[-1]
    energy_fci = get_exact_fci_energy(ham)
    energies_per_type[0].append(energy_aqite)
    energies_per_type[1].append(energy_fci)
    qcs.append(qc)
make_pes_plot(distances, energies_per_type, timesteps_required, filename='H2_pes_bis')
make_entropy_plot(distances, qcs, filename='H2_entropy_vs_distance_bis')



# INITIAL DTS 

# ham, state = build_hamiltonian_and_state(
#     geometry,
#     basis_set,
#     n_elec,
#     active_orb,
#     mapper
# )

# fci_energy = get_exact_fci_energy(ham)

# initial_dts = [0.1, 0.01, 0.001, 0.0001]
# types = len(initial_dts)
# qcs_per_type = [[] for _ in range(types)]
# energies_per_type = [[] for _ in range(types)]
# dts_per_type = [[] for _ in range(types)]
# labels = ['Fixed QITE', 'Adaptive QITE']
# for i, initial_dt in enumerate(initial_dts):

#     # Fixed QITE
#     # qc, energies, times = run_circuit_qite(
#     #     ham,
#     #     state,
#     #     total_time,
#     #     initial_dt,
#     #     domain_size,
#     #     filename=f'{molecule}_out_init-dt={initial_dt}'
#     # )
#     # qcs_per_type[0].append(qc)
#     # converged_energies_per_type[0].append(energies[-1])
#     # make_convergence_plot(qc, times, energies, fci_energy, f'{molecule}_E_vs_time_init-dt={initial_dt}')

#     # Adaptive QITE 
#     qc, energies, times, dts, max_as = run_circuit_qite_adaptive(
#         ham,
#         state,
#         total_time,
#         max_steps=nsteps,
#         initial_dt=initial_dt,
#         target_rotation=max_theta,
#         max_dt=max_dt,
#         min_dt=min_dt,
#         domain_radius=domain_size,
#         filename=f'{molecule}_out_init-dt={initial_dt}_A'
#         )
#     qcs_per_type[i] = qc
#     energies_per_type[i] = energies
#     dts_per_type[i] = dts
# make_energy_dt_vs_iter_plot_general(energies_per_type, dts_per_type, init_dts=initial_dts, fci_energy=fci_energy, filename='H2_energy_dt_vs_iter_bis')
# # make_energy_vs_depth_plot(qcs_per_type, converged_energies_per_type, dts = initial_dts, fci_energy=fci_energy, filename = f'{molecule}_Energy_vs_Depth.pdf')

