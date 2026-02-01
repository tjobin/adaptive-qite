import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from qiskit.quantum_info import entropy
from qiskit.quantum_info import partial_trace, DensityMatrix, Statevector
import numpy as np
import matplotlib.ticker as ticker


plt.rcParams.update({
    'font.size': 12,          # General font size
    'axes.labelsize': 12,     # X and Y labels
    'axes.titlesize': 12,     # Title
    'xtick.labelsize': 12,    # X-axis tick labels
    'ytick.labelsize': 12,    # Y-axis tick labels
    'legend.fontsize': 12,    # Legend
})
colors=[mcolors.TABLEAU_COLORS['tab:blue'],
        mcolors.TABLEAU_COLORS['tab:orange'],
        mcolors.TABLEAU_COLORS['tab:green'],
        mcolors.TABLEAU_COLORS['tab:red'],
        mcolors.TABLEAU_COLORS['tab:purple'],
        mcolors.TABLEAU_COLORS['tab:brown']]

def make_energy_vs_depth_plot(
        qcs_per_type,
        converged_energies_per_type,
        dts,
        fci_energy,
        labels=['Fixed QITE', 'Adaptive QITE'],
        markers=['o', 'x'],
        filename='figs/default_filename.pdf'):
    
    dt_color_map = dict(zip(dts, colors))

    plt.figure()
    marker_handles = [
    Line2D([0], [0], marker='o', markersize=8, color='w', label='Fixed QITE', linewidth=0,
           markerfacecolor='black'),
    Line2D([0], [0], marker='x', markersize=8, color='k', label='Adaptive QITE', linestyle=None, linewidth=0,
           markeredgewidth=1.5,
           markerfacecolor='black')
    ]
    color_handles = [
    Line2D([0], [0], marker='*', markersize=14, color='w', label=rf'(Initial) $dt$ = {dt}',
            markerfacecolor=c) for dt, c in dt_color_map.items()
    ]
    all_handles = marker_handles + color_handles

    plt.xlabel('Circuit depth')
    plt.ylabel('Converged energy [Ha]')
    plt.axhline(y = fci_energy, color='k', linestyle='--', label = f'Exact FCI energy at {fci_energy:.5f} Ha')

    for i, (qcs_type, converged_energies) in enumerate(zip(qcs_per_type, converged_energies_per_type)):
        depths = [qc.depth() for qc in qcs_type]
        for j, (depth, converged_energy) in enumerate(zip(depths, converged_energies)):
            plt.plot(depth, converged_energy, markersize=8, markeredgewidth=1.5, label=labels[i], marker=markers[i], color=colors[j], linestyle=None, alpha=0.8)
    plt.legend(handles = all_handles)
    plt.tight_layout()
    plt.savefig('figs/' + filename + '.pdf')

def make_convergence_plot(qc, times, energies, fci_energy, filename):
    plt.figure()
    plt.axhline(y=energies[-1], color=mcolors.TABLEAU_COLORS['tab:blue'], label=f'Converged energy at {energies[-1]:.5f} Ha')
    plt.axhline(y=fci_energy, color='k', label=f'Exact FCI energy at {fci_energy:.5f} Ha')
    plt.plot(times, energies, markersize=8, label=f'Converged at {energies[-1]:.5f} Ha in {len(energies)} steps (circuit depth : {qc.depth()})', alpha=0.5, linestyle=None, marker='o', linewidth=0)
    plt.xlabel(r'Time ($t_h$)')
    plt.ylabel('Energy (Ha)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'figs/{filename}.pdf')

def make_pes_plot(
        distances,
        energies_per_type,
        timesteps_required,
        labels = ['AQITE energy', 'Exact FCI'],
        markers=['o', '^'],
        filename='default_filename.pdf'
                  ):
    fig, ax1 = plt.subplots()

    # 2. Create the second axis (Time Steps) sharing the same X-axis
    ax2 = ax1.twinx()
    ax1.set_xlabel('Bond distance [Å]')
    ax1.set_ylabel('Energy')
    ax2.set_ylabel('Time steps')
    ax1.tick_params(axis='y')
    ax2.tick_params(axis='y')
    for i, energies in enumerate(energies_per_type):
        if i == 0:
            ax1.plot(distances, energies, label=labels[i], marker=markers[i], alpha=0.7, markersize=8, markeredgewidth=1.5,linestyle=None, linewidth=0, color=mcolors.TABLEAU_COLORS['tab:blue'])
            ax2.plot(distances,  timesteps_required, label='Time steps', marker='^', alpha=0.7, markersize=8, markeredgewidth=1.5,linestyle=None, linewidth=0, color=mcolors.TABLEAU_COLORS['tab:green'])
        else:
            ax1.plot(distances, energies, label=labels[i], marker=markers[i], alpha=0.7, markersize=0, markeredgewidth=0,linestyle='--', color='k')


    # Combine them into one legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles, labels = [handles1[0]] + handles2 + [handles1[1]], [labels1[0]] + labels2 + [labels1[1]]
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.05))    
    fig.tight_layout()
    plt.savefig(f'figs/{filename}.pdf', bbox_inches='tight')
    plt.close(fig)

def make_entropy_plot(
        distances,
        qcs,
        filename
):
    states = [Statevector(qc) for qc in qcs]
    rhos_q0 = [partial_trace(state, [1,3]) for state in states]
    entropies = [entropy(rho) for rho in rhos_q0]
    fig, ax = plt.subplots()
    ax.plot(distances, entropies, marker='o', label='AQITE entanglement entropy', alpha=0.7, markersize=8, markeredgewidth=1.5,linestyle='-.', color='sandybrown')
    # ax.axhline(0.0313, color='k', label='Exact, He')
    ax.set_ylabel('Entanglement entropy')
    ax.set_xlabel('Bond distance')
    ax.legend(loc='best')
    fig.tight_layout()
    plt.savefig(f'figs/{filename}.pdf')
    plt.close(fig)

def make_energy_dt_vs_iter_plot(
        energies_per_initdt,
        dts_per_initdt,
        init_dts,
        fci_energy,
        filename
):
    fig = plt.figure(figsize=(2 * len(dts_per_initdt), 4))
    gs = fig.add_gridspec(1, 4, wspace=0)
    (ax1, ax3, ax5, ax7) = gs.subplots(sharex=False,sharey=True)
    ax2 = ax1.twinx()
    ax4 = ax3.twinx()
    ax6 = ax5.twinx()
    ax8 = ax7.twinx()

    prim_ax = [ax1,ax3,ax5, ax7]
    second_ax = [ax2, ax4, ax6, ax8]
    for i, ax in enumerate(prim_ax):
        ax.set_title(r'$\Delta\tau^{(0)}$' + rf' = $10^{{{-i-1}}}$')
        ax2_p = second_ax[i]
        if i == 0:
            ax2_p.set_ylim(bottom=-0.05, top=0.6)
        ax.set_xlabel('Time steps', color='k')
        ax.set_ylabel('Energy [Ha]', color='tab:blue')
        ax2_p.set_ylabel(r'$^\Delta\tau^{(i)}$' + r' [$\hbar$/Ha]', color='tab:orange')
        ax.tick_params(axis='y', labelcolor='tab:blue')
        ax2_p.tick_params(axis='y', labelcolor='tab:orange')
        iters = np.arange(len(energies_per_initdt[i][0]))
        print(iters)
        
        ax.plot(iters, energies_per_initdt[i][0], marker='o', alpha=0.7, markersize=8, linestyle=None, linewidth=0, color=colors[0])
        ax2_p.plot(iters,  dts_per_initdt[i][0], marker='x', alpha=0.7, markersize=8, markeredgewidth=1.5,linestyle=None, linewidth=0, color=colors[1])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))
        if i == 0:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        else:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))
        # elif i==2:
        #     ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
        #     ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
         
        ax.axhline(fci_energy, linestyle='--', color='k', label='Exact energy')
        # h1, l1 = ax.get_legend_handles_labels()
        # # Plot on axis 2
        # h2, l2 = ax2.get_legend_handles_labels()
        # ax.legend(h1 + h2, l1 + l2, loc='lower right')

    ax4.sharey(ax2)
    ax6.sharey(ax4)
    ax8.sharey(ax6)
    # ax2.sharey(ax6)
    fig.tight_layout()
    plt.savefig(f'figs/{filename}.pdf')
    plt.close(fig)

def make_energy_dt_vs_iter_plot_general(
        energies_per_initdt,
        dts_per_initdt,
        init_dts,
        fci_energy,
        filename
):
    num_dts = len(init_dts)  # Determine the number of initial dts
    num_rows = 2  # Always use 2 rows
    num_cols = (num_dts + num_rows - 1) // num_rows  # Calculate columns needed

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 2.5 * num_rows), sharex=False, sharey=True)
    axes = axes.flatten()  # Flatten axes for easier indexing

    prim_ax = axes[:num_dts]
    second_ax = [ax.twinx() for ax in prim_ax]

    for i, ax in enumerate(prim_ax):
        iters = np.arange(len(energies_per_initdt[i]))

        ax.set_title(r'$\Delta\tau^{(0)}$' + rf' = {init_dts[i]}')
        ax2_p = second_ax[i]
        ax.plot(iters, energies_per_initdt[i], label='Energy', marker='o', alpha=0.7, markersize=8, linestyle=None, linewidth=0, color='tab:blue')
        ax.axhline(fci_energy, color='k', linestyle='--', label='FCI Energy')

        ax2_p.plot(iters, dts_per_initdt[i], label='Time intervals', color='tab:orange', marker='^', alpha=0.7, markersize=8, linestyle=None, linewidth=0)
        
        # X labels only on bottom plots
        if i >= num_cols:
            ax.set_xlabel('Iterations')
        else:
            ax.set_xlabel('')
        
        # Y1 labels only on left plots
        if i % num_cols == 0:
            ax.set_ylabel('Energy')
        else:
            ax.set_ylabel('')
        
        # Y2 labels only on right plots
        if i % num_cols == num_cols - 1:
            ax2_p.set_ylabel('Time interval')
        else:
            ax2_p.set_ylabel('')

    # Hide unused subplots
    for j in range(num_dts, len(axes)):
        axes[j].axis('off')

    min_dt = min(min(dts) for dts in dts_per_initdt)
    max_dt = max(max(dts) for dts in dts_per_initdt)
    for ax2_p in second_ax:
        ax2_p.set_ylim(min_dt - 0.02, max_dt * 1.05)

    handles1, labels1 = axes[-1].get_legend_handles_labels() if len(init_dts) > 1 else axes.get_legend_handles_labels()
    handles2, labels2 = second_ax[-1].get_legend_handles_labels() if len(init_dts) > 1 else second_ax.get_legend_handles_labels()
    handles, labels = [handles1[0]] + handles2 + [handles1[1]], [labels1[0]] + labels2 + [labels1[1]]
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.05))
    fig.subplots_adjust(bottom=0.15, hspace=0.4)
    plt.savefig(f'figs/{filename}.pdf', bbox_inches='tight')
    plt.close(fig)