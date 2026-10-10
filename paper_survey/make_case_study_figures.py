"""Case-study figures for the survey paper (Section 7) that are drawn from the
pipeline's own code rather than from aggregated results:

  figures/interaction_graph_weak_random_8q.png   qubit-interaction graph,
                                                 nodes coloured by Louvain block

The circuit and partition are rebuilt exactly as run_experiment.py builds them
for `--circuit weak_random --n-qubits 8 --seed 0` (its default circuit
parameters, global random/np.random seeded first, since louvain_partition has
no seed of its own). If that run exists under runs/, its recorded `blocks`
are checked against the recomputed partition.

The figures generated from results_master.csv come from
generate_report.py --paper-figures-dir, and the Section 2 figures from
bibliometric_analysis.py; Figure 3 (QFT circuit) is quantikz source in
figures/qft_example.tex.

Usage (from the repo root or paper_survey/):
    python paper_survey/make_case_study_figures.py
"""

import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from M1_finale.final_m1_script import (  # noqa: E402
    build_interaction_graph,
    louvain_partition,
    random_weakly_connected_circuit,
)

FIG_DIR = HERE / "figures"
N_QUBITS, SEED = 8, 0
# run_experiment.py's defaults for the weak_random generator
DEPTH, TWOQ_GATES_TOTAL, CONNECTIVITY_EDGES = 20, 8, 5
# Order of the qubits around the circle: each block's qubits are adjacent and
# blocks linked by an inter-block edge sit next to each other, so no edges
# cross. Specific to this instance's partition {0,2,4} {6,7} {3,5} {1}.
NODE_ORDER = [2, 0, 4, 7, 6, 5, 3, 1]
RECORDED_RUN = REPO / "runs" / f"weak_random_{N_QUBITS}q_stochastic_nsga2_point_las0_g100_p100_seed{SEED}"

# Same palette as bibliometric_analysis.py, so the paper's figures match.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TEXT_PRIMARY, TEXT_SECONDARY = "#0b0b0b", "#52514e"
INTER_DASH = (0, (2.0, 1.6))


def plot_interaction_graph():
    random.seed(SEED)
    np.random.seed(SEED)
    qc = random_weakly_connected_circuit(n_qubits=N_QUBITS, depth=DEPTH, twoq_gates_total=TWOQ_GATES_TOTAL,
                                         connectivity_edges=CONNECTIVITY_EDGES, seed=SEED)
    G = build_interaction_graph(qc)
    blocks = sorted((sorted(b) for b in louvain_partition(qc)), key=lambda b: (-len(b), b))

    metrics = RECORDED_RUN / "metrics.json"
    if metrics.exists():
        recorded = sorted((sorted(b) for b in json.loads(metrics.read_text())["blocks"]),
                          key=lambda b: (-len(b), b))
        if recorded != blocks:
            print(f"❌ recomputed blocks {blocks} differ from {metrics}'s {recorded}")
        else:
            print(f"✅ blocks match {RECORDED_RUN.name}: {blocks}")
    else:
        print(f"⚠️ {RECORDED_RUN.name} not found; blocks not cross-checked")

    # Qubits on a circle, grouped by block: every edge stays visible, and no
    # space is lost to a spring layout pushing disconnected components apart.
    assert sorted(NODE_ORDER) == sorted(G.nodes), "NODE_ORDER must list every qubit once"
    pos = nx.circular_layout(NODE_ORDER)
    block_of = {q: i for i, b in enumerate(blocks) for q in b}

    # Drawn at the printed size (~4.2in of the paper's ~5in text block).
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    weights = nx.get_edge_attributes(G, "weight")
    for inter in (False, True):  # inter-block edges dashed: the gates stage (5) re-injects
        edges = [e for e in G.edges if (block_of[e[0]] != block_of[e[1]]) == inter]
        nx.draw_networkx_edges(G, pos, edgelist=edges, ax=ax, edge_color=TEXT_SECONDARY,
                               width=[0.9 + 0.9 * (weights[e] - 1) for e in edges],
                               style=INTER_DASH if inter else "-")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weights, ax=ax, font_size=8, font_color=TEXT_PRIMARY,
                                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
    for i, b in enumerate(blocks):
        nx.draw_networkx_nodes(G, pos, nodelist=b, ax=ax, node_color=SERIES[i], node_size=330,
                               edgecolors="white", linewidths=1.0,
                               label="Block {" + ", ".join(map(str, b)) + "}")
    nx.draw_networkx_labels(G, pos, labels={q: f"$q_{{{q}}}$" for q in G.nodes}, ax=ax,
                            font_size=8.5, font_color="white")
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=TEXT_SECONDARY, lw=1.8, linestyle=INTER_DASH, label="Inter-block gates"))
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=8,
              labelcolor=TEXT_PRIMARY, markerscale=0.6, handletextpad=0.4, labelspacing=0.9,
              title="Louvain blocks", title_fontsize=8)
    ax.set_aspect("equal")
    ax.margins(0.12)
    ax.set_axis_off()
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / f"interaction_graph_weak_random_{N_QUBITS}q.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"📎 {out}")


if __name__ == "__main__":
    plot_interaction_graph()
