"""Bibliometric analysis of the QIEA literature for the survey paper (Section 2).

Queries OpenAlex (https://openalex.org, free, no login) for the QIEA literature
and for the broader "evolutionary computation + quantum computing" literature,
then produces:

  figures/bibliometric_trend.png      publications per year, 1996-2026
  figures/bibliometric_clusters.png   keyword co-occurrence clusters (Louvain)
  data/bibliometric_top_cited.csv     10 most-cited QIEA papers
  data/bibliometric_clusters.csv      keywords per cluster
  data/bibliometric_summary.json      every count quoted in the paper

Raw query results are cached in data/openalex_<query>.json.gz together with
the query string and retrieval date, so re-running without --refresh
reproduces the figures exactly. --refresh re-queries OpenAlex (counts will
change as OpenAlex grows).

Usage (from the repo root or paper_survey/):
    python paper_survey/bibliometric_analysis.py [--refresh]
"""

import argparse
import collections
import csv
import datetime
import gzip
import json
import math
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import requests

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"

API = "https://api.openalex.org/works"
YEAR_MIN, YEAR_MAX = 1996, 2026

# OpenAlex title_and_abstract.search syntax: quoted phrases, upper-case
# Boolean operators. Hyphens inside phrases are ignored by OpenAlex, so
# "quantum-inspired" also matches "quantum inspired".
QUERIES = {
    "core": '("quantum-inspired evolutionary" OR "quantum-inspired genetic" '
            'OR "quantum evolutionary algorithm" OR "quantum genetic algorithm" '
            'OR QIEA)',
    "broad": '(("evolutionary computation" OR "evolutionary algorithm" '
             'OR "genetic algorithm") AND ("quantum computing" OR "quantum computer"))',
}

# Local re-check of each query against the title + abstract OpenAlex actually
# returns. OpenAlex matches some records on text it does not expose (e.g.
# withheld publisher abstracts), which lets through off-topic hits such as
# quantum-dot physics papers; a record is kept only if this check passes.
VERIFY = {
    "core": [re.compile(
        r"quantum[\s-]+inspired[\s-]+(evolutionary|genetic)"
        r"|quantum[\s-]+(evolutionary|genetic)[\s-]+algorithm|\bQIEAs?\b", re.I)],
    "broad": [re.compile(r"evolutionary[\s-]+(computation|algorithm)|genetic[\s-]+algorithm", re.I),
              re.compile(r"quantum[\s-]+comput(ing|er)", re.I)],
}

# Broad-query papers about quantum problems (EAs applied *to* quantum computing).
QUANTUM_PROBLEM_RE = re.compile(r"quantum circuit|ansatz|variational|QAOA|gate synthesis|circuit design"
                                r"|quantum algorithm", re.I)
BROAD_PERIODS = [(2005, 2020), (2021, YEAR_MAX)]

MO_RE = re.compile(r"(multi|many|bi|two|three|four|tri)[\s-]*objective|\bpareto", re.I)

# Peer-reviewed document types only: drops preprints, software, datasets,
# and similar (2026 alone has ~60 near-duplicate Zenodo uploads).
KEEP_TYPES = {"article", "conference-paper", "book-chapter", "book", "review", "dissertation"}
DROP_SOURCES = {"Zenodo (CERN European Organization for Nuclear Research)"}

# Known OpenAlex metadata errors, each checked by hand.
# Narayanan & Moore's ICEC paper: OpenAlex says 2002, DOI is icec.1996.542334.
# Han et al.'s parallel QGA: OpenAlex says 2002, DOI is cec.2001.934358.
YEAR_CORRECTIONS = {"https://openalex.org/W1962752774": 1996,
                    "https://openalex.org/W2116400650": 2001}

# Keyword normalization for the co-occurrence network.
KEYWORD_SYNONYMS = {
    "0-1 knapsack problem": "knapsack problem",
    "quantum superposition": "superposition",
    "genetic algorithms": "genetic algorithm",
    "evolutionary algorithms": "evolutionary algorithm",
    "multi-objective optimization problems": "multi-objective optimization",
    "multiobjective optimization": "multi-objective optimization",
    "quantum gates": "quantum rotation gate",
    "rotation gate": "quantum rotation gate",
}
MIN_KEYWORD_COUNT = 20      # keywords rarer than this are left out of the network
LOUVAIN_SEED = 0
LABELS_PER_CLUSTER = 8      # only the most frequent keywords are labelled on the figure
EDGE_DRAW_QUANTILE = 0.5    # only the stronger half of within-cluster links is drawn
LABEL_FONTSIZE = 7.5        # keyword labels, in points as printed
TOP_N_CITED = 10

# Reference categorical palette (dataviz skill, light mode), fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ---------------------------------------------------------------- fetching

def _abstract(work):
    inv = work.get("abstract_inverted_index") or {}
    pos = {i: tok for tok, idx in inv.items() for i in idx}
    return " ".join(pos[i] for i in sorted(pos))


def _trim(work):
    loc = work.get("primary_location") or {}
    return {
        "id": work["id"],
        "doi": work.get("doi"),
        "title": work.get("title") or "",
        "year": work.get("publication_year"),
        "type": work.get("type"),
        "language": work.get("language"),
        "cited_by_count": work.get("cited_by_count", 0),
        "authors": [a["author"]["display_name"] for a in work.get("authorships") or []],
        "source": (loc.get("source") or {}).get("display_name"),
        "abstract": _abstract(work),
        "keywords": [k["display_name"] for k in work.get("keywords") or []],
        "is_retracted": work.get("is_retracted", False),
        "is_paratext": work.get("is_paratext", False),
    }


def fetch(name):
    query = QUERIES[name]
    flt = f"title_and_abstract.search:{query},publication_year:{YEAR_MIN}-{YEAR_MAX}"
    records, cursor = [], "*"
    while cursor:
        r = requests.get(API, params={"filter": flt, "per-page": 200, "cursor": cursor}, timeout=60)
        r.raise_for_status()
        page = r.json()
        if not page["results"]:
            break
        records += [_trim(w) for w in page["results"]]
        cursor = page["meta"].get("next_cursor")
        time.sleep(0.2)
    payload = {"query": query, "filter": flt,
               "retrieved": datetime.date.today().isoformat(), "records": records}
    DATA_DIR.mkdir(exist_ok=True)
    with gzip.open(DATA_DIR / f"openalex_{name}.json.gz", "wt", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"✅ fetched {len(records)} records for '{name}'")
    return payload


def load(name, refresh):
    path = DATA_DIR / f"openalex_{name}.json.gz"
    if refresh or not path.exists():
        return fetch(name)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- cleaning

def _norm_title(t):
    return re.sub(r"[^a-z0-9]", "", t.lower())


def clean(name, payload):
    """Apply the filtering steps in order, recording how many records each removes."""
    recs = payload["records"]
    steps = [("retrieved", len(recs))]
    for r in recs:
        r["year"] = YEAR_CORRECTIONS.get(r["id"], r["year"])

    recs = [r for r in recs if r["type"] in KEEP_TYPES and r["source"] not in DROP_SOURCES
            and not r["is_retracted"] and not r["is_paratext"]]
    steps.append(("peer-reviewed types", len(recs)))
    recs = [r for r in recs if r["language"] in (None, "en")]
    steps.append(("English", len(recs)))
    recs = [r for r in recs if all(p.search(r["title"] + " " + r["abstract"]) for p in VERIFY[name])]
    steps.append(("terms verified in title/abstract", len(recs)))

    # Deduplicate by normalized title, keeping the most-cited version.
    best = {}
    for r in recs:
        k = _norm_title(r["title"])
        if k not in best or r["cited_by_count"] > best[k]["cited_by_count"]:
            best[k] = r
    recs = list(best.values())
    steps.append(("deduplicated", len(recs)))

    for r in recs:
        r["multi_objective"] = bool(MO_RE.search(" ".join([r["title"], r["abstract"]] + r["keywords"])))
    return recs, steps


# ---------------------------------------------------------------- figures

def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_trend(core, broad, retrieved):
    years = list(range(YEAR_MIN, YEAR_MAX + 1))
    so = collections.Counter(r["year"] for r in core if not r["multi_objective"])
    mo = collections.Counter(r["year"] for r in core if r["multi_objective"])
    br = collections.Counter(r["year"] for r in broad)

    # Drawn at the printed size (the paper's text block is ~5in wide), so
    # font sizes below are the sizes that appear on the page.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.0, 4.4), sharex=True, layout="constrained",
                                   gridspec_kw={"height_ratios": [3, 2]})
    fig.get_layout_engine().set(h_pad=0.08)
    so_v = [so[y] for y in years]
    mo_v = [mo[y] for y in years]
    ax1.bar(years, so_v, width=0.8, color=SERIES[0], label="Single-objective", edgecolor="white", linewidth=0.6)
    ax1.bar(years, mo_v, width=0.8, bottom=so_v, color=SERIES[1], label="Multi-objective",
            edgecolor="white", linewidth=0.6)
    ax1.set_title("(a) QIEA publications per year", loc="left", fontsize=9, color=TEXT_PRIMARY)
    ax1.set_ylabel("Publications", fontsize=9, color=TEXT_SECONDARY)
    ax1.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax1)

    ax2.bar(years, [br[y] for y in years], width=0.8, color=SERIES[0], edgecolor="white", linewidth=0.6)
    ax2.set_title("(b) Evolutionary computation + quantum computing",
                  loc="left", fontsize=9, color=TEXT_PRIMARY)
    ax2.set_ylabel("Publications", fontsize=9, color=TEXT_SECONDARY)
    ticks = years[::5]  # 1996, 2001, ..., 2026
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([str(y) for y in ticks[:-1]] + [f"{YEAR_MAX}*"])
    ax1.tick_params(labelbottom=True)  # sharex hides the top panel's year labels otherwise
    ax2.set_xlabel(f"Publication year (*{YEAR_MAX} partial, to {retrieved})", fontsize=9, color=TEXT_SECONDARY)
    _style(ax2)
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "bibliometric_trend.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"📎 {out}")


def keyword_clusters(core):
    def norm(k):
        k = k.lower().strip()
        return KEYWORD_SYNONYMS.get(k, k)

    query_re = VERIFY["core"][0]
    docs = [{norm(k) for k in r["keywords"]} for r in core]
    docs = [{k for k in d if not query_re.search(k)} for d in docs]  # query terms are in every record
    freq = collections.Counter(k for d in docs for k in d)
    vocab = {k for k, c in freq.items() if c >= MIN_KEYWORD_COUNT}

    # Insert nodes in sorted order: Louvain's result depends on node order, and
    # set iteration order changes between runs (string hash randomization).
    G = nx.Graph()
    for k in sorted(vocab):
        G.add_node(k, count=freq[k])
    for d in docs:
        ks = sorted(d & vocab)
        for i, a in enumerate(ks):
            for b in ks[i + 1:]:
                w = G.get_edge_data(a, b, {"weight": 0})["weight"]
                G.add_edge(a, b, weight=w + 1)
    G.remove_nodes_from([n for n in list(G) if G.degree(n) == 0])
    # Run Louvain on integer node ids: networkx keeps nodes in sets internally,
    # and only integers (not strings) iterate in the same order on every run.
    names = sorted(G)
    Gi = nx.relabel_nodes(G, {k: i for i, k in enumerate(names)})
    comms = [{names[i] for i in c}
             for c in nx.community.louvain_communities(Gi, weight="weight", seed=LOUVAIN_SEED)]
    comms = sorted(comms, key=lambda c: (-sum(freq[k] for k in c), sorted(c)))
    return G, comms, freq


def _seg_hits_box(p, q, box, steps=30):
    """True if segment p-q passes through `box` (sampled, good enough at pixel scale)."""
    return any(box.contains(p[0] + (q[0] - p[0]) * i / steps, p[1] + (q[1] - p[1]) * i / steps)
               for i in range(steps + 1))


def _segs_cross(p1, p2, q1, q2):
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return (orient(p1, p2, q1) * orient(p1, p2, q2) < 0
            and orient(q1, q2, p1) * orient(q1, q2, p2) < 0)


def _place_labels(ax, fig, pos, keys, node_size, centre, obstacles=(), bands=None):
    """Label `keys` near their nodes. Tries offsets in 8 directions at growing
    distances and keeps the first spot whose box overlaps neither a placed
    label nor any other node, and whose leader line (drawn once a label moves
    away) crosses no other label, node or leader. Directions pointing away
    from the node's cluster centre are tried first, so labels sit on the
    outside of the cluster. `node_size` maps each node to its scatter size
    (points^2); `centre` maps it to its cluster's centre (data units);
    `obstacles` are other artists (e.g. titles) labels must not cover; with
    `bands` (node -> (y_min, y_max) in data units), a label stays within its
    cluster's row, so it cannot be read as belonging to the next cluster."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_px = ax.transData.transform
    px_per_pt = fig.dpi / 72
    node_boxes = {}
    for k, (x, y) in pos.items():
        px, py = to_px((x, y))
        h = 0.5 * node_size[k] ** 0.5 * px_per_pt
        node_boxes[k] = matplotlib.transforms.Bbox([[px - h, py - h], [px + h, py + h]])
    placed = [o.get_window_extent(renderer).expanded(1.05, 1.2) for o in obstacles]
    leaders = []
    # Labels must stay on the canvas (the figure is saved untrimmed at its
    # printed size), with a small margin for text-hinting differences.
    canvas = fig.bbox.padded(-3 * px_per_pt)
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1),
            (1, 0.5), (-1, 0.5), (1, -0.5), (-1, -0.5), (0.5, 1), (-0.5, 1), (0.5, -1), (-0.5, -1)]
    for k in keys:
        x, y = pos[k]
        own = tuple(to_px((x, y)))
        cpx = to_px(centre[k])
        if bands is not None:
            lo, hi = sorted(to_px((0, yb))[1] for yb in bands[k])
        ox, oy = own[0] - cpx[0], own[1] - cpx[1]
        k_dirs = sorted(dirs, key=lambda d: -(d[0] * ox + d[1] * oy) / (math.hypot(*d) * (math.hypot(ox, oy) or 1)))
        chosen = None
        r0 = 0.5 * node_size[k] ** 0.5 + 2  # start just outside the node's edge
        for r in (r0, r0 + 8, r0 + 18, r0 + 30, r0 + 45, r0 + 60, r0 + 80, r0 + 100, r0 + 125):
            for dx, dy in k_dirs:
                ha = "center" if abs(dx) < 1 else ("left" if dx > 0 else "right")
                va = "center" if abs(dy) < 1 else ("bottom" if dy > 0 else "top")
                t = ax.annotate(k, (x, y), xytext=(dx * r, dy * r), textcoords="offset points",
                                ha=ha, va=va, fontsize=LABEL_FONTSIZE, color=TEXT_PRIMARY,
                                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
                box = t.get_window_extent(renderer).expanded(1.03, 1.15)
                ok = ((bands is None or (lo <= box.y0 and box.y1 <= hi))
                      and canvas.x0 <= box.x0 and box.x1 <= canvas.x1 and canvas.y0 <= box.y0 and box.y1 <= canvas.y1
                      and not any(box.overlaps(b) for n, b in node_boxes.items() if n != k)
                      and not any(box.overlaps(b) for b in placed)
                      and not any(_seg_hits_box(a, b, box) for a, b in leaders))
                seg = None
                if ok and r > r0:  # this spot needs a leader line: check it too
                    seg = (own, (own[0] + dx * r * px_per_pt, own[1] + dy * r * px_per_pt))
                    ok = (not any(_seg_hits_box(*seg, b) for b in placed)
                          and not any(_seg_hits_box(*seg, b) for n, b in node_boxes.items() if n != k)
                          and not any(_segs_cross(*seg, *l) for l in leaders))
                if ok:
                    chosen = (t, box, r)
                    if seg:
                        leaders.append(seg)
                    break
                t.remove()
            if chosen:
                break
        if chosen is None:  # fall back to directly above, accepting overlap
            t = ax.annotate(k, (x, y), xytext=(0, 9), textcoords="offset points", ha="center",
                            fontsize=LABEL_FONTSIZE, color=TEXT_PRIMARY,
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
            chosen = (t, t.get_window_extent(renderer), 9)
            print(f"⚠️ label '{k}' could not avoid overlaps")
        t, box, r = chosen
        if r > r0:
            t.arrow_patch = None
            ax.annotate("", (x, y), xytext=t.get_position(), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", color=TEXT_SECONDARY, lw=0.5))
        placed.append(box)


def _ordered_subgraph(G, nodes):
    """Copy of G restricted to `nodes`, built in sorted order (G.subgraph()
    iterates in set order, which would make the layout vary between runs)."""
    nodes = sorted(nodes)
    sub = nx.Graph()
    sub.add_nodes_from(nodes)
    sub.add_weighted_edges_from(sorted((a, b, G[a][b]["weight"]) for a in nodes for b in G[a]
                                       if b in sub and a < b))
    return sub


def plot_clusters(G, comms, freq):
    """One row per cluster, stacked to fill a float page at the full text
    width; within-cluster layout by co-occurrence. Links between clusters are
    left out of the drawing (counted in the summary)."""
    sx = 1.5   # horizontal stretch of each cluster's layout (spans ~[-1, 1]): wide rows suit horizontal labels
    gap = 0.6  # vertical space between rows, in layout units
    # Each row's height grows with its cluster's size, so the densest cluster
    # gets room for its labels' leader lines.
    sy = [max(1.0, len(c) / 16) for c in comms]
    cy = [0.0]  # row i is centred on cy[i] and spans cy[i] +/- sy[i]
    for i in range(1, len(comms)):
        cy.append(cy[-1] - sy[i - 1] - gap - sy[i])
    # Drawn at the printed size (the paper's text block is ~5in wide), so
    # font and node sizes here are the sizes that appear on the page.
    fig, ax = plt.subplots(figsize=(5.0, 6.85), dpi=200)  # place labels at the saved resolution
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    pos = {}
    for i, c in enumerate(comms):
        sub = _ordered_subgraph(G, c)
        p = nx.kamada_kawai_layout(sub, weight=None)  # unweighted: spaces nodes evenly in dense clusters
        pos.update({k: (sx * x, cy[i] + sy[i] * y) for k, (x, y) in p.items()})
    node_size = {k: 12 + 1.2 * freq[k] for k in pos}
    centre = {k: (0, cy[i]) for i, c in enumerate(comms) for k in c}
    bands = {k: (cy[i] - sy[i] - gap / 2, cy[i] + sy[i] + gap / 2) for i, c in enumerate(comms) for k in c}
    titles = []
    for i, c in enumerate(comms):
        sub = _ordered_subgraph(G, c)
        weights = sorted(d["weight"] for _, _, d in sub.edges(data=True))
        cut = weights[int(EDGE_DRAW_QUANTILE * (len(weights) - 1))] if weights else 0
        strong = [(a, b) for a, b, d in sub.edges(data=True) if d["weight"] >= cut]
        nx.draw_networkx_edges(sub, pos, edgelist=strong, ax=ax, edge_color=GRID, width=0.7)
        nodes = sorted(c)
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, ax=ax, node_color=SERIES[i % len(SERIES)],
                               node_size=[node_size[k] for k in nodes], edgecolors="white",
                               linewidths=0.7, label=f"Cluster {i + 1}")
        titles.append(ax.text(-2.95, cy[i] + sy[i] + 0.25, f"Cluster {i + 1}", fontsize=9, color=TEXT_PRIMARY,
                              fontweight="bold", va="top"))
    ax.set_xlim(-3.0, 2.9)
    ax.set_ylim(cy[-1] - sy[-1] - 0.25, cy[0] + sy[0] + 0.3)
    _place_labels(ax, fig, pos, [k for c in comms for k in sorted(c, key=lambda k: (-freq[k], k))[:LABELS_PER_CLUSTER]],
                  node_size, centre, titles, bands)
    ax.set_axis_off()
    out = FIG_DIR / "bibliometric_clusters.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"📎 {out}")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="re-query OpenAlex instead of using the cache")
    args = ap.parse_args()

    payloads = {n: load(n, args.refresh) for n in QUERIES}
    cleaned, steps = {}, {}
    for n, p in payloads.items():
        cleaned[n], steps[n] = clean(n, p)
        print(f"{n}: " + " -> ".join(f"{s} {c}" for s, c in steps[n]))

    core, broad = cleaned["core"], cleaned["broad"]
    plot_trend(core, broad, payloads["core"]["retrieved"])

    G, comms, freq = keyword_clusters(core)
    if len(comms) > len(SERIES):
        print(f"❌ {len(comms)} clusters exceed the {len(SERIES)}-colour palette; raise MIN_KEYWORD_COUNT")
    plot_clusters(G, comms, freq)
    with open(DATA_DIR / "bibliometric_clusters.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "keyword", "count"])
        for i, c in enumerate(comms):
            for k in sorted(c, key=lambda k: (-freq[k], k)):
                w.writerow([i + 1, k, freq[k]])

    top = sorted(core, key=lambda r: -r["cited_by_count"])[:TOP_N_CITED]
    with open(DATA_DIR / "bibliometric_top_cited.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "cited_by_count", "year", "title", "authors", "source", "doi",
                    "multi_objective", "openalex_id"])
        for i, r in enumerate(top, 1):
            w.writerow([i, r["cited_by_count"], r["year"], r["title"], "; ".join(r["authors"]),
                        r["source"], r["doi"], r["multi_objective"], r["id"]])

    years = [r["year"] for r in core]
    core_ids = {r["id"] for r in core}
    broad_periods = {}
    for a, b in BROAD_PERIODS:
        sel = [r for r in broad if a <= r["year"] <= b]
        broad_periods[f"{a}-{b}"] = {
            "papers": len(sel),
            "also_in_qiea_corpus": sum(r["id"] in core_ids for r in sel),
            "quantum_problem": sum(bool(QUANTUM_PROBLEM_RE.search(r["title"] + " " + r["abstract"])) for r in sel),
        }
    summary = {
        "retrieved": {n: p["retrieved"] for n, p in payloads.items()},
        "queries": {n: p["filter"] for n, p in payloads.items()},
        "filter_steps": steps,
        "core_total": len(core),
        "core_multi_objective": sum(r["multi_objective"] for r in core),
        "broad_total": len(broad),
        "core_first_year": min(years),
        "core_per_year": dict(sorted(collections.Counter(years).items())),
        "core_mo_per_year": dict(sorted(collections.Counter(r["year"] for r in core if r["multi_objective"]).items())),
        "broad_periods": broad_periods,
        "broad_per_year": dict(sorted(collections.Counter(r["year"] for r in broad).items())),
        "network": {"keywords": G.number_of_nodes(), "edges": G.number_of_edges(),
                    "clusters": len(comms), "min_keyword_count": MIN_KEYWORD_COUNT},
        "clusters": [sorted(c, key=lambda k: (-freq[k], k)) for c in comms],
        "top_cited": [{"title": r["title"], "year": r["year"], "cites": r["cited_by_count"],
                       "multi_objective": r["multi_objective"]} for r in top],
    }
    with open(DATA_DIR / "bibliometric_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ core {len(core)} ({summary['core_multi_objective']} multi-objective), "
          f"broad {len(broad)}, {len(comms)} keyword clusters")


if __name__ == "__main__":
    main()
