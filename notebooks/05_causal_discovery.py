import warnings
from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import networkx as nx

from sklearn.preprocessing import StandardScaler

from causallearn.search.ConstraintBased.PC import pc

warnings.filterwarnings("ignore")

plt.style.use("ggplot")

print("Libraries Imported Successfully!")

# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

FIGURE_DIR = Path("figures/figures_causal_discovery")
REPORT_DIR = Path("reports")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("Output folders created successfully!")

# ============================================================
# SAVE FIGURE FUNCTION
# ============================================================

def save_plot(filename):

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved -> {filename}")

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(
    "data/processed/final_crop_dataset.csv"
)

print("\nDataset Loaded Successfully!")

print(df.shape)

print(df.head())


# ============================================================
# VARIABLES FOR CAUSAL DISCOVERY
# ============================================================

variables = [

    "yield",

    "avg_ssm",
    "avg_rsm",

    "avg_ndvi",
    "avg_fpar",

    "avg_tavg",
    "avg_tmin",
    "avg_tmax",

    "avg_rad",
    "avg_cwb",

    "awc",
    "bulk_density",
    "drainage_class",

    "crop_area_percentage",

    "harvest_year"

]

data = df[variables].copy()

print()

print("Variables Selected")

print("-"*60)

print(data.columns.tolist())

print()

print("Total Variables :", len(variables))

# ============================================================
# STANDARDIZE VARIABLES
# ============================================================

scaler = StandardScaler()

X = scaler.fit_transform(data)

print()

print("Data Standardized Successfully")

print(X.shape)

# ============================================================
# VARIABLE INDEX MAPPING
# ============================================================

mapping = pd.DataFrame({

    "Node_ID": [f"X{i}" for i in range(len(variables))],

    "Variable": variables

})

print(mapping)

mapping.to_csv(

    REPORT_DIR / "variable_mapping.csv",

    index=False

)

print("\nVariable Mapping Saved!")



# ============================================================
# RUN PC ALGORITHM
# ============================================================

print("=" * 70)
print("RUNNING PC ALGORITHM")
print("=" * 70)

causal_graph = pc(

    X,

    alpha=0.05,

    indep_test="fisherz",

    stable=True,

    uc_rule=0,

    verbose=False

)

print("\nPC Algorithm Completed Successfully!")

# ============================================================
# GRAPH INFORMATION
# ============================================================

graph = causal_graph.G

print()

print("=" * 70)
print("GRAPH INFORMATION")
print("=" * 70)

print("Nodes :", len(graph.get_nodes()))
print("Edges :", len(graph.get_graph_edges()))

# ============================================================
# DECODE LEARNED EDGES
# ============================================================

decoded_edges = []

print()

print("=" * 70)
print("LEARNED CAUSAL RELATIONSHIPS")
print("=" * 70)

for edge in graph.get_graph_edges():

    edge_str = str(edge)

    # Replace highest index first to avoid X1 replacing X10
    for i in reversed(range(len(variables))):

        edge_str = edge_str.replace(
            f"X{i+1}",
            variables[i]
        )

        edge_str = edge_str.replace(
            f"X{i}",
            variables[i]
        )

    decoded_edges.append(edge_str)

    print(edge_str)

print()

print("Total Edges :", len(decoded_edges))

# ============================================================
# SAVE EDGE LIST
# ============================================================

edge_df = pd.DataFrame({

    "Edge": decoded_edges

})

edge_df.to_csv(

    REPORT_DIR / "decoded_pc_edges.csv",

    index=False

)

print("Decoded Edge List Saved!")

# ============================================================
# CREATE NETWORKX GRAPH
# ============================================================

G = nx.DiGraph()

G.add_nodes_from(variables)

for edge in decoded_edges:

    if "-->" in edge:

        source, target = edge.split("-->")

        source = source.strip()

        target = target.strip()

        G.add_edge(source, target)

print()

print("Nodes :", G.number_of_nodes())

print("Edges :", G.number_of_edges())

# ============================================================
# NODE CATEGORIES
# ============================================================

node_colors = {}

for node in G.nodes():

    # Target
    if node == "yield":
        node_colors[node] = "#E74C3C"      # Red

    # Soil
    elif node in [
        "avg_ssm",
        "avg_rsm",
        "awc",
        "bulk_density",
        "drainage_class"
    ]:
        node_colors[node] = "#A67C52"      # Brown

    # Vegetation
    elif node in [
        "avg_ndvi",
        "avg_fpar"
    ]:
        node_colors[node] = "#2ECC71"      # Green

    # Weather
    elif node in [
        "avg_tavg",
        "avg_tmin",
        "avg_tmax",
        "avg_rad",
        "avg_cwb"
    ]:
        node_colors[node] = "#F39C12"      # Orange

    # Management / Other
    else:
        node_colors[node] = "#3498DB"      # Blue


# ============================================================
# MANUAL LAYOUT
# ============================================================

pos = {

    # Weather
    "avg_tmin": (-4,3),
    "avg_tavg": (-2,3),
    "avg_tmax": (0,3),
    "avg_rad": (2,3),
    "avg_cwb": (4,3),

    # Soil
    "awc": (-4,1),
    "bulk_density": (-2,1),
    "drainage_class": (0,1),
    "avg_ssm": (2,1),
    "avg_rsm": (4,1),

    # Vegetation
    "avg_ndvi": (-1,-1),
    "avg_fpar": (1,-1),

    # Other
    "crop_area_percentage": (-2,-3),
    "harvest_year": (2,-3),

    # Outcome
    "yield": (0,-5)
}

# ============================================================
# PROFESSIONAL DAG
# ============================================================

plt.figure(figsize=(16,12))

nx.draw_networkx_nodes(

    G,

    pos,

    node_size=2600,

    node_color=[node_colors[n] for n in G.nodes()],

    edgecolors="black",

    linewidths=1.5

)

nx.draw_networkx_labels(

    G,

    pos,

    font_size=9,

    font_weight="bold"

)

nx.draw_networkx_edges(

    G,

    pos,

    arrows=True,

    arrowsize=22,

    arrowstyle="-|>",

    width=1.8,

    alpha=0.8,

    connectionstyle="arc3,rad=0.08"

)

plt.title(

    "Learned Causal Graph (PC Algorithm)",

    fontsize=18,

    weight="bold"

)

plt.axis("off")

save_plot("01_professional_pc_dag.png")

# ============================================================
# GRAPH STATISTICS
# ============================================================

print("="*70)
print("GRAPH STATISTICS")
print("="*70)

print(f"Nodes                 : {G.number_of_nodes()}")
print(f"Edges                 : {G.number_of_edges()}")
print(f"Density               : {nx.density(G):.3f}")
print(f"Is DAG                : {nx.is_directed_acyclic_graph(G)}")
print(f"Weak Components       : {nx.number_weakly_connected_components(G)}")
print(f"Average Degree        : {sum(dict(G.degree()).values())/G.number_of_nodes():.2f}")

# ============================================================
# DEGREE CENTRALITY
# ============================================================

print("="*70)
print("DEGREE CENTRALITY")
print("="*70)

degree_df = pd.DataFrame({

    "Variable": list(G.nodes()),

    "In_Degree": [G.in_degree(n) for n in G.nodes()],

    "Out_Degree": [G.out_degree(n) for n in G.nodes()],

    "Total_Degree": [G.degree(n) for n in G.nodes()]

})

degree_df = degree_df.sort_values(

    by="Total_Degree",

    ascending=False

)

print(degree_df)

degree_df.to_csv(

    REPORT_DIR / "node_degree_statistics.csv",

    index=False

)

print("\nDegree Statistics Saved!")

# ============================================================
# CENTRALITY MEASURES
# ============================================================

print("="*70)
print("NETWORK CENTRALITY")
print("="*70)

centrality = pd.DataFrame({

    "Variable": list(G.nodes()),

    "Degree_Centrality":
        pd.Series(nx.degree_centrality(G)),

    "Betweenness":
        pd.Series(nx.betweenness_centrality(G)),

    "Closeness":
        pd.Series(nx.closeness_centrality(G))

})

centrality = centrality.sort_values(

    "Degree_Centrality",

    ascending=False

)

print(centrality)

centrality.to_csv(

    REPORT_DIR / "centrality_statistics.csv",

    index=False

)

print("\nCentrality Statistics Saved!")

# ============================================================
# DIRECT CAUSES OF YIELD
# ============================================================

parents = list(G.predecessors("yield"))

print("="*70)
print("DIRECT PARENTS OF YIELD")
print("="*70)

if len(parents)==0:

    print("No direct parents discovered.")

else:

    for i,node in enumerate(parents,1):

        print(f"{i}. {node}")

pd.DataFrame({

    "Direct_Cause": parents

}).to_csv(

    REPORT_DIR/"yield_direct_parents.csv",

    index=False

)

# ============================================================
# DIRECT EFFECTS OF YIELD
# ============================================================

children = list(G.successors("yield"))

print("="*70)
print("DIRECT CHILDREN OF YIELD")
print("="*70)

if len(children)==0:

    print("Yield has no outgoing edges.")

else:

    for i,node in enumerate(children,1):

        print(f"{i}. {node}")

# ============================================================
# ROOT NODES
# ============================================================

roots = [

    node

    for node in G.nodes()

    if G.in_degree(node)==0

]

print("="*70)
print("ROOT VARIABLES")
print("="*70)

for node in roots:

    print(node)

# ============================================================
# TERMINAL NODES
# ============================================================

terminal = [

    node

    for node in G.nodes()

    if G.out_degree(node)==0

]

print("="*70)
print("TERMINAL VARIABLES")
print("="*70)

for node in terminal:

    print(node)

# ============================================================
# RUN NOTEARS ALGORITHM
# ============================================================

from castle.algorithms import Notears



print("=" * 70)
print("RUNNING NOTEARS ALGORITHM")
print("=" * 70)

notears = Notears()

notears.learn(X)

print("\nNOTEARS Completed Successfully!")

# ============================================================
# NOTEARS ADJACENCY MATRIX
# ============================================================

notears_adj = pd.DataFrame(

    notears.causal_matrix,

    index=variables,

    columns=variables

)

print("="*70)
print("NOTEARS ADJACENCY MATRIX")
print("="*70)

print(notears_adj.round(3))

notears_adj.to_csv(

    REPORT_DIR / "notears_adjacency_matrix.csv"

)

print("\nAdjacency Matrix Saved!")

# ============================================================
# INSPECT NOTEARS OUTPUT
# ============================================================

print("="*70)
print("RAW CAUSAL MATRIX")
print("="*70)

print(notears.causal_matrix)

print()

print("Unique values in matrix:")

print(np.unique(notears.causal_matrix))

print()

print("Number of non-zero entries:")

print(np.count_nonzero(notears.causal_matrix))

# ============================================================
# DECODE NOTEARS EDGES
# ============================================================

print("="*70)
print("NOTEARS LEARNED RELATIONSHIPS")
print("="*70)

notears_edges = []

adj = notears.causal_matrix

for i in range(len(variables)):
    for j in range(len(variables)):

        if adj[i, j] == 1:

            source = variables[i]
            target = variables[j]

            print(f"{source}  --->  {target}")

            notears_edges.append([source, target])

print()

print("Total Learned Edges :", len(notears_edges))

notears_df = pd.DataFrame(

    notears_edges,

    columns=["Source","Target"]

)

notears_df.to_csv(

    REPORT_DIR / "notears_edges.csv",

    index=False

)

print("NOTEARS Edge List Saved!")


# ============================================================
# NOTEARS DAG VISUALIZATION
# ============================================================

G_notears = nx.DiGraph()

for source, target in notears_edges:
    G_notears.add_edge(source, target)

plt.figure(figsize=(18,12))

pos = nx.spring_layout(
    G_notears,
    seed=42,
    k=2.6
)

# Highlight important nodes
node_colors = []

for node in G_notears.nodes():

    if node == "yield":
        node_colors.append("tomato")

    elif node == "avg_ssm":
        node_colors.append("limegreen")

    else:
        node_colors.append("skyblue")

nx.draw_networkx_nodes(

    G_notears,

    pos,

    node_size=2800,

    node_color=node_colors,

    edgecolors="black",

    linewidths=1.5

)

nx.draw_networkx_labels(

    G_notears,

    pos,

    font_size=10,

    font_weight="bold"

)

nx.draw_networkx_edges(

    G_notears,

    pos,

    arrows=True,

    arrowsize=22,

    width=2,

    edge_color="gray",

    connectionstyle="arc3,rad=0.08"

)

plt.title(

    "NOTEARS Learned Causal Graph",

    fontsize=16,

    weight="bold"

)

plt.axis("off")

save_plot("02_notears_learned_dag.png")


# ============================================================
# NOTEARS GRAPH STATISTICS
# ============================================================

print("="*70)
print("NOTEARS GRAPH STATISTICS")
print("="*70)

print("Nodes :", G_notears.number_of_nodes())
print("Edges :", G_notears.number_of_edges())
print("Density :", round(nx.density(G_notears),3))
print("Average Degree :",
      round(sum(dict(G_notears.degree()).values()) /
            G_notears.number_of_nodes(),2))
print("Is DAG :", nx.is_directed_acyclic_graph(G_notears))
print("Weakly Connected Components :",
      nx.number_weakly_connected_components(G_notears))


# ============================================================
# DAG COMPARISON
# ============================================================

comparison = pd.DataFrame({

    "Criterion":[

        "Method",
        "Number of Nodes",
        "Number of Edges",
        "Graph Type",
        "Based On",
        "Used for Causal Estimation"

    ],

    "Expert DAG":[

        "Domain Knowledge",
        len(variables),
        "User Defined",
        "Directed",
        "Agricultural Expertise",
        "Yes"

    ],

    "PC Algorithm":[

        "Constraint Based",
        len(graph.get_nodes()),
        len(graph.get_graph_edges()),
        "Directed",
        "Conditional Independence",
        "No"

    ],

    "NOTEARS":[

        "Optimization Based",
        G_notears.number_of_nodes(),
        G_notears.number_of_edges(),
        "Directed",
        "Continuous Optimization",
        "No"

    ]

})

print(comparison)

comparison.to_csv(

    REPORT_DIR / "dag_comparison.csv",

    index=False

)

print("\nComparison Table Saved!")


# ============================================================
# INTERPRETATION
# ============================================================

summary = """
============================================================
CAUSAL DISCOVERY SUMMARY
============================================================

1. Expert DAG

The Expert DAG was constructed using established
agricultural knowledge and agronomic literature.
It explicitly represents assumed causal pathways
between climate, soil, vegetation and crop yield.

------------------------------------------------------------

2. PC Algorithm

The PC Algorithm discovered a larger causal graph
using conditional independence testing.
Several statistically inferred relationships were
identified; however, some agronomically expected
relationships were absent or incorrectly oriented.

------------------------------------------------------------

3. NOTEARS

NOTEARS produced a much sparser directed graph.
Several meaningful relationships were recovered,
including

• avg_rad → avg_tmax
• avg_tmax → avg_tavg
• avg_cwb → avg_ssm
• avg_ssm → avg_rsm
• avg_ndvi → avg_fpar

However, important expert relationships such as

avg_ssm → yield

were not recovered.

------------------------------------------------------------

4. Final Decision

For downstream causal effect estimation
(DoWhy, EconML, Double Machine Learning),
the Expert DAG will be used because it is based
on domain knowledge rather than purely statistical
associations.

The learned DAGs are retained for comparison and
validation purposes only.

============================================================
"""

print(summary)

with open(

    REPORT_DIR / "causal_discovery_summary.txt",

    "w",

    encoding="utf-8"

) as f:

    f.write(summary)

print("Summary Saved!")