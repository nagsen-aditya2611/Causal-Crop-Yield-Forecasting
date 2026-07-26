# ============================================================
# NOTEBOOK 04
# Expert Causal DAG Construction
# ============================================================

import warnings
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from graphviz import Digraph

warnings.filterwarnings("ignore")

plt.style.use("ggplot")

print("="*60)
print("EXPERT CAUSAL DAG")
print("="*60)
print("Libraries Imported Successfully!")
# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

FIGURE_DIR = Path("figures/figures_expert_dag")
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

print("Shape :", df.shape)

print(df.head())


# ============================================================
# DATASET INFORMATION
# ============================================================

print("="*60)
print("DATASET INFORMATION")
print("="*60)

print()

print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])

print()

print("Column Names")

for col in df.columns:
    print("-", col)

# ============================================================
# VARIABLE CATEGORIZATION
# ============================================================

GEOGRAPHY = [
    "latitude"
]

CLIMATE = [
    "avg_tavg",
    "avg_rad",
    "avg_cwb"
]

SOIL = [
    "awc",
    "bulk_density",
    "drainage_class"
]

TREATMENT = [
    "avg_ssm"
]

MEDIATORS = [
    "avg_ndvi",
    "avg_fpar"
]

TIME = [
    "harvest_year"
]

OUTCOME = [
    "yield"
]

print("="*60)
print("VARIABLE GROUPS")
print("="*60)

print("\nGeography")
print(GEOGRAPHY)

print("\nClimate")
print(CLIMATE)

print("\nSoil")
print(SOIL)

print("\nTreatment")
print(TREATMENT)

print("\nMediators")
print(MEDIATORS)

print("\nTime")
print(TIME)

print("\nOutcome")
print(OUTCOME)

# ============================================================
# SCIENTIFIC ASSUMPTIONS
# ============================================================

assumptions = pd.DataFrame({

    "Cause": [

        "Latitude",
        "Latitude",

        "Harvest Year",
        "Harvest Year",

        "Average Temperature",
        "Average Temperature",
        "Average Temperature",

        "Solar Radiation",
        "Solar Radiation",
        "Solar Radiation",

        "Climate Water Balance",

        "Available Water Capacity",
        "Bulk Density",
        "Drainage Class",

        "Soil Moisture",
        "Soil Moisture",
        "Soil Moisture",

        "NDVI",
        "FPAR"

    ],

    "Effect":[

        "Average Temperature",
        "Solar Radiation",

        "Average Temperature",
        "Climate Water Balance",

        "Climate Water Balance",
        "Soil Moisture",
        "NDVI",

        "NDVI",
        "FPAR",
        "Soil Moisture",

        "Soil Moisture",

        "Soil Moisture",
        "Soil Moisture",
        "Soil Moisture",

        "NDVI",
        "FPAR",
        "Yield",

        "Yield",
        "Yield"

    ],

    "Scientific Justification":[

        "Latitude influences regional climatic conditions and temperature.",

        "Latitude determines incoming solar energy received by crops.",

        "Climate varies across years due to interannual weather variability.",

        "Annual rainfall and water balance change over years.",

        "Temperature affects evapotranspiration and water demand.",

        "Temperature influences seasonal soil moisture through evaporation.",

        "Crop growth depends strongly on temperature.",

        "Solar radiation drives photosynthesis.",

        "FPAR depends on incoming radiation absorbed by vegetation.",

        "Radiation influences soil drying and evaporation.",

        "Climate water balance determines available soil water.",

        "High available water capacity improves soil moisture retention.",

        "Dense soils alter infiltration and water storage.",

        "Drainage controls water loss from soil.",

        "Higher soil moisture promotes vegetation growth (NDVI).",

        "Higher soil moisture increases canopy development (FPAR).",

        "Soil moisture directly affects crop yield.",

        "Vegetation vigor contributes to grain production.",

        "Greater absorbed radiation improves biomass and yield."

    ]

})

print("="*70)

print("SCIENTIFIC ASSUMPTIONS")

print("="*70)

print(assumptions)

assumptions.to_csv(

    REPORT_DIR / "expert_dag_assumptions.csv",

    index=False

)

print("Scientific assumptions exported successfully!")

# ============================================================
# FINAL EXPERT DAG
# ============================================================

expert_edges = [

    # ========================================================
    # GEOGRAPHY
    # ========================================================

    ("latitude", "avg_tavg"),

    # ========================================================
    # TEMPORAL
    # ========================================================

    ("harvest_year", "avg_tavg"),
    ("harvest_year", "avg_cwb"),

    # ========================================================
    # CLIMATE
    # ========================================================

    ("avg_tavg", "avg_cwb"),
    ("avg_rad", "avg_cwb"),

    # ========================================================
    # SOIL
    # ========================================================

    ("awc", "avg_ssm"),
    ("bulk_density", "avg_ssm"),
    ("drainage_class", "avg_ssm"),

    # ========================================================
    # CLIMATE → SOIL MOISTURE
    # ========================================================

    ("avg_cwb", "avg_ssm"),

    # ========================================================
    # TREATMENT → MEDIATORS
    # ========================================================

    ("avg_ssm", "avg_ndvi"),
    ("avg_ssm", "avg_fpar"),

    # ========================================================
    # MEDIATORS → OUTCOME
    # ========================================================

    ("avg_ndvi", "yield"),
    ("avg_fpar", "yield"),

    # ========================================================
    # DIRECT EFFECT
    # ========================================================

    ("avg_ssm", "yield")

]

print("="*70)

print("EXPERT DAG EDGES")

print("="*70)

for source, target in expert_edges:

    print(f"{source:20s} -----> {target}")

print()

print("Total Expert Relationships :", len(expert_edges))


# ============================================================
# BUILD EXPERT DAG
# ============================================================

G = nx.DiGraph()

G.add_edges_from(expert_edges)

print("="*70)

print("GRAPH CREATED")

print("="*70)

print()

print("Nodes :", G.number_of_nodes())

print("Edges :", G.number_of_edges())

print()

print("Is DAG :", nx.is_directed_acyclic_graph(G))

# ============================================================
# MANUAL DAG LAYOUT
# ============================================================

pos = {

    # ----------------------------
    # Background
    # ----------------------------

    "latitude": (-3,9),

    "harvest_year": (3,9),

    # ----------------------------
    # Climate
    # ----------------------------

    "avg_tavg": (0,7.3),

    "avg_rad": (-3,6),

    "avg_cwb": (0,5.4),

    # ----------------------------
    # Soil
    # ----------------------------

    "awc": (-4,3.5),

    "bulk_density": (0,3.5),

    "drainage_class": (4,3.5),

    # ----------------------------
    # Treatment
    # ----------------------------

    "avg_ssm": (0,2),

    # ----------------------------
    # Mediators
    # ----------------------------

    "avg_ndvi": (-2,0),

    "avg_fpar": (2,0),

    # ----------------------------
    # Outcome
    # ----------------------------

    "yield": (0,-2)

}


# ============================================================
# NODE COLORS
# ============================================================

color_map = {

    "latitude":"#C39BD3",
    "harvest_year":"#C39BD3",

    "avg_tavg":"#5DADE2",
    "avg_rad":"#5DADE2",
    "avg_cwb":"#5DADE2",

    "awc":"#D2B48C",
    "bulk_density":"#D2B48C",
    "drainage_class":"#D2B48C",

    "avg_ssm":"#F5B041",

    "avg_ndvi":"#58D68D",
    "avg_fpar":"#58D68D",

    "yield":"#EC7063"

}

node_colors = [

    color_map[node]

    for node in G.nodes()

]

# ============================================================
# DRAW EXPERT DAG
# ============================================================
plt.style.use("default")

plt.figure(figsize=(14,11))

nx.draw_networkx_nodes(

    G,

    pos,

    node_color=node_colors,

    node_size=3300,

    edgecolors="black",

    linewidths=1.8

)

nx.draw_networkx_edges(

    G,

    pos,

    arrows=True,

    arrowstyle="-|>",

    arrowsize=25,

    width=2.3,

    edge_color="gray",

    connectionstyle="arc3,rad=0.05"

)

nx.draw_networkx_labels(

    G,

    pos,

    font_size=10,

    font_weight="bold"

)

plt.title(

    "Expert Knowledge-Based Causal Directed Acyclic Graph",

    fontsize=18,

    weight="bold"

)

plt.axis("off")

save_plot("01_expert_dag.png")

print("Expert DAG Figure Saved Successfully!")


# ============================================================
# GRAPHVIZ PUBLICATION-QUALITY DAG
# ============================================================

dot = Digraph(
    "Expert_DAG",
    format="png"
)

dot.attr(rankdir="TB")
dot.attr(splines="ortho")
dot.attr(nodesep="0.45")
dot.attr(ranksep="0.75")
dot.attr(fontname="Helvetica")

# ============================================================
# NODE STYLE
# ============================================================

dot.attr(
    "node",
    shape="box",
    style="rounded,filled",
    fontname="Helvetica",
    fontsize="11",
    color="black"
)

# ============================================================
# NODES
# ============================================================

dot.node("latitude", "Latitude", fillcolor="#D6CDEA")

dot.node("harvest_year", "Harvest Year", fillcolor="#D6CDEA")

dot.node("avg_tavg", "Average\nTemperature", fillcolor="#AED6F1")

dot.node("avg_rad", "Solar\nRadiation", fillcolor="#AED6F1")

dot.node("avg_cwb", "Climate Water\nBalance", fillcolor="#AED6F1")

dot.node("awc", "Available Water\nCapacity", fillcolor="#F5CBA7")

dot.node("bulk_density", "Bulk\nDensity", fillcolor="#F5CBA7")

dot.node("drainage_class", "Drainage\nClass", fillcolor="#F5CBA7")

dot.node("avg_ssm", "Soil Moisture\n(Treatment)", fillcolor="#F8C471")

dot.node("avg_ndvi", "NDVI", fillcolor="#ABEBC6")

dot.node("avg_fpar", "FPAR", fillcolor="#ABEBC6")

dot.node("yield", "Crop Yield", fillcolor="#F1948A")

# ============================================================
# EDGES
# ============================================================

for source, target in expert_edges:
    dot.edge(source, target)

# ============================================================
# SAVE GRAPHVIZ DAG
# ============================================================

graphviz_path = FIGURE_DIR / "02_graphviz_expert_dag"

dot.render(str(graphviz_path), cleanup=True)

print("Graphviz Expert DAG Saved Successfully!")