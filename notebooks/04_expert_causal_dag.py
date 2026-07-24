import warnings
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

warnings.filterwarnings("ignore")

plt.style.use("ggplot")

print("Libraries Imported Successfully!")


# OUTPUT DIRECTORIES

FIGURE_DIR = Path("figures/figures_dag")
REPORT_DIR = Path("reports")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("Output folders ready!")

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
# LOAD DATA
# ============================================================

df = pd.read_csv(

    "data/processed/final_crop_dataset.csv"

)

print("\nDataset Loaded Successfully!")

print(df.shape)

print(df.head())

# ============================================================
# DATASET INFORMATION
# ============================================================

print("="*60)

print("DATASET INFORMATION")

print("="*60)

print()

print("Rows :",df.shape[0])

print("Columns :",df.shape[1])

print()

print(df.columns.tolist())

# ============================================================
# DEFINE CAUSAL VARIABLES
# ============================================================

TREATMENT = "avg_ssm"

OUTCOME = "yield"

CONFOUNDERS = [

    "avg_tavg",
    "avg_tmax",
    "avg_tmin",
    "avg_rad",
    "avg_cwb",
    "awc",
    "bulk_density",
    "drainage_class",
    "latitude",
    "longitude",
    "crop_area_percentage",
    "harvest_year"

]

print("Treatment :", TREATMENT)

print("Outcome :", OUTCOME)

print()

print("Confounders")

for c in CONFOUNDERS:

    print("-", c)

# ============================================================
# EXPERT CAUSAL ASSUMPTIONS
# ============================================================

print("\n" + "="*70)
print("EXPERT CAUSAL ASSUMPTIONS")
print("="*70)

causal_assumptions = {

    "avg_tavg": "Seasonal temperature influences evaporation, crop stress, and yield.",

    "avg_tmax": "High temperature can reduce wheat productivity during critical growth stages.",

    "avg_tmin": "Night-time temperature affects respiration and grain filling.",

    "avg_rad": "Solar radiation drives photosynthesis and evapotranspiration.",

    "avg_cwb": "Climate water balance reflects water deficit or surplus conditions.",

    "awc": "Soil available water capacity determines moisture storage.",

    "bulk_density": "Soil compaction affects infiltration and root growth.",

    "drainage_class": "Drainage controls water retention and waterlogging risk.",

    "latitude": "Represents climatic variation across India.",

    "longitude": "Represents regional environmental differences.",

    "crop_area_percentage": "Represents agricultural intensity and management level.",

    "harvest_year": "Captures yearly climatic and technological variation."

}

for var, reason in causal_assumptions.items():

    print(f"\n{var}")

    print(f"  - {reason}")

with open(REPORT_DIR / "causal_assumptions.txt", "w") as f:

    f.write("EXPERT CAUSAL ASSUMPTIONS\n")

    f.write("=" * 70 + "\n\n")

    for var, reason in causal_assumptions.items():

        f.write(f"{var}: {reason}\n\n")

print("\nCausal assumptions exported successfully!")

# ============================================================
# BUILD EXPERT CAUSAL DAG
# ============================================================

print("\n" + "="*70)
print("BUILDING EXPERT CAUSAL DAG")
print("="*70)

G = nx.DiGraph()

nodes = [

    "Latitude",
    "Longitude",
    "Temperature",
    "Radiation",
    "Soil Properties",
    "Climate Water Balance",
    "Soil Moisture",
    "NDVI",
    "FPAR",
    "Agricultural Intensity",
    "Year",
    "Yield"

]

G.add_nodes_from(nodes)

edges = [

    ("Latitude", "Temperature"),
    ("Latitude", "Radiation"),
    ("Latitude", "Yield"),

    ("Longitude", "Temperature"),
    ("Longitude", "Radiation"),
    ("Longitude", "Yield"),

    ("Temperature", "Climate Water Balance"),
    ("Temperature", "Soil Moisture"),
    ("Temperature", "NDVI"),
    ("Temperature", "Yield"),

    ("Radiation", "NDVI"),
    ("Radiation", "Yield"),

    ("Soil Properties", "Soil Moisture"),
    ("Soil Properties", "Yield"),

    ("Climate Water Balance", "Soil Moisture"),
    ("Climate Water Balance", "Yield"),

    ("Soil Moisture", "NDVI"),
    ("Soil Moisture", "FPAR"),
    ("Soil Moisture", "Yield"),

    ("NDVI", "Yield"),
    ("FPAR", "Yield"),

    ("Agricultural Intensity", "Yield"),

    ("Year", "Soil Moisture"),
    ("Year", "Yield")

]

G.add_edges_from(edges)

print(f"\nNodes : {G.number_of_nodes()}")

print(f"Edges : {G.number_of_edges()}")

print("\nGraph Created Successfully!")

# ============================================================
# VISUALIZE EXPERT DAG
# ============================================================

print("\nGenerating DAG Figure...")

plt.figure(figsize=(16,10))

pos = {

    "Latitude":(-3,3),
    "Longitude":(-3,1),

    "Temperature":(-1,3),
    "Radiation":(-1,1),

    "Soil Properties":(-1,-1),

    "Climate Water Balance":(1,3),
    "Soil Moisture":(1,1),

    "NDVI":(3,2),
    "FPAR":(3,0),

    "Agricultural Intensity":(1,-2),

    "Year":(-3,-2),

    "Yield":(5,1)

}

node_colors = []

for node in G.nodes():

    if node == "Soil Moisture":

        node_colors.append("orange")

    elif node == "Yield":

        node_colors.append("red")

    elif node in ["NDVI","FPAR"]:

        node_colors.append("lightgreen")

    else:

        node_colors.append("skyblue")

nx.draw_networkx(

    G,

    pos=pos,

    node_color=node_colors,

    node_size=4200,

    font_size=10,

    font_weight="bold",

    arrows=True,

    arrowsize=22,

    edge_color="gray",

    width=2

)

plt.title(

    "Expert Causal DAG\nSeasonal Soil Moisture → Wheat Yield",

    fontsize=16,

    weight="bold"

)

plt.axis("off")

save_plot("expert_causal_dag.png")

print("Expert DAG Saved Successfully!")

# ============================================================
# EXPORT DAG
# ============================================================

nx.write_gml(

    G,

    REPORT_DIR / "expert_causal_dag.gml"

)

with open(

    REPORT_DIR / "expert_dag_summary.txt",

    "w"

) as f:

    f.write("EXPERT DAG SUMMARY\n")

    f.write("="*70 + "\n\n")

    f.write(f"Treatment : {TREATMENT}\n")

    f.write(f"Outcome : {OUTCOME}\n\n")

    f.write(f"Nodes : {G.number_of_nodes()}\n")

    f.write(f"Edges : {G.number_of_edges()}\n\n")

    f.write("Edges\n")

    f.write("-"*40 + "\n")

    for edge in G.edges():

        f.write(f"{edge[0]} --> {edge[1]}\n")

print("\nExpert DAG Exported Successfully!")

print("\nNotebook 4 Completed Successfully!")

print(f"\nFigure Saved : {FIGURE_DIR / 'expert_causal_dag.png'}")

print(f"GML Saved    : {REPORT_DIR / 'expert_causal_dag.gml'}")

print(f"Summary      : {REPORT_DIR / 'expert_dag_summary.txt'}")