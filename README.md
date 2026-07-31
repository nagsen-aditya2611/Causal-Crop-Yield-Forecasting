# Causal Crop Yield Benchmarking

This repository contains notebook-based workflows for preprocessing agricultural panel data, exploring dynamical relationships, and performing causal analysis on crop yield datasets.

The current work focuses on wheat data for India and combines panel cleaning, convergent cross mapping (CCM), causal graph construction, and causal effect estimation with DoWhy.

## Scope

The repo appears to have two notebook pipelines:

- `CY-Bench-01/`: data checking, panel cleaning, and CCM preparation on a crop-climate panel.
- `CY-Bench-02/`: wheat preprocessing for India, CCM preparation and runs, DAG construction, and DoWhy-based causal estimation.

This fits a broader scientific workflow where domain structure is used to move from descriptive panel preprocessing toward explicit causal reasoning.

## Repository layout

```text
.
├── info.md
├── CY-Bench-01/
│   ├── 01_data_check.ipynb
│   ├── 02_panel_cleaning.ipynb
│   ├── 03_ccm_prep.ipynb
│   └── 04_ccm_run.ipynb
├── CY-Bench-02/
│   ├── 00_preprocessing.ipynb
│   ├── 01_ccm_prep.ipynb
│   ├── 02_ccm_run.ipynb
│   ├── 03_wheat_dag.ipynb
│   └── 04_dowhy.ipynb
└── data/
    ├── raw/
    ├── processed/
    ├── interim/
    └── results/
```

## Notebook guide

### CY-Bench-01

- `01_data_check.ipynb` checks the source dataset structure and initial consistency.
- `02_panel_cleaning.ipynb` standardizes column names, enforces types, removes duplicates, and writes a cleaned panel.
- `03_ccm_prep.ipynb` prepares panel data for CCM, checks temporal coverage by `adm_id`, and filters eligible units.
- `04_ccm_run.ipynb` runs CCM experiments.

### CY-Bench-02

- `00_preprocessing.ipynb` creates a processed wheat panel for India.
- `01_ccm_prep.ipynb` loads the processed panel, separates dynamic and static variables, checks missingness, and prepares the panel for CCM.
- `02_ccm_run.ipynb` runs CCM on the prepared wheat panel.
- `03_wheat_dag.ipynb` encodes the assumed causal graph for the wheat system.
- `04_dowhy.ipynb` identifies and estimates causal effects and runs refutation tests.

## Data expectations

The notebooks assume a local `data/` directory with subfolders such as `raw/`, `processed/`, and `results/`. Several notebooks use relative paths like `../../data/...` or `Path("../..")`, so cloning location and folder layout matter.

Some notebooks can read either CSV or Excel input. If your raw dataset is still in `.xlsx` form, the Excel engine dependency is needed.

## Variables used

Across the notebooks, the panel contains variables such as:

- identifiers: `crop_name`, `country_code`, `adm_id`, `harvest_year`
- outcomes and agricultural quantities: `yield`, `production`, `harvest_area`
- static or slowly varying attributes: `awc`, `bulk_density`, `drainage_class`, `latitude`, `longitude`, `region_area`, `crop_area`, `crop_area_percentage`, `sos`, `eos`
- meteorological variables: `avg_tmin`, `avg_tmax`, `avg_tavg`, `avg_prec`, `sum_prec`, `avg_rad`, `avg_et0`, `avg_vpd`, `avg_cwb`
- moisture and vegetation variables: `avg_ssm`, `avg_rsm`, `avg_ndvi`, `avg_fpar`

## Methods

The repository currently mixes several layers of analysis:

- panel cleaning and schema harmonization,
- temporal coverage filtering by administrative unit,
- standardization and preprocessing for CCM,
- causal graph specification,
- causal identification and estimation with DoWhy,
- refutation-based robustness checks.

For a scientific workflow like yours, this is a useful bridge between nonlinear or dynamical intuition and causal inference: CCM is probing directional dependence in time-ordered data, while the DAG and DoWhy stage forces explicit identification assumptions before estimating effects.

## Environment setup

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scriptsctivate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter

```bash
jupyter lab
```

## Recommended execution order

### CY-Bench-01

1. `01_data_check.ipynb`
2. `02_panel_cleaning.ipynb`
3. `03_ccm_prep.ipynb`
4. `04_ccm_run.ipynb`

### CY-Bench-02

1. `00_preprocessing.ipynb`
2. `01_ccm_prep.ipynb`
3. `02_ccm_run.ipynb`
4. `03_wheat_dag.ipynb`
5. `04_dowhy.ipynb`

## Reproducibility notes

- The repository is notebook-first rather than package-first.
- Paths are currently hard-coded relatively in several places.
- Raw data may not be committed to the repository.
- Output files are written into processed-data and results folders created by the notebooks.

## Dependency basis

The `requirements.txt` in this repository was assembled by checking notebook imports and the file-reading behavior in the notebooks. In particular, the notebooks directly import `numpy`, `pandas`, `matplotlib`, `sklearn.preprocessing.StandardScaler`, `pyEDM`, and `dowhy`, and the workflow also benefits from notebook runtime packages and Excel support.
