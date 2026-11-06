# PPI-DynCT

PPI-DynCT is a graph-based model for predicting cell-type-specific responses to chemical perturbations. It combines control-state gene expression, cell-type-specific co-expression graphs, protein-protein interaction (PPI) priors, Node2Vec gene representations, and perturbation-conditioned dynamic cell-type embeddings.

The repository contains the training implementation used for the McFarland, Chang, and Nault experiments, preprocessing code, graph data, evaluation utilities, and plotting scripts.

## 1. Project Overview

The model receives a control expression state, a perturbation representation, and a graph describing the relevant cellular context. It predicts the perturbed expression profile for a specified cell type or cell line. The main training entry point is [`training/training_node2vec_dynamic.py`](training/training_node2vec_dynamic.py).

The implementation supports four model variants:

- `dynamic_single_n2v`: single graph branch with Node2Vec and dynamic cell-type conditioning.
- `dynamic_dual_n2v`: co-expression and PPI branches with Node2Vec and dynamic conditioning.
- `dynamic_dual_n2v_global`: dual branch with global scalar fusion.
- `dynamic_dual_n2v_concat`: dual branch with concatenation fusion.

## 2. Method Architecture

PPI-DynCT builds a cell-type representation from graph-encoded control states and conditions it on the perturbation embedding. The dynamic module uses a learned perturbation-dependent update and gate:

```text
control expression + cell-type graph ─┐
                                      ├─ graph encoder ─┐
PPI graph + Node2Vec features ────────┘                 │
                                                        ├─ dynamic cell-type embedding
drug / perturbation features ──────────────────────────┘
                                                        │
                                      response decoder ─┴─ predicted perturbed expression
```

The dynamic update is implemented in the training model modules and is controlled by the `dynamic_celltype` and `readout` sections in each YAML configuration. Node2Vec settings, PPI settings, split targets, and random seeds are recorded in the dataset-specific configuration files.

## 3. Repository Structure

The current source layout is retained so existing imports and experiment scripts continue to work. The directory names below describe the role of the existing folders.

```text
PPI-DynCT/
├── README.md
├── environment.yml
├── LICENSE.txt                         # existing upstream license notice
├── Data/                               # processed datasets, STRING files, splits
├── Data_Process/                      # data preprocessing scripts and notebooks
├── graphs/                             # cell-type-specific graph files
├── model_checkpoints/                  # saved model weights
├── Results/                            # predictions, metrics, and run artifacts
├── training/                           # training, evaluation, baselines, configs
├── model.py, model_dynamic_*.py        # model implementations kept at project root
├── attention_pooling.py, utils*.py      # shared model and data utilities
└── train_results/                      # historical experiment outputs
```

The suggested logical mapping is `Data_Process/ → preprocessing scripts and notebooks`, `models/ → root model modules plus training/model_*.py`, `training/ → training/`, `evaluation/ → training evaluation and baseline scripts`, and `scripts/ → runnable preprocessing and plotting scripts`. No large-scale move is required to use the project.

## 4. Requirements

The supplied environment file targets Python 3.10, PyTorch 1.13.1, and PyTorch Geometric 2.5.3. The main runtime also uses Scanpy, AnnData, NumPy, SciPy, scikit-learn, NetworkX, pandas, PyYAML, tqdm, and h5py.

GPU training requires a PyTorch build and PyTorch Geometric extensions compatible with the installed CUDA toolkit and driver. The exact CUDA wheel selection is machine-specific, so verify the PyTorch and CUDA versions before installing GPU extensions.

## 5. Installation

```bash
git clone https://github.com/huwenhao0112/PPI-DynCT.git
cd PPI-DynCT

conda env create -f environment.yml
conda activate ppi-dynct
```

For a CPU-only run, pass `--device cpu` to the training script. For automatic device selection, use `--device auto`. If the environment solver cannot install a compatible PyTorch Geometric build, install the PyTorch build for the target CUDA version first and then install the matching PyG packages from the official PyG wheel index.

## 6. Data Preparation

The main experiments use three processed datasets.

| Dataset | Expected expression file | Cell object | Evaluation setting |
| --- | --- | --- | --- |
| McFarland | `Data/McFarland_processed.h5ad` | `Data/cells_McFarland.pkl` | held-out cell lines and drug combinations |
| Chang | `Data/Chang_processed.h5ad` | `Data/cells_Chang.pkl` | held-out compounds in the PC9 cell line |
| Nault | `Data/Nault_processed.h5ad` | `Data/cells_Nault.pkl` | held-out liver cell type under TCDD |

The training script also expects the corresponding graph directory under `graphs/` and the shared drug feature file `Data/SMILES_feat_all_datasets.csv`. When the PPI branch is enabled, place the STRING v11.5 files in `Data/`:

```text
Data/9606.protein.links.v11.5.txt
Data/9606.protein.info.v11.5.txt
Data/9606.protein.aliases.v11.5.txt
```

See [`Data/Readme.md`](Data/Readme.md) for the source links, file names, and third-party data usage requirements. The preprocessing scripts and notebooks are in [`Data_Process/`](Data_Process/). The most commonly used graph builders include `build_ppi_network_v2.py`, `Chang_graphs.ipynb`, `McFarland_graphs.ipynb`, and `Nault_graphs.ipynb`.

Large expression matrices, graph pickles, checkpoints, and generated results should be distributed through an appropriate data archive or Git LFS when the GitHub repository size limit is a concern.

## 7. Model Training and Evaluation

Run commands from the repository root. The three configurations below are the entry points supplied for the main datasets.

### Train McFarland

```bash
python training/training_node2vec_dynamic.py \
  --config training/config_node2vec_dynamic_McFarland.yaml \
  --model_variant dynamic_single_n2v \
  --seed 42 \
  --device auto
```

### Train Chang

```bash
python training/training_node2vec_dynamic.py \
  --config training/config_node2vec_dynamic_Chang.yaml \
  --model_variant dynamic_single_n2v \
  --seed 42 \
  --device auto
```

### Train Nault

```bash
python training/training_node2vec_dynamic.py \
  --config training/config_node2vec_dynamic_Nault.yaml \
  --model_variant dynamic_dual_n2v \
  --seed 42 \
  --device auto
```

Useful overrides include `--epochs`, `--batch_size`, `--testing_cell_types`, `--testing_drugs`, `--deterministic`, `--no_plots`, `--run_dir`, and `--overwrite`. Use `python training/training_node2vec_dynamic.py --help` to see the complete list.

Each run writes model checkpoints under `model_checkpoints/` and prediction results under `Results/`. The evaluation stage reports MSE and R² for all genes and the top 100 variable or differentially expressed genes, and writes per-condition `.h5ad` files, scatter plots, `run_metrics.csv`, and `run_manifest.json`.

The deterministic response baselines can be run over the audited protocol catalog with:

```bash
python training/run_response_baselines.py \
  --project-dir . \
  --phases validation test \
  --output-namespace reproducibility
```

Additional baseline, ablation, audit, and plotting entry points are collected in `training/`. Review each script's `--help` output before launching a large experiment because some analyses require additional result files or external baseline environments.

## 8. Reproducibility

The main configurations use seed `42`. To reproduce a run, keep the following fixed:

1. The dataset file, graph directory, drug feature file, and cell object listed in the selected YAML configuration.
2. The training seed and Node2Vec seed.
3. The `testing_cell_type`, `testing_drugs`, `split_mode`, model variant, and Node2Vec options.
4. The PyTorch, PyG, CUDA, and Scanpy versions in the execution environment.

Use `--deterministic` when you need stricter repeatability and record the complete command line. A run directory created with `--run_dir` keeps model files, metrics, plots, and manifests together. The output `run_manifest.json` records the configuration and file paths used by that run.

The paper-facing dataset-to-configuration mapping is:

```text
McFarland → training/config_node2vec_dynamic_McFarland.yaml
Chang     → training/config_node2vec_dynamic_Chang.yaml
Nault     → training/config_node2vec_dynamic_Nault.yaml
```

The repository does not add a new license or citation record here. Before publishing, confirm the license for project-authored code and add `LICENSE` and `CITATION.cff` with the authors, title, version, and DOI information approved for the paper.

## Citation

Citation metadata will be added after the paper title, author list, version, and DOI are finalized.



