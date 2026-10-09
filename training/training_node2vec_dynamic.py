#!/usr/bin/env python
# coding: utf-8

"""
training_node2vec_dynamic.py
============================
Unified training script that combines innovations from:
  1. training_dynamic_celltype.py: Perturbation-conditioned Dynamic Cell-type Embedding
  2. training_node2vec_baseline.py: Node2Vec-augmented gene features + PPI topology features

This script supports:
  - Node2Vec pre-trained embeddings (global, local, or both modes)
  - PPI topology features (degree, clustering, PageRank, k-core)
  - Expression statistics (CV, skewness, fraction expressed)
  - Dynamic cell-type modulation conditioned on perturbation
  - Dual-branch GNN (co-expression + PPI) with fusion gates
  - Statistical readout (max + mean + std pooling)

Supported model variants:
  - dynamic_single_n2v      : Single-branch GNN + Node2Vec + Dynamic cell-type
  - dynamic_dual_n2v        : Dual-branch GNN + Node2Vec + Dynamic cell-type (recommended)
  - dynamic_dual_n2v_global : Dual-branch (global scalar fusion) + Node2Vec + Dynamic
  - dynamic_dual_n2v_concat : Dual-branch (concat fusion) + Node2Vec + Dynamic

Usage::

    # Recommended: dual-branch with Node2Vec
    python training/training_node2vec_dynamic.py \
        --config training/config_node2vec_dynamic_Kang.yaml \
        --model_variant dynamic_dual_n2v

    # Single-branch variant
    python training/training_node2vec_dynamic.py \
        --config training/config_node2vec_dynamic_Kang.yaml \
        --model_variant dynamic_single_n2v

    # Ablation variants
    python training/training_node2vec_dynamic.py \
        --config training/config_node2vec_dynamic_Kang.yaml \
        --model_variant dynamic_dual_n2v_global

    python training/training_node2vec_dynamic.py \
        --config training/config_node2vec_dynamic_Kang.yaml \
        --model_variant dynamic_dual_n2v_concat

Key Innovations Combined:
------------------------
1. Node2Vec Features:
   - Global Node2Vec on full PPI network (captures global protein interaction patterns)
   - Local Node2Vec on cell-type-specific PPI subgraphs (captures cell-type-specific patterns)
   - Normalized via z-score or min-max

2. PPI Topology Features:
   - Degree centrality, weighted degree
   - Clustering coefficient
   - PageRank
   - K-core number
   - Neighbor degree statistics

3. Dynamic Cell-type Embedding:
   - Static: h_c = GNN(G_c^control)
   - Dynamic: h_{c,p} = h_c + γ_{c,p} ⊙ Δh_{c,p}
   where:
     z_{c,p}  = concat(h_c, e_p)
     Δh_{c,p} = tanh(MLP_delta(z))
     γ_{c,p}  = sigmoid(MLP_gate(z))
"""

import os
import sys
import pickle
import argparse
import csv
import json
import random
import hashlib
from pathlib import Path

import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import torch
import networkx as nx
from torch_geometric.data import DataLoader
from tqdm import tqdm
from sklearn import metrics
from scipy.stats import skew as scipy_skew
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, '/root/autodl-tmp/Cell-Type-Specific-Graphs-main')

import utils
from training.native_multiseed_utils import split_samples, sample_cell_type, sample_drug
from utils import (
    loss_fct,
    rank_genes,
    create_anndata,
)

try:
    from utils_ppi import load_ppi_network, create_ppi_graph
except ImportError:
    load_ppi_network = None
    create_ppi_graph = None

from training.model_node2vec_dynamic import (
    DynamicCellTypeGNNNode2Vec,
    DynamicCellTypeDualPriorGNNNode2Vec,
    DynamicCellTypeDualPriorGNNNode2VecGlobal,
    DynamicCellTypeDualPriorGNNNode2VecConcat,
    FusionEntropyLoss,
)

if torch.cuda.is_available():
    print("CUDA is available. GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. Running on CPU.")


def load_config(config_file: str):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def file_sha256(path: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def set_global_seed(seed: int, deterministic: bool = False):
    """Seed Python, NumPy and PyTorch without resetting RNGs inside models."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def collapse_duplicate_degree_checkpoint(state_dict, model):
    """Convert the legacy 73-D projector to the deduplicated 72-D projector.

    Legacy topology input used [degree, weighted_degree, ...], but the graph
    loader ignored the stored ``weight`` key and constructed an unweighted
    NetworkX graph. Consequently both standardized columns were identical.
    Summing their first-layer weights and deleting the duplicate is therefore
    function-preserving for those checkpoints.
    """
    key = 'node_projector.projector.0.weight'
    if key not in state_dict or key not in model.state_dict():
        return state_dict, False
    source = state_dict[key]
    target = model.state_dict()[key]
    if source.ndim != 2 or source.shape[1] != target.shape[1] + 1:
        return state_dict, False
    topology_dim = 6
    base_dim = target.shape[1] - topology_dim
    if base_dim < 2:
        return state_dict, False
    converted = torch.cat([
        source[:, :base_dim],
        source[:, base_dim:base_dim + 1] + source[:, base_dim + 1:base_dim + 2],
        source[:, base_dim + 2:],
    ], dim=1)
    if converted.shape != target.shape:
        return state_dict, False
    state_dict = dict(state_dict)
    state_dict[key] = converted
    return state_dict, True


ABLATION_PROTOCOL = "ppi_features_x_dynamic_v2"

ABLATION_SPECS = {
    # Controlled 2 x 2 design:
    # PPI-derived node features (off/on) x dynamic cell type (off/on).
    # The raw PPI graph branch is deliberately disabled in all four variants.
    "original": {
        "model_variant": "dynamic_single_n2v",
        "ppi_enabled": False,
        "use_dynamic": False,
        "node2vec_mode": "none",
    },
    "ppi": {
        "model_variant": "dynamic_single_n2v",
        "ppi_enabled": False,
        "use_dynamic": False,
        "node2vec_mode": "global",
    },
    "dynamic": {
        "model_variant": "dynamic_single_n2v",
        "ppi_enabled": False,
        "use_dynamic": True,
        "node2vec_mode": "none",
    },
    "full": {
        "model_variant": "dynamic_single_n2v",
        "ppi_enabled": False,
        "use_dynamic": True,
        "node2vec_mode": "global",
    },
}


# =============================================================================
# Node2Vec Training
# =============================================================================

def train_node2vec_model(
    edge_index, num_nodes, embedding_dim=64,
    walk_length=40, context_size=20, walks_per_node=10,
    p=1.0, q=1.0, num_epochs=5, batch_size=256, lr=0.01, seed=42,
):
    """Train Node2Vec embeddings on a graph."""
    from torch_geometric.nn import Node2Vec

    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"    [Node2Vec] device={dev}, dim={embedding_dim}, walk_len={walk_length}, "
          f"epochs={num_epochs}, p={p}, q={q}")

    edge_index_max = edge_index.max().item() if edge_index.numel() > 0 else -1
    if edge_index_max >= num_nodes:
        print(f"    [WARNING] edge_index contains node index {edge_index_max} but num_nodes={num_nodes}. "
              f"Adjusting num_nodes to {edge_index_max + 1}")
        num_nodes = edge_index_max + 1

    n2v = Node2Vec(
        edge_index, embedding_dim=embedding_dim,
        walk_length=walk_length, context_size=context_size,
        walks_per_node=walks_per_node, p=p, q=q,
        num_negative_samples=1, sparse=True,
    ).to(dev)

    loader = n2v.loader(batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(list(n2v.parameters()), lr=lr)

    n2v.train()
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        for pos_batch, neg_batch in loader:
            optimizer.zero_grad()
            loss = n2v.loss(pos_batch.to(dev), neg_batch.to(dev))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"      [Epoch {epoch}/{num_epochs}] loss = {epoch_loss / max(1, len(loader)):.4f}")

    n2v.eval()
    with torch.no_grad():
        valid_nodes = torch.unique(edge_index.flatten()).long()
        embeddings = n2v(valid_nodes.to(dev)).cpu()

        full_embeddings = torch.zeros(num_nodes, embedding_dim, dtype=torch.float32)
        for i, node_idx in enumerate(valid_nodes):
            if node_idx < num_nodes:
                full_embeddings[node_idx] = embeddings[i]
        return full_embeddings


def train_node2vec_local(
    ppi_per_celltype, all_gene_names, embedding_dim=64,
    walk_length=40, context_size=20, walks_per_node=10,
    p=1.0, q=1.0, num_epochs=5, batch_size=256, lr=0.01, seed=42,
):
    """Train Node2Vec on each cell-type's PPI subgraph separately."""
    n_genes = len(all_gene_names)
    results = {}

    for ct, ppi_data in tqdm(ppi_per_celltype.items(), desc="  [Node2Vec local] Training per cell type"):
        ei = ppi_data['edge_index']
        if ei.numel() == 0 or ei.shape[1] < 3:
            results[ct] = torch.zeros(n_genes, embedding_dim)
            continue

        src_nodes = set(ei[0].tolist())
        dst_nodes = set(ei[1].tolist())
        local_nodes = sorted(src_nodes | dst_nodes)
        if len(local_nodes) < 3:
            results[ct] = torch.zeros(n_genes, embedding_dim)
            continue

        local_to_new = {n: i for i, n in enumerate(local_nodes)}
        n_local = len(local_nodes)
        remapped_ei = torch.tensor(
            [[local_to_new[n] for n in ei[0].tolist()],
             [local_to_new[n] for n in ei[1].tolist()]],
            dtype=torch.long,
        )

        local_emb = train_node2vec_model(
            edge_index=remapped_ei, num_nodes=n_local,
            embedding_dim=embedding_dim, walk_length=walk_length,
            context_size=context_size, walks_per_node=walks_per_node,
            p=p, q=q, num_epochs=num_epochs, batch_size=batch_size,
            lr=lr, seed=seed,
        )

        global_emb = torch.zeros(n_genes, embedding_dim)
        for local_idx, global_idx in enumerate(local_nodes):
            global_emb[global_idx] = local_emb[local_idx]

        results[ct] = global_emb

    return results


# =============================================================================
# PPI Topology Feature Computation
# =============================================================================

def compute_ppi_topology_features(ppi_graph, verbose=True, node_names=None):
    """
    Compute topology features for each gene in the PPI network.

    Returns dict: {gene_name: [degree, clustering, pagerank, k_core,
                               neighbor_degree_mean, neighbor_degree_std]}
    """
    if isinstance(ppi_graph, nx.DiGraph):
        ppi_graph = ppi_graph.to_undirected()

    n_nodes = ppi_graph.number_of_nodes()
    if verbose:
        print(f"  Computing topology features for {n_nodes} nodes...")

    degree_dict = dict(ppi_graph.degree())

    clustering_dict = nx.clustering(ppi_graph)

    try:
        pagerank_dict = nx.pagerank(ppi_graph, weight=None)
    except Exception:
        pagerank_dict = {n: 1.0/n_nodes for n in ppi_graph.nodes()}

    try:
        kcore_dict = nx.core_number(ppi_graph)
    except Exception:
        kcore_dict = {n: 0 for n in ppi_graph.nodes()}

    neighbor_degree_mean_dict = {}
    neighbor_degree_std_dict = {}
    for node in ppi_graph.nodes():
        neighbors = list(ppi_graph.neighbors(node))
        if len(neighbors) > 0:
            neigh_degrees = [degree_dict.get(n, 0) for n in neighbors]
            neighbor_degree_mean_dict[node] = np.mean(neigh_degrees)
            neighbor_degree_std_dict[node] = np.std(neigh_degrees) if len(neighbors) > 1 else 0.0
        else:
            neighbor_degree_mean_dict[node] = 0.0
            neighbor_degree_std_dict[node] = 0.0

    features_dict = {}
    for node in ppi_graph.nodes():
        if node_names is not None:
            key = node_names[node]
        else:
            key = node
        features_dict[key] = [
            degree_dict.get(node, 0),
            clustering_dict.get(node, 0),
            pagerank_dict.get(node, 0),
            kcore_dict.get(node, 0),
            neighbor_degree_mean_dict.get(node, 0),
            neighbor_degree_std_dict.get(node, 0),
        ]

    if verbose:
        feat_array = np.array(list(features_dict.values()))
        print(f"  Topology features stats:")
        print(f"    Degree:        min={feat_array[:,0].min():.1f}, max={feat_array[:,0].max():.1f}, mean={feat_array[:,0].mean():.1f}")
        print(f"    Pagerank:      min={feat_array[:,2].min():.6f}, max={feat_array[:,2].max():.6f}, mean={feat_array[:,2].mean():.6f}")
        print(f"    Clustering:    min={feat_array[:,1].min():.3f}, max={feat_array[:,1].max():.3f}, mean={feat_array[:,1].mean():.3f}")

    return features_dict


def normalize_features(feat_matrix, method='zscore'):
    """Normalize feature matrix column-wise."""
    feat_matrix = np.array(feat_matrix, dtype=np.float32)

    if method == 'zscore':
        col_means = feat_matrix.mean(axis=0)
        col_stds = feat_matrix.std(axis=0)
        feat_matrix = (feat_matrix - col_means) / (col_stds + 1e-8)
    elif method == 'minmax':
        col_mins = feat_matrix.min(axis=0)
        col_maxs = feat_matrix.max(axis=0)
        feat_matrix = (feat_matrix - col_mins) / (col_maxs - col_mins + 1e-8)
    elif method == 'none':
        pass

    return feat_matrix


# =============================================================================
# Node Feature Building
# =============================================================================

def safe_zscore(x):
    """Z-score normalization along last axis, std=0 returns 0."""
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (x - mean) / std


def safe_minmax(x):
    """Min-max normalization to [0, 1] along last axis."""
    mn = x.min(axis=0, keepdims=True)
    mx = x.max(axis=0, keepdims=True)
    rng = mx - mn
    rng[rng < 1e-8] = 1.0
    return (x - mn) / rng


def normalize_n2v(arr, method='zscore'):
    if method == 'zscore':
        return safe_zscore(arr)
    elif method == 'minmax':
        return safe_minmax(arr)
    return arr


def build_node_features(
    cell_type_network, ctrl_data, adata,
    node2vec_global=None, node2vec_local=None,
    normalize='zscore', mode='global',
    gene_to_idx=None, topology_features=None,
    use_expr_stats=True,
):
    """
    Build augmented node features combining:
      - Mean and variance expression (from control cells)
      - Node2Vec embeddings (global and/or local)
      - PPI topology features
      - Expression statistics (CV, skewness, fraction expressed, S/G2M scores)

    mode: 'global', 'local', or 'both'
    """
    ct_keys = list(cell_type_network.keys())

    n2v_global_fill = None
    if node2vec_global is not None:
        n2v_arr = node2vec_global.numpy()
        n2v_global_fill = n2v_arr.mean(axis=0)

    n2v_local_fill = None
    if node2vec_local is not None and ct_keys:
        first_ct = ct_keys[0]
        if first_ct in node2vec_local:
            n2v_local_fill = node2vec_local[first_ct].numpy().mean(axis=0)
        else:
            n2v_local_fill = np.zeros(node2vec_local[list(node2vec_local.keys())[0]].shape[1])

    for ct in tqdm(ct_keys, desc="Building node features"):
        g = cell_type_network[ct]
        genes = g.pos.tolist()
        n_local = len(genes)

        ctrl_subset = ctrl_data[ctrl_data.obs['cell_type'] == ct, genes].copy()
        try:
            if hasattr(ctrl_subset.X, 'toarray'):
                x_arr = ctrl_subset.X.toarray().astype(np.float32)
            elif hasattr(ctrl_subset.X, 'A'):
                x_arr = ctrl_subset.X.A.astype(np.float32)
            else:
                x_arr = np.asarray(ctrl_subset.X, dtype=np.float32)
        except Exception:
            x_arr = np.asarray(ctrl_subset.X.toarray(), dtype=np.float32)

        mean_expr = x_arr.mean(axis=0)
        var_expr = x_arr.var(axis=0)

        f_mean, f_std = mean_expr.mean(), mean_expr.std()
        mean_expr = (mean_expr - f_mean) / (f_std + 1e-8)
        f_mean, f_std = var_expr.mean(), var_expr.std()
        var_expr = (var_expr - f_mean) / (f_std + 1e-8)

        features = [mean_expr[:, np.newaxis], var_expr[:, np.newaxis]]

        # Expression statistics
        if use_expr_stats:
            std_expr = x_arr.std(axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                cv_expr = std_expr / (mean_expr + 1e-8)
                cv_expr = np.nan_to_num(cv_expr, nan=0.0, posinf=0.0, neginf=0.0)

            skew_expr = scipy_skew(x_arr, axis=0)
            skew_expr = np.nan_to_num(skew_expr, nan=0.0, posinf=0.0, neginf=0.0)

            frac_expr = (x_arr > 0).sum(axis=0) / x_arr.shape[0]

            # S/G2M scores are cell-level values, not gene-level node features.
            extra_feats = [cv_expr, skew_expr, frac_expr]
            for i, arr in enumerate(extra_feats):
                f_mean, f_std = arr.mean(), arr.std()
                if f_std > 1e-8:
                    extra_feats[i] = (arr - f_mean) / f_std
                else:
                    extra_feats[i] = arr - f_mean
            extra_feats = [f[:, np.newaxis] for f in extra_feats]

            if x_arr.shape[0] > 1:
                features.extend(extra_feats)

        # Global Node2Vec
        if mode in ('global', 'both') and node2vec_global is not None:
            global_n2v = np.zeros((n_local, node2vec_global.shape[1]), dtype=np.float32)
            for i, gene in enumerate(genes):
                g_idx = gene_to_idx.get(gene, -1)
                if g_idx >= 0:
                    global_n2v[i] = node2vec_global[g_idx].numpy()
                else:
                    global_n2v[i] = n2v_global_fill
            global_n2v = normalize_n2v(global_n2v, normalize)
            features.append(global_n2v)

        # Local Node2Vec
        if mode in ('local', 'both') and node2vec_local is not None:
            if ct in node2vec_local:
                local_n2v_raw = node2vec_local[ct].numpy()
                local_n2v = np.zeros((n_local, local_n2v_raw.shape[1]), dtype=np.float32)
                for i, gene in enumerate(genes):
                    g_idx = gene_to_idx.get(gene, -1)
                    if g_idx >= 0:
                        local_vec = local_n2v_raw[g_idx]
                        if local_vec.abs().sum() > 1e-6:
                            local_n2v[i] = local_vec
                        elif node2vec_global is not None:
                            local_n2v[i] = node2vec_global[g_idx].numpy()
                        else:
                            local_n2v[i] = n2v_local_fill
                    else:
                        local_n2v[i] = n2v_local_fill
            else:
                dim = node2vec_local[list(node2vec_local.keys())[0]].shape[1]
                local_n2v = np.zeros((n_local, dim), dtype=np.float32)
            local_n2v = normalize_n2v(local_n2v, normalize)
            features.append(local_n2v)

        # Topology features
        if topology_features is not None:
            topology_dim = len(next(iter(topology_features.values())))
            topo_arr = np.zeros((n_local, topology_dim), dtype=np.float32)
            for i, gene in enumerate(genes):
                if gene in topology_features:
                    topo_arr[i] = topology_features[gene]
            topo_arr = normalize_features(topo_arr, method=normalize)
            features.append(topo_arr)

        feat_matrix = np.hstack(features).astype(np.float32)
        cell_type_network[ct].x = torch.from_numpy(feat_matrix)
        print(f"  {ct:30s}  x.shape = {feat_matrix.shape}")


# =============================================================================
# Training Loop
# =============================================================================

def train_node2vec_dynamic(
    model,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    cell_type_network,
    ppi_network: dict,
    train_loader,
    multi_pert: bool = True,
    entropy_beta: float = 0.0,
    device: str = 'cuda',
):
    """
    Training loop for Node2Vec + Dynamic Cell-type models.

    Parameters
    ----------
    model : nn.Module
        One of DynamicCellTypeGNNNode2Vec, DynamicCellTypeDualPriorGNNNode2Vec, etc.
    num_epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    weight_decay : float
        Weight decay for Adam optimizer.
    cell_type_network : dict
        Co-expression graphs per cell type (with augmented node features).
    ppi_network : dict
        PPI graphs per cell type.
    train_loader : DataLoader
        Training data loader.
    multi_pert : bool
        Whether to use multi-drug SMILES embeddings.
    entropy_beta : float
        Coefficient for fusion gate entropy regularisation.
    device : str
        Device to train on ('cuda' or 'cpu').

    Returns
    -------
    model : nn.Module
        Trained model.
    """
    print('Training Starts')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=lr * 0.01
    )

    model = model.to(device).float()

    entropy_loss_fn = FusionEntropyLoss(beta=entropy_beta) if entropy_beta > 0 else None
    use_ppi = bool(ppi_network)

    for epoch in tqdm(range(num_epochs), leave=False):
        running_loss = 0.0
        train_epoch_loss = 0.0

        for sample in tqdm(train_loader, leave=False):
            model.train()
            sample = sample.to(device)
            cell_type = sample.cell_type
            ctrl = sample.x

            if multi_pert:
                pert_label = sample.pert_label
            else:
                pert_label = None

            y = sample.y

            cell_graphs_x = {
                Cell: cell_type_network[Cell].x.to(device)
                for Cell in np.unique(cell_type)
            }
            cell_graphs_edges = {
                Cell: cell_type_network[Cell].edge_index.to(device)
                for Cell in np.unique(cell_type)
            }

            ppi_graphs_x = {}
            ppi_graphs_edges = {}
            if use_ppi:
                ppi_graphs_x = {
                    Cell: ppi_network[Cell].x.to(device)
                    for Cell in np.unique(cell_type)
                    if Cell in ppi_network
                }
                ppi_graphs_edges = {
                    Cell: ppi_network[Cell].edge_index.to(device)
                    for Cell in np.unique(cell_type)
                    if Cell in ppi_network
                }

            has_gate_logits = hasattr(model, 'fusion_gate') and use_ppi

            if has_gate_logits:
                out, extras = model(
                    cell_graphs_x, cell_graphs_edges,
                    cell_type, cell_graphs_edges.keys(), ctrl, pert_label,
                    None,
                    ppi_graphs_x=ppi_graphs_x,
                    ppi_graphs_edges=ppi_graphs_edges,
                    return_gate_logits=True,
                )
                gate_logits = extras.get("gate_logits")
            else:
                out = model(
                    cell_graphs_x, cell_graphs_edges,
                    cell_type, cell_graphs_edges.keys(), ctrl, pert_label,
                    None,
                    ppi_graphs_x=ppi_graphs_x,
                    ppi_graphs_edges=ppi_graphs_edges,
                )
                gate_logits = None

            loss = loss_fct(out, y, sample.cov_drug)

            if entropy_loss_fn is not None and gate_logits is not None:
                loss = loss + entropy_loss_fn(gate_logits)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        train_epoch_loss = running_loss / len(train_loader)
        scheduler.step()

        if epoch % 10 == 0:
            if hasattr(model, 'dynamic_modulator'):
                scale_val = model.dynamic_modulator.dynamic_scale.item()
                print(f"Epoch {epoch}, train loss: {train_epoch_loss:.6f}  |  dynamic_scale: {scale_val:.4f}")
            else:
                print(f"Epoch {epoch}, train loss: {train_epoch_loss:.6f}")

    return model


# =============================================================================
# Inference Helper
# =============================================================================

def inference_node2vec_dynamic(
    cell_type_network: dict,
    ppi_network: dict,
    model,
    save_path_res: str,
    ood_loader,
    adata,
    degs_dict: dict,
    device: str = 'cuda',
    mean_or_std: bool = True,
    plot: bool = True,
    multi_pert: bool = True,
):
    """
    Inference for Node2Vec + Dynamic cell-type models.
    """
    pred, truth, cov_drugs = [], [], []
    metric_rows = []
    use_ppi = bool(ppi_network)

    with torch.no_grad():
        model.eval()
        for sample in tqdm(ood_loader, leave=False):
            sample = sample.to(device)
            cell_type = sample.cell_type
            ctrl = sample.x

            if multi_pert:
                pert_label = sample.pert_label
            else:
                pert_label = None

            y = sample.y

            cell_graphs_x = {
                Cell: cell_type_network[Cell].x.to(device)
                for Cell in np.unique(cell_type)
            }
            cell_graphs_edges = {
                Cell: cell_type_network[Cell].edge_index.to(device)
                for Cell in np.unique(cell_type)
            }

            ppi_graphs_x = {}
            ppi_graphs_edges = {}
            if use_ppi:
                ppi_graphs_x = {
                    Cell: ppi_network[Cell].x.to(device)
                    for Cell in np.unique(cell_type)
                    if Cell in ppi_network
                }
                ppi_graphs_edges = {
                    Cell: ppi_network[Cell].edge_index.to(device)
                    for Cell in np.unique(cell_type)
                    if Cell in ppi_network
                }

            out = model(
                cell_graphs_x, cell_graphs_edges,
                cell_type, cell_graphs_edges.keys(), ctrl, pert_label,
                None,
                ppi_graphs_x=ppi_graphs_x,
                ppi_graphs_edges=ppi_graphs_edges,
            )

            pred.extend(out)
            truth.extend(y)
            cov_drugs.extend(sample.cov_drug)

    pred = torch.stack(pred).cpu().numpy()
    truth = torch.stack(truth).cpu().numpy()
    perts = np.array(cov_drugs)

    for p in sorted(set(perts)):
        c = next(
            (ct for ct in cell_type_network if p.startswith(f"{ct}_")),
            p.split("_", 1)[0],
        )
        d = p[len(c) + 1:] if p.startswith(f"{c}_") else p
        pos_genes = cell_type_network[c].pos

        p_pred = pred[:, pos_genes.tolist()]
        p_truth = truth[:, pos_genes.tolist()]
        pert_idx = np.where(perts == p)[0]
        y_p = p_truth[pert_idx]
        pred_p = p_pred[pert_idx]

        Ann_Data = utils.create_anndata(pred_p, y_p, adata, cell_type_network, p)
        DEGs = degs_dict[p]
        Ann_Data.uns['DEGs'] = DEGs
        Path(save_path_res).mkdir(parents=True, exist_ok=True)
        Ann_Data.write(Path(save_path_res) / f'{p}_pred.h5ad')

        mse = np.mean((y_p - pred_p) ** 2)
        print(f"{p}  MSE: {mse:.6f}")

        if mean_or_std:
            x_stat = np.mean(y_p, axis=0)
            y_stat = np.mean(pred_p, axis=0)
            stat_name = "mean"
        else:
            x_stat = np.std(y_p, axis=0)
            y_stat = np.std(pred_p, axis=0)
            stat_name = "std"

        r2_all = metrics.r2_score(x_stat, y_stat)
        if len(DEGs) >= 2:
            r2_deg = metrics.r2_score(x_stat[DEGs], y_stat[DEGs])
        else:
            r2_deg = np.nan
        print(f"  R² ({stat_name}, all genes):  {r2_all:.4f}")
        print(f"  R² ({stat_name}, top 100 DEGs): {r2_deg:.4f}")

        common = {
            "cov_drug": p,
            "cell_type": c,
            "drug": d,
            "stat": stat_name,
            "n_cells": int(len(pert_idx)),
            "mse": float(mse),
        }
        metric_rows.extend([
            {
                **common,
                "gene_scope": "all",
                "n_genes": int(len(x_stat)),
                "r2": float(r2_all),
            },
            {
                **common,
                "gene_scope": "top100",
                "n_genes": int(len(DEGs)),
                "r2": float(r2_deg) if np.isfinite(r2_deg) else None,
            },
        ])

        if plot:
            sns.set_style("darkgrid")
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.regplot(x=x_stat, y=y_stat, ci=None, color="#1C2E54", ax=ax)
            ax.text(
                ax.get_xlim()[1] * 0.65, ax.get_ylim()[1] * 0.85,
                f'$R^2_{{all}}$ = {r2_all:.4f}',
                fontsize='large',
            )
            ax.text(
                ax.get_xlim()[1] * 0.65, ax.get_ylim()[1] * 0.75,
                f'$R^2_{{DEGs}}$ = {r2_deg:.4f}',
                fontsize='large',
            )
            ax.set_xlabel(f'True {stat_name} expression')
            ax.set_ylabel(f'Predicted {stat_name} expression')
            Path(save_path_res).mkdir(parents=True, exist_ok=True)
            fig.savefig(
                Path(save_path_res) / f'{p}_{stat_name}_scatter.pdf',
                bbox_inches='tight',
            )
            plt.close(fig)

    return metric_rows


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Node2Vec + Dynamic Cell-type Embedding Training"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='/root/autodl-tmp/Cell-Type-Specific-Graphs-main/training/config_node2vec_dynamic_Kang.yaml',
        help='Path to config YAML file.',
    )
    parser.add_argument(
        '--model_variant',
        choices=[
            'dynamic_single_n2v',
            'dynamic_dual_n2v',
            'dynamic_dual_n2v_global',
            'dynamic_dual_n2v_concat',
        ],
        default=None,
        help='Model variant. Overrides config file if set.',
    )
    parser.add_argument(
        '--ablation_variant',
        choices=list(ABLATION_SPECS),
        default=None,
        help=(
            "Canonical 2x2 ablation: original, ppi, dynamic, or full. "
            "This overrides model_variant, raw ppi.enabled, and node2vec_mode."
        ),
    )
    parser.add_argument('--seed', type=int, default=None,
                        help='Training seed; overrides config seed.')
    parser.add_argument('--node2vec_seed', type=int, default=None,
                        help='Node2Vec seed; defaults to the training seed.')
    parser.add_argument('--run_dir', type=str, default=None,
                        help='Isolated output directory for this run.')
    parser.add_argument('--split_mode', choices=['loco', 'lodo', 'double_ood', 'pair'], default='loco',
                        help='LOCO removes the held-out cell line from all training conditions.')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override the configured number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override the configured batch size.')
    parser.add_argument('--testing_cell_types', nargs='+', default=None,
                        help='Override testing_cell_type from the config.')
    parser.add_argument('--testing_drugs', nargs='+', default=None,
                        help='Override testing_drugs from the config.')
    parser.add_argument('--deterministic', action='store_true',
                        help='Request deterministic PyTorch algorithms (warn-only).')
    parser.add_argument('--no_plots', action='store_true',
                        help='Skip per-condition scatter plots.')
    parser.add_argument('--node2vec_dim', type=int, default=None,
                        help='Override Node2Vec embedding dimension')
    parser.add_argument('--node2vec_mode', type=str, default=None,
                        choices=['none', 'global', 'local', 'both'],
                        help='Node2Vec mode: global, local, or both')
    parser.add_argument('--normalize_n2v', type=str, default=None,
                        choices=['zscore', 'minmax', 'none'],
                        help='Normalization for Node2Vec features')
    parser.add_argument('--overwrite', action='store_true',
                        help='Retrain Node2Vec even if cached .pt exists')
    parser.add_argument('--use_expr_stats', type=str, default=None,
                        choices=['true', 'false'],
                        help='Enable extra expression stats')
    parser.add_argument('--readout_mode', type=str, default=None,
                        choices=['max', 'stat', 'residual_stat'],
                        help='Override graph readout mode.')
    parser.add_argument('--collect_dynamic_mechanisms', action='store_true',
                        help='Export dynamic counterfactuals and gamma/delta_h/residual.')
    parser.add_argument('--mechanism_swaps', type=int, default=8,
                        help='Wrong-drug and wrong-cell donors per condition.')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help='Load a state_dict checkpoint before inference or fine-tuning.')
    parser.add_argument('--skip_training', action='store_true',
                        help='Run inference only; requires --load_checkpoint.')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                        help='Execution device; cpu is useful when a shared GPU is busy.')
    parser.add_argument('--cells_pkl', type=str, default=None,
                        help='Override cells pickle; use corrected-control data for formal runs.')
    args = parser.parse_args()

    config = load_config(args.config)

    project_dir = config.get('project_dir', '/root/autodl-tmp/Cell-Type-Specific-Graphs-main')
    data_path = config.get('data_path', 'Data/')
    dataset = config.get('dataset_h5ad', 'Kang_processed.h5ad')
    graphs_data = config.get('graphs_data', 'Kang')
    cells_pkl = args.cells_pkl or config.get('cells_pkl', f'cells_{graphs_data}.pkl')
    save_path_results = config.get('save_path_results', 'Results/')
    save_path_models = config.get('save_path_models', 'model_checkpoints/')
    graphs_path = config.get('graphs_path', 'graphs/')
    multi_pert = config.get('multi_pert', False)

    testing_cell_type = args.testing_cell_types or config.get('testing_cell_type', ['CD4 T cells'])
    testing_drugs = args.testing_drugs or config.get('testing_drugs', ['stimulated'])
    seed = int(args.seed if args.seed is not None else config.get('seed', 42))
    node2vec_seed = int(args.node2vec_seed if args.node2vec_seed is not None else seed)
    set_global_seed(seed, deterministic=args.deterministic)

    params = config.get('params', {
        'hidden_channels': 128,
        'weight_decay': 1e-05,
        'in_head': 1,
        'learning_rate': -3,
        'num_epochs': 200,
        'batch_size': 256,
    })
    params = dict(params)
    if args.epochs is not None:
        params['num_epochs'] = args.epochs
    if args.batch_size is not None:
        params['batch_size'] = args.batch_size

    n2v_cfg = config.get('node2vec', {})
    node2vec_dim = args.node2vec_dim or n2v_cfg.get('dim', 64)
    node2vec_mode = args.node2vec_mode or n2v_cfg.get('mode', 'both')
    normalize_method = args.normalize_n2v or n2v_cfg.get('normalize', 'zscore')

    use_expr_stats_str = args.use_expr_stats or n2v_cfg.get('use_expr_stats', 'true')
    if isinstance(use_expr_stats_str, bool):
        use_expr_stats = use_expr_stats_str
    elif isinstance(use_expr_stats_str, str):
        use_expr_stats = use_expr_stats_str.lower() == 'true'
    else:
        use_expr_stats = bool(use_expr_stats_str)

    walk_length = n2v_cfg.get('walk_length', 40)
    num_epochs_n2v = n2v_cfg.get('num_epochs', 10)
    ppi_dir = os.path.join(project_dir, n2v_cfg.get('ppi_dir', f'Data/PPI_data_v2/{graphs_data}'))
    context_size = n2v_cfg.get('context_size', 20)
    walks_per_node = n2v_cfg.get('walks_per_node', 10)
    p_param = n2v_cfg.get('p', 1.0)
    q_param = n2v_cfg.get('q', 1.0)
    lr_n2v = n2v_cfg.get('lr', 0.01)
    batch_size_n2v = n2v_cfg.get('batch_size', 256)

    dynamic_cfg = config.get('dynamic_celltype', {})
    dynamic_scale_init = dynamic_cfg.get('dynamic_scale_init', 0.1)

    readout_cfg = config.get('readout', {})
    readout_mode = args.readout_mode or readout_cfg.get('mode', 'stat')
    readout_dropout = readout_cfg.get('dropout', 0.0)

    ppi_cfg = config.get('ppi', {})
    ppi_enabled = ppi_cfg.get('enabled', False)
    string_links_path = ppi_cfg.get('string_links_path', 'Data/9606.protein.links.v11.5.txt')
    string_info_path = ppi_cfg.get('string_info_path', 'Data/9606.protein.info.v11.5.txt')
    string_alias_path = ppi_cfg.get('string_alias_path', 'Data/9606.protein.aliases.v11.5.txt')
    species_prefix = ppi_cfg.get('species_prefix', '9606.')
    ppi_source = ppi_cfg.get('source', 'string')
    min_score = ppi_cfg.get('min_score', 400)
    ppi_top_quantile = ppi_cfg.get('top_quantile', 0.95)
    fusion_entropy_beta = ppi_cfg.get('fusion_entropy_beta', 0.05)

    model_variant = config.get('model_variant', 'dynamic_dual_n2v')
    if args.model_variant is not None:
        model_variant = args.model_variant
    use_dynamic = True
    if args.ablation_variant is not None:
        ablation_spec = ABLATION_SPECS[args.ablation_variant]
        model_variant = ablation_spec['model_variant']
        ppi_enabled = ablation_spec['ppi_enabled']
        use_dynamic = ablation_spec['use_dynamic']
        node2vec_mode = ablation_spec['node2vec_mode']

    ppi_feature_enabled = node2vec_mode != 'none'

    full_data_path = os.path.join(project_dir, data_path)
    full_dataset_path = os.path.join(full_data_path, dataset)
    if args.run_dir:
        run_dir = os.path.abspath(args.run_dir)
        full_save_path_results = os.path.join(run_dir, 'results')
        full_save_path_models = os.path.join(run_dir, 'models')
    else:
        run_dir = None
        full_save_path_results = os.path.join(project_dir, save_path_results)
        full_save_path_models = os.path.join(project_dir, save_path_models)
    full_graphs_path = os.path.join(project_dir, graphs_path, graphs_data)

    print("=" * 70)
    print("Node2Vec + Dynamic Cell-type Embedding Training")
    print("=" * 70)
    print(f"Model variant:        {model_variant}")
    print(f"Ablation variant:     {args.ablation_variant or 'custom'}")
    print(f"PPI feature enabled:  {ppi_feature_enabled}")
    print(f"Raw PPI graph branch: {ppi_enabled}")
    print(f"Dynamic enabled:      {use_dynamic}")
    print(f"Training seed:        {seed}")
    print(f"Node2Vec seed:        {node2vec_seed}")
    print(f"Dynamic scale init:   {dynamic_scale_init}")
    print(f"Readout mode:         {readout_mode}")
    print(f"Node2Vec dim:         {node2vec_dim}")
    print(f"Node2Vec mode:        {node2vec_mode}")
    print(f"Node2Vec normalize:   {normalize_method}")
    print(f"Expr stats enabled:   {use_expr_stats}")
    if ppi_enabled:
        print(f"PPI source:           {ppi_source}")
        print(f"min_score:            {min_score}")
        print(f"top_quantile:         {ppi_top_quantile}")
        print(f"Entropy beta:         {fusion_entropy_beta}")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Load AnnData
    # -------------------------------------------------------------------------
    print("\n[Step 0] Loading data ...")
    adata = sc.read(full_dataset_path)
    try:
        del adata.raw
    except Exception:
        pass

    adata.obs['cov_drug'] = (
        adata.obs['cell_type'].astype(str) + '_' + adata.obs['condition'].astype(str)
    )
    gene_names = list(adata.var_names)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    n_genes = len(gene_names)
    print(f"  adata: {adata.n_obs} cells x {n_genes} genes")

    # -------------------------------------------------------------------------
    # Load co-expression graphs
    # -------------------------------------------------------------------------
    print("\n[Step 1] Loading coexpression graphs ...")
    cell_type_network = {}
    for ct in adata.obs['cell_type'].unique():
        gp = os.path.join(full_graphs_path, f"{ct}_coexpr_graph.pkl")
        if os.path.exists(gp):
            cell_type_network[ct] = torch.load(gp)
            print(f"  {ct:30s}  {len(cell_type_network[ct].pos)} genes")

    # -------------------------------------------------------------------------
    # Load / build PPI graphs
    # -------------------------------------------------------------------------
    ppi_network = {}
    ppi_global = None
    ppi_per_celltype = None

    if ppi_enabled and load_ppi_network is not None:
        links_full = os.path.join(project_dir, string_links_path)
        info_full = os.path.join(project_dir, string_info_path)
        alias_full = os.path.join(project_dir, string_alias_path) if string_alias_path else None
        if not os.path.exists(links_full):
            links_full = os.path.join(full_data_path, string_links_path)
        if not os.path.exists(info_full):
            info_full = os.path.join(full_data_path, string_info_path)
        if alias_full is not None and not os.path.exists(alias_full):
            alias_full = os.path.join(full_data_path, string_alias_path)

        if os.path.exists(links_full):
            print(f"\n[Step 2] Loading STRING PPI from {links_full} ...")
            ppi_global = load_ppi_network(
                string_links_path=links_full,
                string_info_path=info_full,
                string_alias_path=alias_full,
                species_prefix=species_prefix,
                source=ppi_source,
            )

            print("[Step 2b] Building cell-type-specific PPI subgraphs ...")
            ctrl_data = adata[adata.obs.condition == 'control']
            for cell_type in tqdm(adata.obs.cell_type.unique(), desc="PPI graphs"):
                coexpr_g = cell_type_network[cell_type]
                ppi_network[cell_type] = create_ppi_graph(
                    adata,
                    ppi_global,
                    coexpr_g,
                    cell_type=cell_type,
                    min_score=min_score,
                    top_quantile=ppi_top_quantile,
                    gene_key="gene_name",
                    celltype_key="cell_type",
                )

            for cell_type, ppi_g in ppi_network.items():
                ctrl_subset = ctrl_data[
                    ctrl_data.obs.cell_type == cell_type,
                    cell_type_network[cell_type].pos.tolist()
                ].copy()
                mean_expr = torch.mean(torch.tensor(ctrl_subset.X.A), dim=0)
                var_expr = torch.var(torch.tensor(ctrl_subset.X.A), dim=0)
                ppi_g.x = torch.cat(
                    [mean_expr.unsqueeze(1), var_expr.unsqueeze(1)], dim=1
                ).float()

            print(f"[PPI] Built PPI graphs for {len(ppi_network)} cell types.")

    if ppi_enabled and not ppi_network:
        raise RuntimeError(
            "This run requires the PPI branch, but no cell-type PPI graphs were built. "
            "Check utils_ppi.py and the STRING links/info/alias paths in the config."
        )

    # -------------------------------------------------------------------------
    # Load / build Node2Vec data
    # -------------------------------------------------------------------------
    ppi_global_path = os.path.join(ppi_dir, 'ppi_global.pt')
    ppi_ct_path = os.path.join(ppi_dir, 'ppi_per_celltype', 'all_celltypes.pt')

    if node2vec_mode != 'none' and os.path.exists(ppi_global_path):
        print(f"\n[Step 2c] Loading global PPI from: {ppi_global_path}")
        ppi_global = torch.load(ppi_global_path)
        print(f"  Global PPI edges: {ppi_global['edge_index'].shape[1]:,}")

    if node2vec_mode != 'none' and ppi_global is None:
        raise FileNotFoundError(
            f"PPI feature mode '{node2vec_mode}' requires {ppi_global_path}."
        )

    if node2vec_mode in ('local', 'both'):
        if os.path.exists(ppi_ct_path):
            print(f"\n[Step 2d] Loading per-cell-type PPI from: {ppi_ct_path}")
            ppi_per_celltype = torch.load(ppi_ct_path)
            print(f"  Loaded {len(ppi_per_celltype)} cell-type PPI subgraphs")

    # -------------------------------------------------------------------------
    # Compute PPI topology features
    # -------------------------------------------------------------------------
    topology_features = None
    if node2vec_mode != 'none' and ppi_global is not None:
        print(f"\n[Step 2e] Computing PPI topology features ...")
        edge_index = ppi_global['edge_index']
        if 'edge_weight' in ppi_global:
            G_nx = nx.Graph()
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[0, i].item(), edge_index[1, i].item()
                w = ppi_global['edge_weight'][i].item()
                G_nx.add_edge(src, dst, weight=w)
        else:
            G_nx = nx.from_edgelist(edge_index.t().tolist())

        topology_features = compute_ppi_topology_features(G_nx, verbose=True, node_names=gene_names)

    # -------------------------------------------------------------------------
    # Train / load Node2Vec
    # -------------------------------------------------------------------------
    node2vec_global = None
    node2vec_local = None

    if node2vec_mode in ('global', 'both') and ppi_global is not None:
        cache_path = os.path.join(
            ppi_dir, f'gene_node2vec_{node2vec_dim}d_seed{node2vec_seed}.pt'
        )

        if os.path.exists(cache_path) and not args.overwrite:
            print(f"\n[Step 3a] Loading cached global Node2Vec from: {cache_path}")
            cache = torch.load(cache_path)
            node2vec_global = cache['embeddings']
            print(f"  Loaded embeddings shape: {node2vec_global.shape}")
        else:
            print(f"\n[Step 3a] Training global Node2Vec (dim={node2vec_dim}) ...")
            node2vec_global = train_node2vec_model(
                edge_index=ppi_global['edge_index'], num_nodes=n_genes,
                embedding_dim=node2vec_dim, walk_length=walk_length,
                context_size=context_size, walks_per_node=walks_per_node,
                p=p_param, q=q_param, num_epochs=num_epochs_n2v,
                batch_size=batch_size_n2v, lr=lr_n2v, seed=node2vec_seed,
            )
            norms = node2vec_global.norm(dim=1, keepdim=True)
            norms = torch.clamp(norms, min=1e-8)
            node2vec_global = node2vec_global / norms

            os.makedirs(ppi_dir, exist_ok=True)
            torch.save({
                'embeddings': node2vec_global, 'gene_names': gene_names,
                'dim': node2vec_dim,
                'config': {'walk_length': walk_length, 'context_size': context_size,
                           'walks_per_node': walks_per_node, 'p': p_param, 'q': q_param,
                           'num_epochs': num_epochs_n2v}
            }, cache_path)
            print(f"  Saved to: {cache_path}")

        n2v_np = node2vec_global.numpy()
        print(f"  Embedding stats: min={n2v_np.min():.4f}, max={n2v_np.max():.4f}, "
              f"mean={n2v_np.mean():.4f}, std={n2v_np.std():.4f}")

    if node2vec_mode in ('local', 'both') and ppi_per_celltype is not None:
        cache_path = os.path.join(
            ppi_dir,
            f'gene_node2vec_{node2vec_dim}d_per_celltype_seed{node2vec_seed}.pt',
        )

        if os.path.exists(cache_path) and not args.overwrite:
            print(f"\n[Step 3b] Loading cached local Node2Vec from: {cache_path}")
            cache = torch.load(cache_path)
            node2vec_local = cache['celltype_embeddings']
            print(f"  Loaded {len(node2vec_local)} cell-type embeddings")
        else:
            print(f"\n[Step 3b] Training local Node2Vec (dim={node2vec_dim}) ...")
            node2vec_local = train_node2vec_local(
                ppi_per_celltype=ppi_per_celltype,
                all_gene_names=gene_names,
                embedding_dim=node2vec_dim,
                walk_length=walk_length, context_size=context_size,
                walks_per_node=walks_per_node,
                p=p_param, q=q_param, num_epochs=num_epochs_n2v,
                batch_size=batch_size_n2v, lr=lr_n2v, seed=node2vec_seed,
            )
            os.makedirs(ppi_dir, exist_ok=True)
            torch.save({
                'celltype_embeddings': node2vec_local,
                'gene_names': gene_names, 'dim': node2vec_dim,
            }, cache_path)
            print(f"  Saved to: {cache_path}")

    # -------------------------------------------------------------------------
    # Build augmented node features
    # -------------------------------------------------------------------------
    print(f"\n[Step 4] Building node features (mode={node2vec_mode}) ...")
    ctrl_data = adata[adata.obs['condition'] == 'control']

    build_node_features(
        cell_type_network=cell_type_network,
        ctrl_data=ctrl_data, adata=adata,
        node2vec_global=node2vec_global,
        node2vec_local=node2vec_local,
        normalize=normalize_method, mode=node2vec_mode,
        gene_to_idx=gene_to_idx,
        topology_features=topology_features,
        use_expr_stats=use_expr_stats,
    )
    # Both branches must receive the same node descriptors; only the edge prior
    # changes. This also keeps feature dimensions aligned when Node2Vec is used.
    for cell_type, ppi_graph in ppi_network.items():
        if cell_type not in cell_type_network:
            continue
        coexpr_x = cell_type_network[cell_type].x
        if ppi_graph.x.shape[0] != coexpr_x.shape[0]:
            raise ValueError(
                f"PPI/coexpression node mismatch for {cell_type}: "
                f"{ppi_graph.x.shape[0]} vs {coexpr_x.shape[0]}"
            )
        ppi_graph.x = coexpr_x.clone()

    # -------------------------------------------------------------------------
    # Load cell data
    # -------------------------------------------------------------------------
    print("\n[Step 5] Loading cell data ...")
    cells_path = (
        cells_pkl
        if os.path.isabs(cells_pkl)
        else os.path.join(full_data_path, cells_pkl)
    )
    with open(cells_path, 'rb') as f:
        cell_all_Data = pickle.load(f)

    testing_cov_drug = list(adata[
        adata.obs['condition'].isin(testing_drugs) &
        adata.obs['cell_type'].isin(testing_cell_type)
    ].obs['cov_drug'].unique())

    cells_train, cells_ood = split_samples(
        cell_all_Data, testing_cov_drug, testing_cell_type,
        split_mode=args.split_mode, testing_drugs=testing_drugs,
    )
    print(f"  Train cells: {len(cells_train)}, OOD cells: {len(cells_ood)}")

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    train_dataloader = DataLoader(
        cells_train,
        batch_size=params['batch_size'],
        shuffle=True,
        generator=train_generator,
    )

    # -------------------------------------------------------------------------
    # Initialize model
    # -------------------------------------------------------------------------
    print("\n[Step 6] Initializing model ...")
    torch.manual_seed(seed)

    sample_ct = list(cell_type_network.keys())[0]
    node_feat_dim = cell_type_network[sample_ct].x.shape[1]
    print(f"  Node feature dimension: {node_feat_dim}")

    if args.ablation_variant is not None and not use_expr_stats:
        expected_node_feat_dim = 2
        if node2vec_mode == 'global':
            expected_node_feat_dim += node2vec_dim + 6
        if node_feat_dim != expected_node_feat_dim:
            raise RuntimeError(
                f"Canonical variant '{args.ablation_variant}' expected "
                f"{expected_node_feat_dim} node features but built {node_feat_dim}."
            )

    total_genes = n_genes
    num_perts = 110
    pert_dim = 124

    if model_variant == "dynamic_single_n2v":
        print("[Model] Using DynamicCellTypeGNNNode2Vec (single-branch + Node2Vec + dynamic)")
        model = DynamicCellTypeGNNNode2Vec(
            total_genes=total_genes,
            num_perts=num_perts,
            act_fct='Sigmoid',
            hidden_channels=params['hidden_channels'],
            in_head=params['in_head'],
            multi_pert=multi_pert,
            pert_dim=pert_dim,
            dynamic_scale_init=dynamic_scale_init,
            readout_mode=readout_mode,
            readout_dropout=readout_dropout,
            node_feat_dim=node_feat_dim,
            use_dynamic=use_dynamic,
        )

    elif model_variant == "dynamic_dual_n2v":
        print("[Model] Using DynamicCellTypeDualPriorGNNNode2Vec (dual-branch + Node2Vec + dynamic)")
        model = DynamicCellTypeDualPriorGNNNode2Vec(
            total_genes=total_genes,
            num_perts=num_perts,
            act_fct='Sigmoid',
            hidden_channels=params['hidden_channels'],
            in_head=params['in_head'],
            multi_pert=multi_pert,
            pert_dim=pert_dim,
            fusion_entropy_beta=fusion_entropy_beta,
            dynamic_scale_init=dynamic_scale_init,
            readout_mode=readout_mode,
            readout_dropout=readout_dropout,
            node_feat_dim=node_feat_dim,
            use_dynamic=use_dynamic,
        )

    elif model_variant == "dynamic_dual_n2v_global":
        print("[Model] Using DynamicCellTypeDualPriorGNNNode2VecGlobal (global fusion + Node2Vec + dynamic)")
        model = DynamicCellTypeDualPriorGNNNode2VecGlobal(
            total_genes=total_genes,
            num_perts=num_perts,
            act_fct='Sigmoid',
            hidden_channels=params['hidden_channels'],
            in_head=params['in_head'],
            multi_pert=multi_pert,
            pert_dim=pert_dim,
            dynamic_scale_init=dynamic_scale_init,
            readout_mode=readout_mode,
            readout_dropout=readout_dropout,
            node_feat_dim=node_feat_dim,
            use_dynamic=use_dynamic,
        )

    elif model_variant == "dynamic_dual_n2v_concat":
        print("[Model] Using DynamicCellTypeDualPriorGNNNode2VecConcat (concat fusion + Node2Vec + dynamic)")
        model = DynamicCellTypeDualPriorGNNNode2VecConcat(
            total_genes=total_genes,
            num_perts=num_perts,
            act_fct='Sigmoid',
            hidden_channels=params['hidden_channels'],
            in_head=params['in_head'],
            multi_pert=multi_pert,
            pert_dim=pert_dim,
            dynamic_scale_init=dynamic_scale_init,
            readout_mode=readout_mode,
            readout_dropout=readout_dropout,
            node_feat_dim=node_feat_dim,
            use_dynamic=use_dynamic,
        )

    else:
        raise ValueError(
            f"Unknown model_variant '{model_variant}'. "
            "Use one of: dynamic_single_n2v, dynamic_dual_n2v, dynamic_dual_n2v_global, dynamic_dual_n2v_concat."
        )

    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if args.load_checkpoint:
        checkpoint = torch.load(args.load_checkpoint, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        state_dict, collapsed_duplicate = collapse_duplicate_degree_checkpoint(state_dict, model)
        if collapsed_duplicate:
            print('Collapsed legacy duplicate degree columns: 73-D -> 72-D (function-preserving).')
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded checkpoint: {args.load_checkpoint}")
    if args.skip_training and not args.load_checkpoint:
        raise ValueError('--skip_training requires --load_checkpoint')
    model = model.to(device)

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
    if args.skip_training:
        print("\n[Step 7] Training skipped; checkpoint inference only.")
    else:
        print("\n[Step 7] Training ...")
        model = train_node2vec_dynamic(
            model=model,
            num_epochs=params['num_epochs'],
            lr=10 ** params['learning_rate'],
            weight_decay=params['weight_decay'],
            cell_type_network=cell_type_network,
            ppi_network=ppi_network,
            train_loader=train_dataloader,
            multi_pert=multi_pert,
            entropy_beta=fusion_entropy_beta,
            device=device,
        )

    # -------------------------------------------------------------------------
    # Save model
    # -------------------------------------------------------------------------
    variant_label = args.ablation_variant or model_variant
    variant_tag = f"_seed{seed}_n2v{node2vec_dim}d_{node2vec_mode}_{variant_label}"
    if ppi_enabled:
        variant_tag += "_ppi"
    model_path = Path(full_save_path_models) / f'{graphs_data}{variant_tag}_model.pt'
    Path(full_save_path_models).mkdir(parents=True, exist_ok=True)
    if not args.skip_training:
        torch.save(model.state_dict(), model_path)
        print(f"Model saved: {model_path}")
    else:
        print(f"Using loaded model without overwrite: {args.load_checkpoint}")

    # -------------------------------------------------------------------------
    # DEG analysis
    # -------------------------------------------------------------------------
    print("\n[Step 8] DEG analysis ...")
    degs_dict = {}
    for cov_drug in tqdm(
        set(adata[adata.obs.cov_drug.isin(testing_cov_drug)].obs['cov_drug']),
        desc="DEG analysis"
    ):
        ood_cell = cov_drug.split('_')[0]
        genes = cell_type_network[ood_cell].pos.tolist()

        adata_cov = adata[
            adata.obs.cov_drug.isin([cov_drug, f'{ood_cell}_control']), genes
        ].copy()

        sc.tl.rank_genes_groups(
            adata_cov,
            groupby='cov_drug',
            rankby_abs=True,
            method='t-test',
            corr_method='benjamini-hochberg',
            reference=f'{ood_cell}_control',
            n_genes=len(adata_cov.var),
        )
        dedf = sc.get.rank_genes_groups_df(adata_cov, group=cov_drug)
        dedf = dedf.loc[dedf['pvals_adj'] < 0.05].copy()
        DEGs_name = rank_genes(dedf)

        mask = adata_cov.var_names.isin(DEGs_name)
        degs_dict[cov_drug] = np.where(mask)[0]

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    print("\n[Step 9] Inference ...")
    ood_loader = DataLoader(cells_ood, batch_size=params['batch_size'], shuffle=False)

    save_res_mean = os.path.join(full_save_path_results, f'results{variant_tag}_mean/')
    save_res_std = os.path.join(full_save_path_results, f'results{variant_tag}_std/')

    print("-" * 60)
    print("Mean Expression Prediction")
    print("-" * 60)
    mean_metric_rows = inference_node2vec_dynamic(
        cell_type_network, ppi_network, model, save_res_mean,
        ood_loader, adata, degs_dict,
        device=device, mean_or_std=True, plot=not args.no_plots, multi_pert=multi_pert,
    )

    print("-" * 60)
    print("Std Expression Prediction")
    print("-" * 60)
    std_metric_rows = inference_node2vec_dynamic(
        cell_type_network, ppi_network, model, save_res_std,
        ood_loader, adata, degs_dict,
        device=device, mean_or_std=False, plot=not args.no_plots, multi_pert=multi_pert,
    )

    summary_dir = (
        Path(run_dir)
        if run_dir is not None
        else Path(full_save_path_results) / f"run{variant_tag}"
    )
    summary_dir.mkdir(parents=True, exist_ok=True)
    if args.collect_dynamic_mechanisms:
        if not use_dynamic:
            raise ValueError('--collect_dynamic_mechanisms requires a dynamic model')
        from dynamic_mechanism_validation import collect_mechanism_bundle
        mechanism_path = collect_mechanism_bundle(
            model=model, ood_loader=ood_loader,
            cell_type_network=cell_type_network, ppi_network=ppi_network,
            adata=adata,
            output_npz=summary_dir / 'dynamic_mechanism_bundle.npz',
            device=device, n_swaps=args.mechanism_swaps, seed=seed,
        )
        print(f'Dynamic mechanism bundle: {mechanism_path}')
    metric_rows = mean_metric_rows + std_metric_rows
    for row in metric_rows:
        row.update({
            "ablation_protocol": ABLATION_PROTOCOL,
            "dataset": graphs_data,
            "seed": seed,
            "node2vec_seed": node2vec_seed,
            "ablation_variant": args.ablation_variant or "custom",
            "model_variant": model_variant,
            "ppi_enabled": bool(ppi_enabled),
            "ppi_feature_enabled": bool(ppi_feature_enabled),
            "dynamic_enabled": bool(use_dynamic),
            "node2vec_mode": node2vec_mode,
        })

    metrics_path = summary_dir / "run_metrics.csv"
    if metric_rows:
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metric_rows)

    manifest = {
        "status": "complete",
        "native_variant": "full" if args.ablation_variant is None else args.ablation_variant,
        "native_script": "training_node2vec_dynamic.py",
        "ablation_protocol": ABLATION_PROTOCOL,
        "dataset": graphs_data,
        "config": os.path.abspath(args.config),
        "cells_pkl": os.path.abspath(cells_path),
        "cells_pkl_sha256": file_sha256(cells_path),
        "seed": seed,
        "node2vec_seed": node2vec_seed,
        "ablation_variant": args.ablation_variant,
        "model_variant": model_variant,
        "ppi_enabled": bool(ppi_enabled),
        "ppi_feature_enabled": bool(ppi_feature_enabled),
        "dynamic_enabled": bool(use_dynamic),
        "node2vec_mode": node2vec_mode,
        "node2vec_dim": int(node2vec_dim),
        "node_feature_dim": int(node_feat_dim),
        "ppi_topology_dim": 6 if ppi_feature_enabled else 0,
        "normalize_n2v": normalize_method,
        "use_expr_stats": bool(use_expr_stats),
        "readout_mode": readout_mode,
        "testing_cell_types": list(testing_cell_type),
        "testing_drugs": list(testing_drugs),
        "split_mode": args.split_mode,
        "train_samples": int(len(cells_train)),
        "ood_samples": int(len(cells_ood)),
        "train_cell_types": sorted({sample_cell_type(s) for s in cells_train}),
        "train_drugs": sorted({sample_drug(s) for s in cells_train}),
        "num_epochs": int(params["num_epochs"]),
        "batch_size": int(params["batch_size"]),
        "metrics_file": str(metrics_path),
        "model_file": str(model_path),
    }
    with (summary_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("DONE")
    print(f"  Results: {save_res_mean}")
    print(f"  Model:   {model_path}")
    print(f"  Node2Vec: dim={node2vec_dim}, mode={node2vec_mode}")
    print(f"  Node features: {node_feat_dim} dims")
    print(f"  Dynamic cell-type: {model_variant}")
    print(f"  Metrics: {metrics_path}")
    print("=" * 70)
