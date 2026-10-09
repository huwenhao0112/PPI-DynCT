#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_ppi_network.py
========================

STRING PPI 网络预处理 

  步骤一：ENSP ID → Gene Symbol 映射
    STRING 的 protein ID 通常是 9606.ENSP00000354587 格式，
    需要映射到表达矩阵中的 gene symbol（TP53）。

  步骤二：按置信度筛选高置信 PPI 边
    combined_score 范围 0–1000，建议三档阈值：
      400 — 中等置信度（消融实验）
      700 — 高置信度（主实验，推荐）
      900 — 极高置信度（消融实验）
    注意：阈值太低会导致图过于稠密，稀释细胞类型特异性。

  步骤三：针对每个细胞类型提取 PPI 子图
    对细胞类型 c：
      V_c = 该细胞类型共表达图里的基因集合（不引入新基因！）
      E_ppi^c = {(i,j) | i ∈ V_c, j ∈ V_c, STRING_score(i,j) ≥ threshold}
    边权：w_ppi(i,j) = combined_score(i,j) / 1000 （归一化到 0-1）

输入文件：
  9606.protein.links.v11.5.txt   — STRING PPI 边（含 combined_score）
  9606.protein.aliases.v11.5.txt — ENSP → gene symbol 映射（离线，无需 mygene）
  9606.protein.info.v11.5.txt   — protein 元信息（可选）

输出文件：
  Data/PPI_data_v2/{dataset}/
    ensp_to_symbol.pkl       — ENSP → gene symbol 映射
    gene_to_idx.pkl          — gene symbol → index
    ppi_global_edge_index.pt  — 全局 PPI 边（所有基因）
    ppi_global_weight.pt      — 全局 PPI 权重（归一化 combined_score）
    ppi_per_celltype/         — 每个细胞类型的 PPI 子图
      {celltype}_ppi.pt      — {'edge_index': LongTensor, 'weight': FloatTensor}

用法：
  python build_ppi_network_v2.py --dataset Kang --threshold 700
  python Data_Notebooks/build_ppi_network_v2.py --dataset Nault --threshold 700
  python Data_Notebooks/build_ppi_network_v2.py --dataset McFarland --threshold 700
"""

import os
import sys
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import scanpy as sc
from torch_geometric.utils import to_undirected

warnings.filterwarnings('ignore')


# =============================================================================
# STEP 1: ENSP → Gene Symbol 映射
# =============================================================================

def load_info_preferred_name(info_file, species_prefix='9606.'):
    """
    从 STRING protein.info 文件提取 ENSP → preferred_name（基因 symbol）映射。

    info 文件格式（tab 分隔，含表头）：
        #string_protein_id  preferred_name  protein_size  annotation
        9606.ENSP00000000233  ARF5  180  ADP-ribosylation factor ...

    preferred_name 通常是官方基因 symbol，覆盖率远高于 aliases。
    """
    print(f"[PPI] Loading preferred_name from: {info_file}")
    if not os.path.exists(info_file):
        print(f"[PPI] WARNING: {info_file} not found")
        return {}

    ensp_to_symbol = {}
    with open(info_file, 'r') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            string_id = parts[0]
            preferred = parts[1].strip()
            if not string_id.startswith(species_prefix):
                continue
            if not preferred:
                continue
            ensp = string_id.replace(species_prefix, '')
            ensp_to_symbol[ensp] = preferred

    print(f"[PPI] Loaded {len(ensp_to_symbol):,} ENSP → preferred_name mappings")
    return ensp_to_symbol


def load_aliases_mapping(aliases_file, species_prefix='9606.'):
    """
    从 STRING protein.aliases 文件建立 ENSP → gene symbol 映射。

    aliases 文件格式（无表头）：
      STRING_id   alias   source
      9606.ENSP00000354587  TP53   stringdb

    策略：
      1. 优先用 stringdb 来源的映射（最可靠）
      2. 如果没有 stringdb，用任何可用映射
      3. 一个 ENSP 可能对应多个 gene symbol → 只保留第一个
    """
    print(f"[PPI] Loading aliases from: {aliases_file}")
    ensp_to_symbol = {}    # ensp_id → gene_symbol (best match)
    symbol_to_ensp = {}    # gene_symbol → set of ensp_ids

    if not os.path.exists(aliases_file):
        print(f"[PPI] WARNING: {aliases_file} not found")
        return None

    with open(aliases_file, 'r') as f:
        header = f.readline()   # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            string_id = parts[0]
            alias     = parts[1]
            source    = parts[2] if len(parts) > 2 else ''

            if not string_id.startswith(species_prefix):
                continue
            ensp = string_id.replace(species_prefix, '')

            # stringdb 来源最可靠
            if source == 'stringdb':
                ensp_to_symbol[ensp] = alias
            elif ensp not in ensp_to_symbol:
                # 如果没有 stringdb 映射，先用第一个可用映射
                ensp_to_symbol[ensp] = alias

            if alias not in symbol_to_ensp:
                symbol_to_ensp[alias] = set()
            symbol_to_ensp[alias].add(ensp)

    print(f"[PPI] Loaded {len(ensp_to_symbol):,} ENSP → alias mappings")
    return ensp_to_symbol, symbol_to_ensp


def merge_ensp_mappings(preferred_map, aliases_map):
    """
    合并 preferred_name 和 aliases 映射，preferred 优先。

    Returns:
        merged: dict[ensp] -> symbol（优先用 preferred，fallback 到 aliases）
        symbol_to_ensp: dict[symbol] -> set of ensp ids
    """
    merged = {}

    # Step 1: 先加入 aliases（低优先级）
    if aliases_map:
        for ensp, alias in aliases_map.items():
            if ensp not in merged:
                merged[ensp] = alias

    # Step 2: 用 preferred_name 覆盖（高优先级）
    for ensp, symbol in preferred_map.items():
        merged[ensp] = symbol

    # Step 3: 建立反向索引
    symbol_to_ensp = {}
    for ensp, symbol in merged.items():
        if symbol not in symbol_to_ensp:
            symbol_to_ensp[symbol] = set()
        symbol_to_ensp[symbol].add(ensp)

    return merged, symbol_to_ensp


def map_with_mygene_fallback(all_ensps, gene_symbols):
    """
    备用：使用 mygene 在线查询 ENSP → gene symbol。
    只有在 aliases 文件缺失时调用。
    """
    print("[PPI] Using mygene online query (aliases file not available)...")
    try:
        import mygene
    except ImportError:
        print("[PPI] Installing mygene...")
        os.system("pip install mygene -q")
        import mygene

    mg = mygene.MyGeneInfo()
    ensp_list = list(all_ensps)
    batch_size = 1000
    all_mappings = {}

    for i in range(0, len(ensp_list), batch_size):
        batch = ensp_list[i:i + batch_size]
        try:
            results = mg.querymany(
                batch, scopes='ensembl.protein', fields='symbol',
                species='human', verbose=False
            )
            for r in results:
                if 'symbol' in r:
                    sym = r['symbol']
                    if isinstance(sym, list):
                        sym = sym[0]
                    if sym and sym in gene_symbols:
                        all_mappings[r['query']] = sym
        except Exception as e:
            print(f"[PPI] Batch {i // batch_size} failed: {e}")
            continue

    print(f"[PPI] mygene mapped {len(all_mappings):,} ENSPs to gene symbols")
    return all_mappings


# =============================================================================
# STEP 2: 加载并筛选 STRING PPI
# =============================================================================

def load_string_ppi(ppi_file, threshold=700, species_prefix='9606.'):
    """
    加载 STRING protein.links 文件，按 combined_score 阈值筛选。

    文件格式（无表头）：
      protein1   protein2   combined_score
      9606.ENSP00000354587  9606.ENSP00000415872  999

    combined_score: 0–1000（越高越可信）
      ≥ 900: 极高置信（物理互作）
      ≥ 700: 高置信（功能互作，主实验推荐）
      ≥ 400: 中等置信（包含功能关联）
    """
    print(f"[PPI] Loading STRING PPI from: {ppi_file}")
    print(f"[PPI] Confidence threshold: {threshold} (combined_score ≥ {threshold})")

    # 检测文件格式（可能有不同分隔符）
    try:
        df = pd.read_csv(ppi_file, sep=' ')
    except Exception:
        try:
            df = pd.read_csv(ppi_file, sep='\t')
        except Exception:
            # 尝试自动检测
            with open(ppi_file, 'r') as f:
                first = f.readline()
            sep = '\t' if '\t' in first else ' '
            df = pd.read_csv(ppi_file, sep=sep)

    print(f"[PPI] Total PPI pairs in STRING: {len(df):,}")
    print(f"[PPI] Columns: {df.columns.tolist()}")

    # 筛选物种
    if 'protein1' in df.columns:
        df = df[df['protein1'].str.startswith(species_prefix) &
                 df['protein2'].str.startswith(species_prefix)]
        print(f"[PPI] After species filter (9606.): {len(df):,}")

    # 筛选置信度
    if 'combined_score' in df.columns:
        df = df[df['combined_score'] >= threshold]
    else:
        # 尝试其他可能的列名
        score_cols = [c for c in df.columns if 'score' in c.lower() or 'combined' in c.lower()]
        if score_cols:
            df = df[df[score_cols[0]] >= threshold]
        else:
            print("[PPI] WARNING: No score column found — keeping all edges")

    print(f"[PPI] After threshold filter (≥{threshold}): {len(df):,}")

    # 归一化权重到 0-1
    if 'combined_score' in df.columns:
        df['ppi_weight'] = df['combined_score'] / 1000.0
    elif 'score' in str(df.columns).lower():
        score_col = [c for c in df.columns if 'score' in c.lower()][0]
        df['ppi_weight'] = df[score_col] / 1000.0
    else:
        df['ppi_weight'] = 0.5  # 默认中等权重

    return df


# =============================================================================
# STEP 3: 构建全局 PPI edge_index（与 adata 对齐）
# =============================================================================

def build_global_ppi_edges(ppi_df, adata, ensp_to_symbol, species_prefix='9606.'):
    """
    构建全局 PPI 边索引（所有基因）。

    Returns:
        ppi_edge_index: LongTensor (2, E)
        ppi_weights:    FloatTensor (E,)  归一化权重
    """
    gene_list = adata.var_names.tolist()
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}

    # 建立 ENSP → gene_idx 映射
    ensp_to_idx = {}
    for ensp, symbol in ensp_to_symbol.items():
        if symbol in gene_to_idx:
            ensp_to_idx[ensp] = gene_to_idx[symbol]

    print(f"[PPI] ENSPs mapped to adata genes: {len(ensp_to_idx):,}")

    src, dst, weights = [], [], []
    for _, row in ppi_df.iterrows():
        e1 = row['protein1'].replace(species_prefix, '')
        e2 = row['protein2'].replace(species_prefix, '')

        i1 = ensp_to_idx.get(e1)
        i2 = ensp_to_idx.get(e2)

        if i1 is not None and i2 is not None and i1 != i2:
            src.append(i1)
            dst.append(i2)
            weights.append(row['ppi_weight'])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    ppi_weights = torch.tensor(weights, dtype=torch.float32)

    # 转为无向图（去重 + 对称化）
    edge_index, ppi_weights = to_undirected(edge_index, ppi_weights)
    # 去除自环
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    ppi_weights = ppi_weights[mask]

    print(f"[PPI] Global PPI edges (undirected, no self-loops): {edge_index.shape[1]:,}")
    if ppi_weights.numel() > 0:
        print(f"[PPI] PPI weight range: [{ppi_weights.min():.3f}, {ppi_weights.max():.3f}]")
    else:
        print("[PPI] PPI weight range: N/A (no edges)")

    return edge_index, ppi_weights, gene_to_idx, ensp_to_idx


# =============================================================================
# STEP 4: 针对每个细胞类型构建 PPI 子图
# =============================================================================

def build_celltype_ppi_subgraphs(
    ppi_edge_index, ppi_weights,
    cell_type_network, adata, gene_to_idx,
    save_dir=None
):
    """
    针对每个细胞类型，提取该细胞类型基因集合 V_c 上的 PPI 子图。

    设计原则：
      - 节点集合 V_c = 原论文该细胞类型共表达图中的基因
        （不引入表达矩阵以外的基因，保证公平对比）
      - E_ppi^c = { (i,j) | i,j ∈ V_c, (i,j) ∈ global_PPI }

    Args:
        ppi_edge_index:   全局 PPI 边索引
        ppi_weights:     全局 PPI 权重（与 edge_index 对应）
        cell_type_network: dict[ct] -> Data with .pos (gene indices)
        adata:            AnnData for gene name list
        gene_to_idx:      dict{gene_name -> index in adata}
        save_dir:         可选：保存每个子图

    Returns:
        ppi_subgraph_dict: dict[ct] -> {'edge_index': LongTensor, 'weight': FloatTensor}
    """
    print("\n[PPI] Building cell-type-specific PPI subgraphs ...")
    adata_gene_names = adata.var_names.tolist()

    ppi_subgraph_dict = {}

    for ct, graph_data in cell_type_network.items():
        pos = graph_data.pos
        pos_list = pos.tolist() if hasattr(pos, 'tolist') else list(pos)
        local_genes = [adata_gene_names[i] for i in pos_list]
        local_gene_to_idx = {g: i for i, g in enumerate(local_genes)}

        # 全局索引 → 局部索引
        global_to_local = {}
        for local_i, gene in enumerate(local_genes):
            g_idx = gene_to_idx.get(gene, -1)
            if g_idx >= 0:
                global_to_local[g_idx] = local_i

        src, dst = ppi_edge_index
        local_src, local_dst, local_w = [], [], []

        for edge_idx in range(ppi_edge_index.size(1)):
            i_global = src[edge_idx].item()
            j_global = dst[edge_idx].item()
            w = ppi_weights[edge_idx].item()

            i_local = global_to_local.get(i_global)
            j_local = global_to_local.get(j_global)

            if i_local is not None and j_local is not None:
                local_src.append(i_local)
                local_dst.append(j_local)
                local_w.append(w)

        dev = ppi_edge_index.device
        if len(local_src) > 0:
            ei = torch.tensor([local_src, local_dst], dtype=torch.long, device=dev)
            wt = torch.tensor(local_w, dtype=torch.float32, device=dev)
        else:
            ei = torch.empty((2, 0), dtype=torch.long, device=dev)
            wt = torch.tensor([], dtype=torch.float32, device=dev)

        ppi_subgraph_dict[ct] = {'edge_index': ei, 'weight': wt}

        if save_dir:
            os.makedirs(os.path.join(save_dir, 'ppi_per_celltype'), exist_ok=True)
            torch.save({'edge_index': ei.cpu(), 'weight': wt.cpu()},
                        os.path.join(save_dir, 'ppi_per_celltype', f'{ct}_ppi.pt'))

        print(f"  {ct:30s}  V_c={len(local_genes):4d} genes  "
              f"E_ppi={ei.shape[1]:5d} edges  "
              f"avg_weight={wt.mean().item():.3f}" if len(wt) > 0 else f"  {ct:30s}  V_c={len(local_genes):4d} genes  E_ppi=0")

    return ppi_subgraph_dict


# =============================================================================
# STEP 5: 计算每个细胞的节点度（用于可选的增强特征）
# =============================================================================

def compute_ppi_degrees(ppi_edge_index, num_genes):
    """计算每个基因的 PPI degree（连接数）。"""
    degree = torch.zeros(num_genes, dtype=torch.float32)
    src, dst = ppi_edge_index
    degree.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float32))
    degree.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
    return degree


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Build STRING PPI network for dual-branch GAT')
    parser.add_argument('--dataset', type=str, default='Kang',
                        choices=['Kang', 'McFarland', 'Chang', 'Nault', 'NeurIPS', 'Norman', 'Replogle'],
                        help='Dataset name')
    parser.add_argument('--threshold', type=int, default=700,
                        help='STRING combined_score threshold (default: 700)')
    parser.add_argument('--string_dir', type=str, default=None,
                        help='Directory containing STRING files')
    parser.add_argument('--species', type=str, default=None,
                        help='Species prefix (e.g., 9606 for human, 10090 for mouse). Auto-detected from dataset if not provided.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output files')
    args = parser.parse_args()

    project_dir = '/root/autodl-tmp/Cell-Type-Specific-Graphs-main'
    string_dir = args.string_dir or project_dir
    save_dir   = os.path.join(project_dir, 'Data', 'PPI_data_v2', args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'ppi_per_celltype'), exist_ok=True)

    # ---- 物种映射：数据集 → NCBI taxonomy ID ----
    # 9606 = Human, 10090 = Mouse
    species_map = {
        'Kang':      '9606',
        'McFarland': '9606',
        'Chang':     '9606',
        'Norman':    '9606',
        'Replogle':  '9606',
        'Nault':     '10090',   # 小鼠
        'NeurIPS':   '9606',
    }

    species_prefix = args.species or species_map.get(args.dataset, '9606')
    print(f"[PPI] Species: {species_prefix} ({'Human' if species_prefix == '9606' else 'Mouse'})")

    # ---- 确定数据集路径 ----
    dataset_map = {
        'Kang':      ('Data/Kang_processed.h5ad',         'graphs/Kang/',       'Data/cells_Kang.pkl'),
        'McFarland': ('Data/McFarland_processed.h5ad',    'graphs/McFarland/', 'Data/cells_McFarland.pkl'),
        'Chang':     ('Data/Chang_processed.h5ad',        'graphs/Chang/',      'Data/cells_Chang.pkl'),
        'Nault':     ('Data/Nault_processed.h5ad',        'graphs/Nault/',      'Data/cells_Nault.pkl'),
        'NeurIPS':   ('Data/NeurIPS_processed.h5ad',      'graphs/NeurIPS/',    'Data/cells_NeurIPS.pkl'),
    }

    if args.dataset not in dataset_map:
        print(f"[PPI] Dataset {args.dataset} not configured — please add manually")
        return

    adata_path, graphs_dir, _ = dataset_map[args.dataset]
    adata_path  = os.path.join(project_dir, adata_path)
    graphs_dir  = os.path.join(project_dir, graphs_dir)

    # ---- STRING 文件路径 ----
    ppi_file     = os.path.join(string_dir, f'{species_prefix}.protein.links.v11.5.txt')
    aliases_file = os.path.join(string_dir, f'{species_prefix}.protein.aliases.v11.5.txt')
    info_file    = os.path.join(string_dir, f'{species_prefix}.protein.info.v11.5.txt')

    print("=" * 60)
    print(f"PPI Network Builder — Dataset: {args.dataset}")
    print(f"  Species: {species_prefix} ({'Human' if species_prefix == '9606' else 'Mouse'})")
    print(f"  STRING directory: {string_dir}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Save directory: {save_dir}")
    print("=" * 60)

    # ---- Step 0: 加载 adata 和共表达图 ----
    print("\n[Step 0] Loading adata and coexpression graphs ...")
    adata = sc.read(adata_path)
    print(f"  adata: {adata.n_obs} cells × {adata.n_vars} genes")

    cell_type_network = {}
    for ct in adata.obs['cell_type'].unique():
        gp = os.path.join(graphs_dir, f"{ct}_coexpr_graph.pkl")
        if os.path.exists(gp):
            cell_type_network[ct] = torch.load(gp)
            print(f"  {ct}: {len(cell_type_network[ct].pos)} genes")

    # ---- Step 1: ENSP → Gene Symbol 映射（合并 preferred_name + aliases）----
    print("\n[Step 1] ENSP → Gene Symbol mapping ...")
    preferred_map = load_info_preferred_name(info_file, species_prefix=species_prefix)
    aliases_result = load_aliases_mapping(aliases_file, species_prefix=species_prefix)
    if aliases_result is not None:
        _, aliases_symbol_map = aliases_result
        ensp_to_symbol, symbol_to_ensp = merge_ensp_mappings(preferred_map, aliases_result[0])
    else:
        aliases_symbol_map = {}
        ensp_to_symbol = preferred_map

    # 统计映射覆盖率
    adata_gene_set = set(adata.var_names)
    mapped_genes = set(ensp_to_symbol.values()) & adata_gene_set
    print(f"  ENSP→symbol 映射覆盖 adata 基因: {len(mapped_genes)}/{adata.n_vars} ({len(mapped_genes)/adata.n_vars*100:.1f}%)")
    if len(mapped_genes) < adata.n_vars * 0.5:
        print(f"  [WARNING] 覆盖率偏低，建议检查 STRING 文件或使用 mygene 在线映射")

    # ---- Step 2: 加载 STRING PPI 并筛选 ----
    print(f"\n[Step 2] Loading STRING PPI (threshold={args.threshold}) ...")
    ppi_df = load_string_ppi(ppi_file, threshold=args.threshold, species_prefix=species_prefix)

    # ---- Step 3: 构建全局 PPI edge_index ----
    print("\n[Step 3] Building global PPI edge index ...")
    ppi_edge_index, ppi_weights, gene_to_idx, ensp_to_idx = build_global_ppi_edges(
        ppi_df, adata, ensp_to_symbol, species_prefix=species_prefix
    )

    # ---- Step 4: 构建每个细胞类型的 PPI 子图 ----
    print("\n[Step 4] Building cell-type-specific PPI subgraphs ...")
    ppi_subgraph_dict = build_celltype_ppi_subgraphs(
        ppi_edge_index, ppi_weights,
        cell_type_network, adata, gene_to_idx,
        save_dir=save_dir
    )

    # ---- Step 5: 计算 PPI degree（用于可选增强特征）----
    print("\n[Step 5] Computing PPI degrees ...")
    ppi_degree = compute_ppi_degrees(ppi_edge_index, adata.n_vars)
    print(f"  PPI degree range: [{ppi_degree.min():.0f}, {ppi_degree.max():.0f}], mean={ppi_degree.mean():.1f}")

    # ---- Step 6: 保存 ----
    print("\n[Step 6] Saving processed PPI data ...")

    with open(os.path.join(save_dir, 'ensp_to_symbol.pkl'), 'wb') as f:
        pickle.dump(ensp_to_symbol, f)

    with open(os.path.join(save_dir, 'gene_to_idx.pkl'), 'wb') as f:
        pickle.dump(gene_to_idx, f)

    torch.save({
        'edge_index': ppi_edge_index.cpu(),
        'weight':     ppi_weights.cpu(),
    }, os.path.join(save_dir, 'ppi_global.pt'))

    # 合并所有细胞类型子图
    torch.save({
        ct: {
            'edge_index': v['edge_index'].cpu(),
            'weight':     v['weight'].cpu() if v['weight'].numel() > 0 else torch.tensor([]),
        }
        for ct, v in ppi_subgraph_dict.items()
    }, os.path.join(save_dir, 'ppi_per_celltype', 'all_celltypes.pt'))

    # PPI degree
    torch.save({'ppi_degree': ppi_degree.cpu()},
                os.path.join(save_dir, 'ppi_degree.pt'))

    # 记录元信息
    meta = {
        'dataset':         args.dataset,
        'species':         species_prefix,
        'threshold':       args.threshold,
        'n_global_edges':  int(ppi_edge_index.shape[1]),
        'n_genes_mapped':  len(ensp_to_idx),
        'n_celltypes':     len(ppi_subgraph_dict),
        'ppi_weight_range': [float(ppi_weights.min()), float(ppi_weights.max())],
        'ppi_degree_stats': {
            'min': float(ppi_degree.min()),
            'max': float(ppi_degree.max()),
            'mean': float(ppi_degree.mean()),
        }
    }

    print(f"\n[PPI] === Summary ===")
    print(f"  Dataset:           {args.dataset}")
    print(f"  Species:           {species_prefix} ({'Human' if species_prefix == '9606' else 'Mouse'})")
    print(f"  Confidence:       {args.threshold}")
    print(f"  Global PPI edges: {meta['n_global_edges']:,}")
    print(f"  Genes in PPI:     {meta['n_genes_mapped']:,}")
    print(f"  Cell types:       {meta['n_celltypes']}")
    print(f"  PPI weight range: {meta['ppi_weight_range']}")
    print(f"\n[PPI] Saved to: {save_dir}")
    print(f"  ensp_to_symbol.pkl")
    print(f"  gene_to_idx.pkl")
    print(f"  ppi_global.pt")
    print(f"  ppi_per_celltype/all_celltypes.pt")
    print(f"  ppi_degree.pt")


if __name__ == '__main__':
    main()
