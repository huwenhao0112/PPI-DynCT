"""
model_node2vec_dynamic.py
=========================
Unified model that combines:
  1. Node2Vec-augmented node features (from training_node2vec_baseline.py)
  2. Perturbation-conditioned Dynamic Cell-type Embedding (from model_dynamic_statreadout.py)
  3. Dual-branch GNN with PPI prior (co-expression + PPI)
  4. Statistical readout (max + mean + std pooling)

Supported model variants:
  - dynamic_single_n2v      : Single-branch GNN + Node2Vec features + Dynamic cell-type embedding
  - dynamic_dual_n2v        : Dual-branch GNN (coexpr + PPI) + Node2Vec features + Dynamic cell-type (recommended)
  - dynamic_dual_n2v_global : Dual-branch GNN (global scalar fusion) + Node2Vec + Dynamic cell-type (ablation A)
  - dynamic_dual_n2v_concat : Dual-branch GNN (concat fusion) + Node2Vec + Dynamic cell-type (ablation B)
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATConv, Linear, TransformerConv
from torch_geometric.typing import Adj, OptTensor

from model import MLP


# ============================================================================
#  Statistical Readout Blocks
# ============================================================================

class StatisticalReadout(nn.Module):
    """
    Graph readout that summarises node embeddings with multiple statistics.

    Given node embeddings ``h_nodes`` of shape ``[num_genes, hidden_dim]``,
    this module computes max, mean and standard deviation over the node
    dimension, concatenates them, and projects back to ``hidden_dim``.

    Output shape: ``[1, hidden_dim]``.
    """

    def __init__(self, hidden_channels: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.project = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
            nn.Dropout(dropout),
        )

    def forward(self, h_nodes: Tensor) -> Tensor:
        h_max = torch.max(h_nodes, dim=0, keepdim=True)[0]
        h_mean = torch.mean(h_nodes, dim=0, keepdim=True)
        h_std = torch.std(h_nodes, dim=0, keepdim=True)
        h_cat = torch.cat([h_max, h_mean, h_std], dim=-1)
        return self.project(h_cat)


class ResidualStatisticalReadout(nn.Module):
    """
    Residual graph readout that keeps the max-pooled signal and adds a learned
    statistics-based residual.
    """

    def __init__(self, hidden_channels: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.ln_max = nn.LayerNorm(hidden_channels)
        self.stat_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
            nn.Dropout(dropout),
        )
        self.beta_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(self, h_nodes: Tensor) -> Tensor:
        h_max = torch.max(h_nodes, dim=0, keepdim=True)[0]
        h_mean = torch.mean(h_nodes, dim=0, keepdim=True)
        h_std = torch.std(h_nodes, dim=0, keepdim=True)

        h_max_proj = self.ln_max(h_max)
        h_stat = self.stat_mlp(torch.cat([h_mean, h_std], dim=-1))

        beta = 0.2 * torch.sigmoid(self.beta_logit)
        return h_max_proj + beta * h_stat


# ============================================================================
#  Dynamic Cell-type Modulator
# ============================================================================

class DynamicCellTypeModulator(nn.Module):
    """
    Perturbation-conditioned dynamic residual on the static cell-type embedding.

    Given:
        h_static : (B, D)   static cell-type embedding from GNN
        e_p      : (B, D)   perturbation context embedding

    The modulator computes:
        z      = concat(h_static, e_p)
        Δh     = tanh(MLP_delta(z))
        γ      = sigmoid(MLP_gate(z))
        h_dyn  = h_static + dynamic_scale * γ * Δh
    """

    def __init__(self, hidden_channels: int, dynamic_scale_init: float = 0.1):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.mlp_delta = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.Tanh(),
        )

        self.mlp_gate = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.Sigmoid(),
        )

        self._dynamic_scale_raw = nn.Parameter(torch.tensor(dynamic_scale_init))

    @property
    def dynamic_scale(self) -> Tensor:
        return F.softplus(self._dynamic_scale_raw)

    def forward(
        self,
        h_static: Tensor,
        e_p: Tensor,
        return_components: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        if e_p.dim() == 1:
            e_p = e_p.unsqueeze(0)
        if e_p.size(0) != h_static.size(0):
            e_p = e_p.expand(h_static.size(0), -1)

        z = torch.cat([h_static, e_p], dim=-1)

        delta_h = self.mlp_delta(z)
        gamma = self.mlp_gate(z)

        residual = self.dynamic_scale * gamma * delta_h
        h_dynamic = h_static + residual
        if return_components:
            return h_dynamic, {
                "gamma": gamma,
                "delta_h": delta_h,
                "dynamic_residual": residual,
                "dynamic_scale": self.dynamic_scale.reshape(1),
                "pert_context": e_p,
            }
        return h_dynamic


class CapacityMatchedMLPModulator(nn.Module):
    """Ungated context MLP used as a capacity-matched mechanism control.

    The dynamic gate contains ``6D^2 + 6D + 1`` trainable parameters.  This
    control contains ``6D^2 + 7D`` parameters (a relative difference below
    0.3% for the hidden sizes used in this repository), but it has neither a
    multiplicative gate nor a residual connection to ``h_static``::

        h_context = tanh(MLP(concat(h_static, e_p)))

    Matching the input information and almost exactly matching capacity makes
    it possible to distinguish the gated-residual inductive bias from the
    effect of merely adding another neural-network block.
    """

    def __init__(self, hidden_channels: int):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels * 2),
            nn.LayerNorm(hidden_channels * 2),
            nn.ReLU(),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.Tanh(),
        )

    def forward(
        self,
        h_static: Tensor,
        e_p: Tensor,
        return_components: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        if e_p.dim() == 1:
            e_p = e_p.unsqueeze(0)
        if e_p.size(0) != h_static.size(0):
            e_p = e_p.expand(h_static.size(0), -1)
        h_context = self.mlp(torch.cat([h_static, e_p], dim=-1))
        if return_components:
            unavailable = torch.full_like(h_context, torch.nan)
            return h_context, {
                "gamma": unavailable,
                "delta_h": unavailable,
                "dynamic_residual": h_context - h_static,
                "dynamic_scale": torch.full(
                    (1,), torch.nan, device=h_static.device, dtype=h_static.dtype,
                ),
                "pert_context": e_p,
            }
        return h_context


def modulator_parameter_count(hidden_channels: int, kind: str) -> int:
    """Return the analytical parameter count for a context modulator."""
    d = int(hidden_channels)
    if kind == "dynamic":
        return 6 * d * d + 6 * d + 1
    if kind == "capacity_mlp":
        return 6 * d * d + 7 * d
    if kind == "static":
        return 0
    raise ValueError(f"Unknown modulator kind: {kind!r}")


def build_context_modulator(
    kind: str, hidden_channels: int, dynamic_scale_init: float,
) -> Optional[nn.Module]:
    if kind == "static":
        return None
    if kind == "dynamic":
        return DynamicCellTypeModulator(hidden_channels, dynamic_scale_init)
    if kind == "capacity_mlp":
        return CapacityMatchedMLPModulator(hidden_channels)
    raise ValueError(
        f"Unknown modulator_type={kind!r}; expected dynamic, static, or capacity_mlp"
    )


def apply_dynamic_modulator(
    module: nn.Module,
    h_static: Tensor,
    e_p: Tensor,
    disable_dynamic: bool = False,
    return_diagnostics: bool = False,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Apply the dynamic branch with an inference-time knockout and diagnostics."""
    if getattr(module, "modulator_type", "dynamic") == "static" or disable_dynamic:
        zeros = torch.zeros_like(h_static)
        context = e_p
        if context.dim() == 1:
            context = context.unsqueeze(0)
        if context.size(0) != h_static.size(0):
            context = context.expand(h_static.size(0), -1)
        diagnostics = {
            "gamma": zeros,
            "delta_h": zeros,
            "dynamic_residual": zeros,
            "dynamic_scale": torch.zeros(1, device=h_static.device),
            "pert_context": context,
        }
        return h_static, diagnostics if return_diagnostics else {}
    modulator = getattr(module, "dynamic_modulator", None)
    if modulator is None:
        raise RuntimeError("Non-static model is missing its context modulator")
    if return_diagnostics:
        return modulator(h_static, e_p, return_components=True)
    return modulator(h_static, e_p), {}


# ============================================================================
#  Fusion components
# ============================================================================

class FusionGate(nn.Module):
    """Gene-pair-level attention gate for dual-branch fusion."""

    def __init__(self, branch_dim: int, hidden_ratio: float = 1.0):
        super().__init__()
        hidden = max(int(branch_dim * hidden_ratio), 8)
        self.fc = nn.Sequential(
            nn.Linear(branch_dim * 4, hidden),
            nn.Softplus(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        h_coex: Tensor,
        h_ppi: Tensor,
        edge_index: Adj,
    ) -> Tuple[Tensor, Tensor]:
        src, dst = edge_index[0], edge_index[1]
        h_coex_src = h_coex[src]
        h_coex_dst = h_coex[dst]
        h_ppi_src = h_ppi[src]
        h_ppi_dst = h_ppi[dst]
        cat = torch.cat([h_coex_src, h_coex_dst, h_ppi_src, h_ppi_dst], dim=-1)
        gate_logits = self.fc(cat)
        alpha = torch.sigmoid(gate_logits)
        return alpha, gate_logits


class FusionEntropyLoss(nn.Module):
    """Regulariser that pushes attention gates away from degenerate values."""

    def __init__(self, beta: float = 0.05):
        super().__init__()
        self.beta = beta

    def forward(self, gate_logits: Tensor) -> Tensor:
        alpha = torch.sigmoid(gate_logits)
        eps = 1e-7
        alpha = alpha.clamp(eps, 1 - eps)
        entropy = alpha * torch.log(alpha) + (1 - alpha) * torch.log(1 - alpha)
        return -self.beta * entropy.mean()


# ============================================================================
#  Perturbation Encoder
# ============================================================================

class PerturbationEncoder(nn.Module):
    """
    Converts raw perturbation information into a fixed-size embedding vector.
    Supports binary condition (Kang) and multi-drug SMILES embeddings.
    """

    def __init__(
        self,
        hidden_channels: int,
        multi_pert: bool = True,
        pert_dim: int = 124,
        num_conditions: int = 2,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.multi_pert = multi_pert
        self.pert_dim = pert_dim

        if not multi_pert:
            self.condition_encoder = nn.Embedding(num_conditions, hidden_channels)
        else:
            self.pert_proj = nn.Sequential(
                nn.Linear(pert_dim, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.ReLU(),
            )

    def forward(
        self,
        pert_label: Optional[Tensor],
        condition_str: Optional[str] = None,
    ) -> Tensor:
        if not self.multi_pert:
            if condition_str is not None:
                cond_idx = 0 if str(condition_str).lower() in ("control", "ctrl") else 1
                cond_idx = torch.tensor([cond_idx], dtype=torch.long,
                                       device=next(self.parameters()).device)
                return self.condition_encoder(cond_idx)
            else:
                return self.condition_encoder(
                    torch.tensor([1], dtype=torch.long,
                                device=next(self.parameters()).device)
                )
        else:
            if pert_label is None:
                raise ValueError(
                    "multi_pert=True but pert_label is None. "
                    "Provide pre-computed perturbation features."
                )
            pert = pert_label.to(torch.float32)
            if pert.dim() == 1:
                pert = pert.unsqueeze(0)
            return self.pert_proj(pert)


# ============================================================================
#  Feature Projector (for Node2Vec augmented features)
# ============================================================================

class NodeFeatureProjector(nn.Module):
    """
    Projects augmented node features (mean + var + Node2Vec + topology + expr_stats)
    down to a fixed hidden_dim so the GNN backbone can process them.

    Input feature dim: 2 (mean, var) + n2v_dim (optional) + 7 (topology, optional)
                     + 5 (expr_stats: CV, skew, frac, S_score, G2M_score, optional)
    Output: hidden_channels
    """

    def __init__(
        self,
        in_feat_dim: int,
        hidden_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_feat_dim = in_feat_dim
        self.hidden_channels = hidden_channels

        self.projector = nn.Sequential(
            nn.Linear(in_feat_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.projector(x)


# ============================================================================
#  Single-branch Dynamic GNN with Node2Vec (co-expression only)
# ============================================================================

class DynamicCellTypeGNNNode2Vec(nn.Module):
    """
    Single-branch GNN with:
      - Node2Vec-augmented node features (projected to hidden_dim)
      - Perturbation-conditioned Dynamic Cell-type Embedding
      - Configurable statistical readout

    Architecture::

        Node Features (mean, var, n2v, topology, expr_stats)
                │
                ▼
        NodeFeatureProjector ──► hidden_dim
                │
                ▼
        TransformerConv ──► GATConv
                │
                ▼
        graph readout → h_static (1, D)
                │
                ├──► concat(h_static, e_p) ──► DynamicCellTypeModulator ──► h_dynamic
                │
                ▼
        ctrl ⊕ h_dynamic ⊕ pert_emb ──► MLP ──► prediction
    """

    def __init__(
        self,
        total_genes: int,
        num_perts: int,
        hidden_channels: int,
        in_head: int,
        output_channels: int = 1,
        act_fct: str = "Softplus",
        multi_pert: bool = True,
        pert_dim: int = 124,
        dynamic_scale_init: float = 0.1,
        readout_mode: Literal["max", "stat", "residual_stat"] = "max",
        readout_dropout: float = 0.0,
        node_feat_dim: int = 2,
        node_feat_dropout: float = 0.0,
        use_dynamic: bool = True,
        modulator_type: Optional[str] = None,
    ):
        super().__init__()

        self.total_genes = total_genes
        self.in_head = in_head
        self.hid = hidden_channels
        self.act_fct = act_fct
        self.multi_pert = multi_pert
        self.pert_dim = pert_dim
        self.readout_mode = readout_mode
        self.node_feat_dim = node_feat_dim
        self.modulator_type = modulator_type or ("dynamic" if use_dynamic else "static")
        self.use_dynamic = self.modulator_type != "static"

        # Node feature projector (from augmented features to hidden_dim)
        self.node_projector = NodeFeatureProjector(
            in_feat_dim=node_feat_dim,
            hidden_channels=hidden_channels,
            dropout=node_feat_dropout,
        )

        # GNN layers
        self.conv1 = TransformerConv(-1, hidden_channels, heads=in_head)
        self.lin1 = Linear(-1, hidden_channels * in_head)

        self.conv2 = GATConv(-1, hidden_channels, heads=in_head,
                            add_self_loops=True, concat=False)
        self.lin2 = Linear(-1, hidden_channels)

        # Perturbation encoder
        self.pert_encoder = PerturbationEncoder(
            hidden_channels=hidden_channels,
            multi_pert=multi_pert,
            pert_dim=pert_dim,
        )

        self.embd_pert = MLP([124, 124], act_fct)

        # Dynamic cell-type modulator
        self.dynamic_modulator = build_context_modulator(
            self.modulator_type, hidden_channels, dynamic_scale_init,
        )

        # Readout
        if readout_mode == "max":
            self.readout = None
        elif readout_mode == "stat":
            self.readout = StatisticalReadout(hidden_channels, dropout=readout_dropout)
        elif readout_mode == "residual_stat":
            self.readout = ResidualStatisticalReadout(hidden_channels, dropout=readout_dropout)
        else:
            raise ValueError(
                f"Unsupported readout_mode '{readout_mode}'. "
                "Choose from: max, stat, residual_stat."
            )

        predictor_in = total_genes + hidden_channels + (124 if multi_pert else 0)
        self.lin_predict = MLP([predictor_in, 1024, total_genes], act_fct)

    def _gnn_branch(self, x: Tensor, edge_index: Adj):
        x = self.node_projector(x)
        h1 = self.conv1(x, edge_index) + self.lin1(x)

        N = x.size(0)
        self_loops = torch.arange(N, device=x.device, dtype=torch.long)
        edge_idx_expanded = torch.cat(
            [edge_index, torch.stack([self_loops, self_loops])], dim=1,
        )

        h2 = self.conv2(h1, edge_idx_expanded) + self.lin2(h1)
        return h2

    def _pool(self, h_node: Tensor) -> Tensor:
        if self.readout_mode == "max" or self.readout is None:
            return torch.max(h_node, dim=0)[0].unsqueeze(0)
        return self.readout(h_node)

    def forward(
        self,
        cell_graphs_x: Dict[str, Tensor],
        cell_graphs_edges: Dict[str, Adj],
        cell_line,
        cell_type_keys,
        ctrl: Tensor,
        pert: Optional[Tensor],
        pos,
        ppi_graphs_x=None,
        ppi_graphs_edges=None,
        return_dynamic_emb: bool = False,
        dynamic_pert_override: Optional[Tensor] = None,
        disable_dynamic: bool = False,
        return_dynamic_diagnostics: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict]:
        fused_embs = {}

        for key in cell_type_keys:
            h_node = self._gnn_branch(cell_graphs_x[key], cell_graphs_edges[key])
            fused_embs[key] = self._pool(h_node)

        h_static = torch.cat([fused_embs[c] for c in cell_line], dim=0)

        dynamic_pert = pert if dynamic_pert_override is None else dynamic_pert_override
        e_p = self.pert_encoder(pert_label=dynamic_pert)
        h_dynamic, dynamic_extras = apply_dynamic_modulator(
            self, h_static, e_p, disable_dynamic, return_dynamic_diagnostics,
        )

        if self.multi_pert:
            x = torch.cat([ctrl, h_dynamic,
                           self.embd_pert(pert.to(torch.float32))], dim=1)
        else:
            x = torch.cat([ctrl, h_dynamic], dim=1)
        out = self.lin_predict(x)

        if return_dynamic_emb or return_dynamic_diagnostics:
            dynamic_extras.update({"h_static": h_static, "h_dynamic": h_dynamic})
            return out, dynamic_extras
        return out


# ============================================================================
#  Dual-branch Dynamic GNN with Node2Vec (co-expression + PPI)
# ============================================================================

class DynamicCellTypeDualPriorGNNNode2Vec(nn.Module):
    """
    Dual-branch GNN (co-expression + PPI) with:
      - Node2Vec-augmented node features
      - Perturbation-conditioned Dynamic Cell-type Embedding
      - Gene-pair-level fusion gates
      - Configurable statistical readout

    Architecture::

        Node Features (mean, var, n2v, topology, expr_stats)
                │
                ├───► Coexpr branch: NodeProjector → TransformerConv → GATConv
                │                                      │
                │                                edge-level gate α
                │                                      │
                └───► PPI branch: NodeProjector → TransformerConv → GATConv
                                                         │
                                                  weighted sum: α·h_coex + (1-α)·h_ppi
                                                         │
                                            graph readout → h_static (1, D)
                                                         │
                                                         ├──► concat(h_static, e_p)
                                                         │    ──► DynamicCellTypeModulator
                                                         │
                                                         ▼
                            ctrl ⊕ h_dynamic ⊕ pert_emb ──► MLP ──► prediction
    """

    def __init__(
        self,
        total_genes: int,
        num_perts: int,
        hidden_channels: int,
        in_head: int,
        output_channels: int = 1,
        act_fct: str = "Softplus",
        multi_pert: bool = True,
        pert_dim: int = 124,
        fusion_entropy_beta: float = 0.05,
        dynamic_scale_init: float = 0.1,
        readout_mode: Literal["max", "stat", "residual_stat"] = "max",
        readout_dropout: float = 0.0,
        node_feat_dim: int = 2,
        node_feat_dropout: float = 0.0,
        use_dynamic: bool = True,
        modulator_type: Optional[str] = None,
    ):
        super().__init__()

        self.total_genes = total_genes
        self.in_head = in_head
        self.hid = hidden_channels
        self.act_fct = act_fct
        self.multi_pert = multi_pert
        self.pert_dim = pert_dim
        self.fusion_entropy_beta = fusion_entropy_beta
        self.readout_mode = readout_mode
        self.node_feat_dim = node_feat_dim
        self.modulator_type = modulator_type or ("dynamic" if use_dynamic else "static")
        self.use_dynamic = self.modulator_type != "static"

        branch_out_dim = hidden_channels

        # Shared node feature projector for both branches
        self.node_projector = NodeFeatureProjector(
            in_feat_dim=node_feat_dim,
            hidden_channels=hidden_channels,
            dropout=node_feat_dropout,
        )

        # Co-expression branch
        self.coex_conv1 = TransformerConv(-1, hidden_channels, heads=in_head)
        self.coex_lin1 = Linear(-1, hidden_channels * in_head)

        self.coex_conv2 = GATConv(-1, hidden_channels, heads=in_head,
                                  add_self_loops=True, concat=False)
        self.coex_lin2 = Linear(-1, hidden_channels)

        # PPI branch
        self.ppi_conv1 = TransformerConv(-1, hidden_channels, heads=in_head)
        self.ppi_lin1 = Linear(-1, hidden_channels * in_head)

        self.ppi_conv2 = GATConv(-1, hidden_channels, heads=in_head,
                                 add_self_loops=True, concat=False)
        self.ppi_lin2 = Linear(-1, hidden_channels)

        # Fusion gate and entropy loss
        self.fusion_gate = FusionGate(branch_out_dim)
        self.entropy_loss = FusionEntropyLoss(beta=fusion_entropy_beta)

        # Perturbation encoder
        self.pert_encoder = PerturbationEncoder(
            hidden_channels=hidden_channels,
            multi_pert=multi_pert,
            pert_dim=pert_dim,
        )

        self.embd_pert = MLP([124, 124], act_fct)

        # Dynamic cell-type modulator
        self.dynamic_modulator = build_context_modulator(
            self.modulator_type, hidden_channels, dynamic_scale_init,
        )

        # Readout
        if readout_mode == "max":
            self.readout = None
        elif readout_mode == "stat":
            self.readout = StatisticalReadout(hidden_channels, dropout=readout_dropout)
        elif readout_mode == "residual_stat":
            self.readout = ResidualStatisticalReadout(hidden_channels, dropout=readout_dropout)
        else:
            raise ValueError(
                f"Unsupported readout_mode '{readout_mode}'. "
                "Choose from: max, stat, residual_stat."
            )

        predictor_in = total_genes + hidden_channels + (124 if multi_pert else 0)
        self.lin_predict = MLP([predictor_in, 1024, total_genes], act_fct)

    def _gnn_branch(
        self,
        x: Tensor,
        edge_index: Adj,
        conv1, lin1, conv2, lin2,
    ) -> Tuple[Tensor, Adj]:
        x_proj = self.node_projector(x)
        h1 = conv1(x_proj, edge_index) + lin1(x_proj)

        N = x_proj.size(0)
        self_loops = torch.arange(N, device=x_proj.device, dtype=torch.long)
        edge_index_expanded = torch.cat(
            [edge_index, torch.stack([self_loops, self_loops])],
            dim=1,
        )

        h2 = conv2(h1, edge_index_expanded) + lin2(h1)
        return h2, edge_index_expanded

    def _scatter_fuse(
        self,
        h_coex: Tensor,
        h_ppi: Tensor,
        edge_index: Adj,
        alpha: Tensor,
    ) -> Tensor:
        N = h_coex.size(0)
        src, dst = edge_index[0], edge_index[1]

        h_edge_fused = alpha * h_coex[src] + (1 - alpha) * h_ppi[src]

        node_fused = torch.zeros(N, h_coex.size(1), device=h_coex.device)
        node_fused.index_add_(0, src, h_edge_fused)

        deg = torch.zeros(N, device=h_coex.device)
        deg.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float32))
        deg = deg.clamp(min=1.0)
        node_fused = node_fused / deg.unsqueeze(-1)

        gate_sum = torch.zeros(N, 1, device=h_coex.device)
        gate_cnt = torch.zeros(N, 1, device=h_coex.device)
        gate_sum.index_add_(0, src, alpha)
        gate_cnt.index_add_(0, src, torch.ones_like(alpha))
        beta = (gate_sum / gate_cnt.clamp(min=1.0)).clamp(0.0, 1.0)

        h_out = beta * h_coex + (1 - beta) * h_ppi
        return 0.5 * h_out + 0.5 * node_fused

    def _pool(self, h_node: Tensor) -> Tensor:
        if self.readout_mode == "max" or self.readout is None:
            return torch.max(h_node, dim=0)[0].unsqueeze(0)
        return self.readout(h_node)

    def forward(
        self,
        cell_graphs_x: Dict[str, Tensor],
        cell_graphs_edges: Dict[str, Adj],
        cell_line,
        cell_type_keys,
        ctrl: Tensor,
        pert: Optional[Tensor],
        pos,
        ppi_graphs_x: Optional[Dict[str, Tensor]] = None,
        ppi_graphs_edges: Optional[Dict[str, Adj]] = None,
        return_gate_logits: bool = False,
        return_dynamic_emb: bool = False,
        dynamic_pert_override: Optional[Tensor] = None,
        disable_dynamic: bool = False,
        return_dynamic_diagnostics: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict]:
        fused_embs = {}
        all_gate_logits = []

        for key in cell_type_keys:
            x_coex = cell_graphs_x[key]
            edge_coex = cell_graphs_edges[key]
            h_coex, edge_coex_out = self._gnn_branch(
                x_coex, edge_coex,
                self.coex_conv1, self.coex_lin1,
                self.coex_conv2, self.coex_lin2,
            )

            if (
                ppi_graphs_x is not None and ppi_graphs_edges is not None
                and key in ppi_graphs_x and key in ppi_graphs_edges
            ):
                x_ppi = ppi_graphs_x[key]
                edge_ppi = ppi_graphs_edges[key]
                h_ppi, _ = self._gnn_branch(
                    x_ppi, edge_ppi,
                    self.ppi_conv1, self.ppi_lin1,
                    self.ppi_conv2, self.ppi_lin2,
                )
            else:
                h_ppi = h_coex
                edge_ppi_out = edge_coex_out

            alpha, gate_logits = self.fusion_gate(h_coex, h_ppi, edge_coex_out)
            if return_gate_logits and self.training:
                all_gate_logits.append(gate_logits)

            node_fused = self._scatter_fuse(h_coex, h_ppi, edge_coex_out, alpha)
            fused_embs[key] = self._pool(node_fused)

        h_static = torch.cat([fused_embs[c] for c in cell_line], dim=0)

        dynamic_pert = pert if dynamic_pert_override is None else dynamic_pert_override
        e_p = self.pert_encoder(pert_label=dynamic_pert)
        h_dynamic, dynamic_extras = apply_dynamic_modulator(
            self, h_static, e_p, disable_dynamic, return_dynamic_diagnostics,
        )

        if self.multi_pert:
            x = torch.cat([ctrl, h_dynamic,
                           self.embd_pert(pert.to(torch.float32))], dim=1)
        else:
            x = torch.cat([ctrl, h_dynamic], dim=1)
        out = self.lin_predict(x)

        extras = {}
        if return_gate_logits and self.training and all_gate_logits:
            extras["gate_logits"] = torch.cat(all_gate_logits, dim=0)
        if return_dynamic_emb:
            extras["h_static"] = h_static
            extras["h_dynamic"] = h_dynamic
        if return_dynamic_diagnostics:
            extras.update(dynamic_extras)
            extras["h_static"] = h_static
            extras["h_dynamic"] = h_dynamic

        if extras:
            return out, extras
        return out


# ============================================================================
#  Ablation variants
# ============================================================================

class DynamicCellTypeDualPriorGNNNode2VecGlobal(nn.Module):
    """
    Ablation A — global scalar fusion + Node2Vec + Dynamic cell-type.
    """

    def __init__(
        self,
        total_genes: int,
        num_perts: int,
        hidden_channels: int,
        in_head: int,
        act_fct: str = "Softplus",
        multi_pert: bool = True,
        pert_dim: int = 124,
        dynamic_scale_init: float = 0.1,
        readout_mode: Literal["max", "stat", "residual_stat"] = "max",
        readout_dropout: float = 0.0,
        node_feat_dim: int = 2,
        node_feat_dropout: float = 0.0,
        use_dynamic: bool = True,
        modulator_type: Optional[str] = None,
    ):
        super().__init__()
        self.total_genes = total_genes
        self.in_head = in_head
        self.hid = hidden_channels
        self.act_fct = act_fct
        self.multi_pert = multi_pert
        self.readout_mode = readout_mode
        self.node_feat_dim = node_feat_dim
        self.modulator_type = modulator_type or ("dynamic" if use_dynamic else "static")
        self.use_dynamic = self.modulator_type != "static"

        # Node projector
        self.node_projector = NodeFeatureProjector(
            in_feat_dim=node_feat_dim,
            hidden_channels=hidden_channels,
            dropout=node_feat_dropout,
        )

        # Co-expression branch
        self.coex_conv1 = TransformerConv(-1, hidden_channels, heads=in_head)
        self.coex_lin1 = Linear(-1, hidden_channels * in_head)
        self.coex_conv2 = GATConv(-1, hidden_channels, heads=in_head,
                                  add_self_loops=True, concat=False)
        self.coex_lin2 = Linear(-1, hidden_channels)

        # PPI branch
        self.ppi_conv1 = TransformerConv(-1, hidden_channels, heads=in_head)
        self.ppi_lin1 = Linear(-1, hidden_channels * in_head)
        self.ppi_conv2 = GATConv(-1, hidden_channels, heads=in_head,
                                 add_self_loops=True, concat=False)
        self.ppi_lin2 = Linear(-1, hidden_channels)

        # Global fusion weight
        self.coex_weight = nn.Parameter(torch.tensor(0.5))

        # Perturbation encoder
        self.pert_encoder = PerturbationEncoder(hidden_channels, multi_pert, pert_dim)
        self.embd_pert = MLP([124, 124], act_fct)

        # Dynamic modulator
        self.dynamic_modulator = build_context_modulator(
            self.modulator_type, hidden_channels, dynamic_scale_init,
        )

        # Readout
        if readout_mode == "max":
            self.readout = None
        elif readout_mode == "stat":
            self.readout = StatisticalReadout(hidden_channels, dropout=readout_dropout)
        elif readout_mode == "residual_stat":
            self.readout = ResidualStatisticalReadout(hidden_channels, dropout=readout_dropout)
        else:
            raise ValueError(
                f"Unsupported readout_mode '{readout_mode}'. "
                "Choose from: max, stat, residual_stat."
            )

        predictor_in = total_genes + hidden_channels + (124 if multi_pert else 0)
        self.lin_predict = MLP([predictor_in, 1024, total_genes], act_fct)

    def _gnn_branch(self, x, edge_index, conv1, lin1, conv2, lin2):
        x_proj = self.node_projector(x)
        h1 = conv1(x_proj, edge_index) + lin1(x_proj)
        N = x_proj.size(0)
        self_loops = torch.arange(N, device=x_proj.device, dtype=torch.long)
        edge_expanded = torch.cat([edge_index, torch.stack([self_loops, self_loops])], dim=1)
        h2 = conv2(h1, edge_expanded) + lin2(h1)
        return h2

    def _pool(self, h_node: Tensor) -> Tensor:
        if self.readout_mode == "max" or self.readout is None:
            return torch.max(h_node, dim=0)[0].unsqueeze(0)
        return self.readout(h_node)

    def forward(
        self,
        cell_graphs_x, cell_graphs_edges,
        cell_line, cell_type_keys,
        ctrl, pert, pos,
        ppi_graphs_x=None, ppi_graphs_edges=None,
        **kwargs,
    ) -> Tensor:
        fused_embs = {}
        w = torch.sigmoid(self.coex_weight)

        for key in cell_type_keys:
            h_coex = self._gnn_branch(
                cell_graphs_x[key], cell_graphs_edges[key],
                self.coex_conv1, self.coex_lin1,
                self.coex_conv2, self.coex_lin2,
            )

            if ppi_graphs_x and ppi_graphs_edges and key in ppi_graphs_x and key in ppi_graphs_edges:
                h_ppi = self._gnn_branch(
                    ppi_graphs_x[key], ppi_graphs_edges[key],
                    self.ppi_conv1, self.ppi_lin1,
                    self.ppi_conv2, self.ppi_lin2,
                )
            else:
                h_ppi = h_coex

            h_fused = w * h_coex + (1 - w) * h_ppi
            fused_embs[key] = self._pool(h_fused)

        h_static = torch.cat([fused_embs[c] for c in cell_line], dim=0)
        dynamic_pert = kwargs.get("dynamic_pert_override", pert)
        e_p = self.pert_encoder(pert_label=dynamic_pert)
        return_diag = bool(kwargs.get("return_dynamic_diagnostics", False))
        h_dynamic, extras = apply_dynamic_modulator(
            self, h_static, e_p, bool(kwargs.get("disable_dynamic", False)), return_diag,
        )

        if self.multi_pert:
            x = torch.cat([ctrl, h_dynamic, self.embd_pert(pert.to(torch.float32))], dim=1)
        else:
            x = torch.cat([ctrl, h_dynamic], dim=1)
        out = self.lin_predict(x)
        if return_diag:
            extras.update({"h_static": h_static, "h_dynamic": h_dynamic})
            return out, extras
        return out


class DynamicCellTypeDualPriorGNNNode2VecConcat(nn.Module):
    """
    Ablation B — concatenation fusion + Node2Vec + Dynamic cell-type.
    """

    def __init__(
        self,
        total_genes: int,
        num_perts: int,
        hidden_channels: int,
        in_head: int,
        act_fct: str = "Softplus",
        multi_pert: bool = True,
        pert_dim: int = 124,
        dynamic_scale_init: float = 0.1,
        readout_mode: Literal["max", "stat", "residual_stat"] = "max",
        readout_dropout: float = 0.0,
        node_feat_dim: int = 2,
        node_feat_dropout: float = 0.0,
        use_dynamic: bool = True,
        modulator_type: Optional[str] = None,
    ):
        super().__init__()
        self.total_genes = total_genes
        self.in_head = in_head
        self.hid = hidden_channels
        self.act_fct = act_fct
        self.multi_pert = multi_pert
        self.readout_mode = readout_mode
        self.node_feat_dim = node_feat_dim
        self.modulator_type = modulator_type or ("dynamic" if use_dynamic else "static")
        self.use_dynamic = self.modulator_type != "static"

        # Node projector
        self.node_projector = NodeFeatureProjector(
            in_feat_dim=node_feat_dim,
            hidden_channels=hidden_channels,
            dropout=node_feat_dropout,
        )

        # Co-expression branch
        self.coex_conv1 = TransformerConv(-1, hidden_channels, heads=in_head)
        self.coex_lin1 = Linear(-1, hidden_channels * in_head)
        self.coex_conv2 = GATConv(-1, hidden_channels, heads=in_head,
                                  add_self_loops=True, concat=False)
        self.coex_lin2 = Linear(-1, hidden_channels)

        # PPI branch
        self.ppi_conv1 = TransformerConv(-1, hidden_channels, heads=in_head)
        self.ppi_lin1 = Linear(-1, hidden_channels * in_head)
        self.ppi_conv2 = GATConv(-1, hidden_channels, heads=in_head,
                                 add_self_loops=True, concat=False)
        self.ppi_lin2 = Linear(-1, hidden_channels)

        # Perturbation encoder
        self.pert_encoder = PerturbationEncoder(hidden_channels, multi_pert, pert_dim)
        self.embd_pert = MLP([124, 124], act_fct)

        # Dynamic modulator
        self.dynamic_modulator = build_context_modulator(
            self.modulator_type, hidden_channels, dynamic_scale_init,
        )

        # Readout
        if readout_mode == "max":
            self.readout = None
        elif readout_mode == "stat":
            self.readout = StatisticalReadout(hidden_channels, dropout=readout_dropout)
        elif readout_mode == "residual_stat":
            self.readout = ResidualStatisticalReadout(hidden_channels, dropout=readout_dropout)
        else:
            raise ValueError(
                f"Unsupported readout_mode '{readout_mode}'. "
                "Choose from: max, stat, residual_stat."
            )

        predictor_in = total_genes + 2 * hidden_channels + (124 if multi_pert else 0)
        self.lin_predict = MLP([predictor_in, 1024, total_genes], act_fct)

    def _gnn_branch(self, x, edge_index, conv1, lin1, conv2, lin2):
        x_proj = self.node_projector(x)
        h1 = conv1(x_proj, edge_index) + lin1(x_proj)
        N = x_proj.size(0)
        self_loops = torch.arange(N, device=x_proj.device, dtype=torch.long)
        edge_expanded = torch.cat([edge_index, torch.stack([self_loops, self_loops])], dim=1)
        h2 = conv2(h1, edge_expanded) + lin2(h1)
        return h2

    def _pool(self, h_node: Tensor) -> Tensor:
        if self.readout_mode == "max" or self.readout is None:
            return torch.max(h_node, dim=0)[0].unsqueeze(0)
        return self.readout(h_node)

    def forward(
        self,
        cell_graphs_x, cell_graphs_edges,
        cell_line, cell_type_keys,
        ctrl, pert, pos,
        ppi_graphs_x=None, ppi_graphs_edges=None,
        **kwargs,
    ) -> Tensor:
        fused_embs = {}

        for key in cell_type_keys:
            h_coex = self._gnn_branch(
                cell_graphs_x[key], cell_graphs_edges[key],
                self.coex_conv1, self.coex_lin1,
                self.coex_conv2, self.coex_lin2,
            )

            if ppi_graphs_x and ppi_graphs_edges and key in ppi_graphs_x and key in ppi_graphs_edges:
                h_ppi = self._gnn_branch(
                    ppi_graphs_x[key], ppi_graphs_edges[key],
                    self.ppi_conv1, self.ppi_lin1,
                    self.ppi_conv2, self.ppi_lin2,
                )
            else:
                h_ppi = h_coex

            h_concat = torch.cat([h_coex, h_ppi], dim=-1)
            fused_embs[key] = self._pool(h_concat)

        h_static = torch.cat([fused_embs[c] for c in cell_line], dim=0)
        dynamic_pert = kwargs.get("dynamic_pert_override", pert)
        e_p = self.pert_encoder(pert_label=dynamic_pert)
        return_diag = bool(kwargs.get("return_dynamic_diagnostics", False))
        h_dynamic, extras = apply_dynamic_modulator(
            self, h_static, e_p, bool(kwargs.get("disable_dynamic", False)), return_diag,
        )

        if self.multi_pert:
            x = torch.cat([ctrl, h_dynamic, self.embd_pert(pert.to(torch.float32))], dim=1)
        else:
            x = torch.cat([ctrl, h_dynamic], dim=1)
        out = self.lin_predict(x)
        if return_diag:
            extras.update({"h_static": h_static, "h_dynamic": h_dynamic})
            return out, extras
        return out

