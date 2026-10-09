from torch_geometric.nn import SAGEConv, GCNConv as PyG_GCNConv, GATConv, SGConv, Linear, sequential, TransformerConv, global_mean_pool, GCNConv
import torch
import torch_geometric.utils
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.utils import *
import torch
import pickle
import torch.nn as nn


class MLP(torch.nn.Module):
    def __init__(self, sizes, mid_layer_act, batch_norm=False, last_layer_act="linear"):
        """
        Multi-layer perceptron (MLP) implementation
        :param sizes: list containing the sizes of each layer
        :param mid_layer_act: activation function for the middle layers ('Sigmoid' or 'Softplus')
        :param batch_norm: whether to include batch normalization after hidden layers
        :param last_layer_act: activation function for the last layer (default is "linear")
        """
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            # Add linear layer from sizes[s] -> sizes[s+1]
            layers = layers + [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                # Optionally add BatchNorm if enabled (skip for last layer)
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 1 else None,
                # Add activation function (choose Sigmoid or Softplus)
                torch.nn.Sigmoid() if mid_layer_act == 'Sigmoid' else torch.nn.Softplus()
            ]

        # Remove None layers if batch_norm=False and drop last activation
        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act  # store last activation (not directly applied here)
        self.network = torch.nn.Sequential(*layers)  # build sequential MLP

    def forward(self, x):
        # Forward pass through the network
        return self.network(x)


class GNN(torch.nn.Module):
    def __init__(self, total_genes, num_perts, hidden_channels, in_head,
                 output_channels=1, act_fct=None, multi_pert=True,
                 init_seed=3407):
        """
        Graph Neural Network (GNN) model
        :param total_genes: number of genes (input size)
        :param num_perts: number of perturbations
        :param hidden_channels: hidden dimension size
        :param in_head: number of attention heads
        :param output_channels: output dimension (default=1)
        :param act_fct: activation function for MLPs
        :param multi_pert: whether to include perturbation embeddings
        """
        super().__init__()
        if init_seed is not None:
            torch.manual_seed(int(init_seed))

        self.total_genes = total_genes
        self.in_head = in_head
        self.hid = hidden_channels
        self.act_fct = act_fct
        self.multi_pert = multi_pert

        # First convolution: Transformer-based convolution layer
        self.conv1 = TransformerConv(-1, hidden_channels, heads=self.in_head)
        # Linear layer for projection after conv1
        self.lin1 = Linear(-1, hidden_channels * self.in_head)

        # Second convolution: Graph Attention (GAT) layer
        self.conv2 = GATConv(-1, hidden_channels, heads=self.in_head,
                             add_self_loops=True, concat=False)
        # Linear projection after conv2
        self.lin2 = Linear(-1, hidden_channels)

        # Perturbation embedding network (fixed size [124 -> 124])
        self.embd_pert = MLP([124, 124], self.act_fct)

        # Final prediction network (depends on multi_pert flag)
        self.lin_predict = None
        if multi_pert == True:
            # Input includes genes + hidden features (after conv1+lin1+pool) + perturbation embedding
            # conv1 output has hidden_channels * in_head dimensions (TransformerConv with concat=True)
            predictor_in = self.total_genes + hidden_channels * in_head + 124
            self.lin_predict = MLP([predictor_in, 1024, self.total_genes], self.act_fct)
        else:
            # Input includes genes + hidden features only
            self.lin_predict = MLP([self.total_genes + (hidden_channels * self.in_head),
                                    1024, self.total_genes], self.act_fct)

    def forward(self, x, edge_index, cell_line, cell_type_keys, ctrl, pert, pos):
        """
        Forward pass of the GNN
        :param x: dict of node feature tensors per cell type
        :param edge_index: dict of edge indices per cell type
        :param cell_line: list of cell types present in batch
        :param cell_type_keys: all available cell type keys
        :param ctrl: control input features
        :param pert: perturbation input
        :param pos: positional encodings (currently unused)
        """
        for key in cell_type_keys:
            edge_index[key] = edge_index[key]  # keep edge structure
            # Apply first conv + linear projection
            x[key] = self.conv1(x[key], edge_index[key]) + self.lin1(x[key])
            # Aggregate node features with max pooling
            x[key] = torch.max(x[key], dim=0)[0].unsqueeze(0)

        # Collect features for all cell lines in batch
        cell_type_fet = [x[c] for c in cell_line]
        cell_type_fet = torch.cat(cell_type_fet, dim=0)

        if self.multi_pert == False:
            # Concatenate control and cell-type features
            x = torch.cat([ctrl, cell_type_fet], dim=1)
            # Predict gene expression
            x = self.lin_predict(x)
        if self.multi_pert == True:
            # Concatenate control and cell-type features
            x = torch.cat([ctrl, cell_type_fet], dim=1)
            # Add perturbation embedding
            x = torch.cat([x, self.embd_pert(pert.to(torch.float32))], dim=1)
            # Predict gene expression
            x = self.lin_predict(x)

        return x


class SubstateAwareGNN(torch.nn.Module):
    """
    Substates-aware GNN.

    It extends the original cell-type-level GNN readout by adding an explicit
    cell-state / substate-level embedding. The graph encoder and overall
    prediction head stay aligned with the original GNN so that substate-aware
    training can reuse existing graph preprocessing and training logic.
    """

    def __init__(self, total_genes, num_perts, hidden_channels, in_head,
                 output_channels=1, act_fct=None, multi_pert=True,
                 substate_feature_dim=None, substate_embedding_dim=16):
        """
        Initialize the substate-aware model.

        :param total_genes: number of genes / output expression dimension
        :param num_perts: number of perturbations
        :param hidden_channels: hidden dimension for graph encoder layers
        :param in_head: number of attention heads
        :param output_channels: output dimension (default=1, retained for compatibility)
        :param act_fct: activation function for MLPs
        :param multi_pert: whether to include perturbation embeddings
        :param substate_feature_dim: input dimension of substate feature vector
        :param substate_embedding_dim: output dimension of substate embedding
        """
        super().__init__()
        torch.manual_seed(42)

        if substate_feature_dim is None:
            raise ValueError(
                "Substate-aware model requires substate_feature_dim "
                "(input dimension of substate feature vectors)."
            )

        self.total_genes = total_genes
        self.in_head = in_head
        self.hid = hidden_channels
        self.act_fct = act_fct
        self.multi_pert = multi_pert
        self.substate_feature_dim = int(substate_feature_dim)
        self.substate_embedding_dim = int(substate_embedding_dim)

        # ---------------------------
        # Cell-type-level graph encoder
        # ---------------------------
        # Reuse the original transformer-style graph encoder followed by max pooling.
        self.conv1 = TransformerConv(-1, hidden_channels, heads=self.in_head)
        self.lin1 = Linear(-1, hidden_channels * self.in_head)

        # ---------------------------
        # Perturbation embedding branch
        # ---------------------------
        # Preserve the original perturbation representation so drug-conditioned
        # predictions remain compatible with existing training pipelines.
        self.embd_pert = MLP([124, 124], self.act_fct)

        # ---------------------------
        # Substates-level encoder
        # ---------------------------
        # Learn a compact cell-state embedding from substate features.
        self.substate_encoder = MLP(
            [self.substate_feature_dim, 64, self.substate_embedding_dim],
            self.act_fct,
        )

        # ---------------------------
        # Prediction head
        # ---------------------------
        # Final input = control expression +
        #               cell-type graph embedding +
        #               perturbation embedding (optional) +
        #               substate embedding
        # Always create both heads to support optional perturbation features
        # Note: cell_type_fet dimension is hidden_channels * in_head (due to concat of attention heads)
        self.lin_predict_with_pert = MLP(
            [
                self.total_genes + hidden_channels * in_head + 124 + self.substate_embedding_dim,
                1024,
                self.total_genes,
            ],
            self.act_fct,
        )
        self.lin_predict_no_pert = MLP(
            [
                self.total_genes + hidden_channels * in_head + self.substate_embedding_dim,
                1024,
                self.total_genes,
            ],
            self.act_fct,
        )

    def forward(self, x, edge_index, cell_line, cell_type_keys, ctrl, pert, pos,
                substate_feat=None):
        """
        Forward pass of the substate-aware GNN.

        :param x: dict of node feature tensors per cell type
        :param edge_index: dict of edge indices per cell type
        :param cell_line: list of cell types present in batch
        :param cell_type_keys: all available cell type keys
        :param ctrl: control input features
        :param pert: perturbation input
        :param pos: positional encodings (currently unused)
        :param substate_feat: cell-state / substate feature tensor with batch dimension
        """
        if substate_feat is None:
            raise ValueError(
                "SubstateAwareGNN requires substate_feat because it is intended "
                "only for use_substate=True."
            )

        for key in cell_type_keys:
            edge_index[key] = edge_index[key]
            x[key] = self.conv1(x[key], edge_index[key]) + self.lin1(x[key])
            x[key] = torch.max(x[key], dim=0)[0].unsqueeze(0)

        cell_type_fet = [x[c] for c in cell_line]
        cell_type_fet = torch.cat(cell_type_fet, dim=0)

        if substate_feat.shape[0] != ctrl.shape[0]:
            raise ValueError(
                "substate_feat batch dimension must match ctrl batch dimension, "
                f"but got substate_feat={substate_feat.shape} and ctrl={ctrl.shape}."
            )

        substate_embedding = self.substate_encoder(substate_feat.to(torch.float32))

        if pert is None:
            x = torch.cat([ctrl, cell_type_fet, substate_embedding], dim=1)
            x = self.lin_predict_no_pert(x)
        else:
            x = torch.cat([ctrl, cell_type_fet], dim=1)
            x = torch.cat([x, self.embd_pert(pert.to(torch.float32))], dim=1)
            x = torch.cat([x, substate_embedding], dim=1)
            x = self.lin_predict_with_pert(x)

        return x


class GNN_GCN(torch.nn.Module):
    """
    GCN-based GNN model: same architecture as GNN but with GCNConv instead of GATConv.

    GCNConv does not support multi-head attention natively (unlike GATConv).
    To keep output dimensions consistent, we apply multiple GCNConv layers in parallel
    (one per head) and concatenate their outputs, then project with lin2.
    This yields the same [num_nodes, hidden_channels] output as GNN's GATConv.
    """

    def __init__(self, total_genes, num_perts, hidden_channels, in_head,
                 output_channels=1, act_fct=None, multi_pert=True):
        super().__init__()
        torch.manual_seed(42)

        self.total_genes = total_genes
        self.in_head = in_head
        self.hid = hidden_channels
        self.act_fct = act_fct
        self.multi_pert = multi_pert

        # First convolution: TransformerConv (same as GNN)
        self.conv1 = TransformerConv(-1, hidden_channels, heads=self.in_head)
        self.lin1 = Linear(-1, hidden_channels * self.in_head)

        # Second convolution: multi-head GCN
        # Each head is an independent GCNConv; outputs are concatenated.
        # Final dimension: hidden_channels * in_head (same as GNN's conv1 output)
        self.gcn_heads = nn.ModuleList([
            GCNConv(-1, hidden_channels, improved=True, add_self_loops=True)
            for _ in range(in_head)
        ])
        # Project concatenated multi-head GCN output back to hidden_channels
        self.lin2 = Linear(-1, hidden_channels)

        # Perturbation embedding network (identical to GNN)
        self.embd_pert = MLP([124, 124], self.act_fct)

        # Final prediction network (identical to GNN)
        self.lin_predict = None
        if multi_pert:
            self.lin_predict = MLP([self.total_genes + hidden_channels + 124,
                                    1024, self.total_genes], self.act_fct)
        else:
            self.lin_predict = MLP([self.total_genes + (hidden_channels * self.in_head),
                                    1024, self.total_genes], self.act_fct)

    def forward(self, x, edge_index, cell_line, cell_type_keys, ctrl, pert, pos):
        for key in cell_type_keys:
            edge_index[key] = edge_index[key]
            # conv1 + residual (same as GNN)
            x[key] = self.conv1(x[key], edge_index[key]) + self.lin1(x[key])
            # Multi-head GCN: each head produces [num_nodes, hidden_channels]
            head_outs = [act(h(x[key], edge_index[key]))
                         for h, act in zip(self.gcn_heads,
                                          [torch.relu] * len(self.gcn_heads))]
            # Concatenate heads: [num_nodes, hidden_channels * in_head]
            x[key] = torch.cat(head_outs, dim=-1)
            # Project back to hidden_channels
            x[key] = self.lin2(x[key])
            x[key] = torch.max(x[key], dim=0)[0].unsqueeze(0)

        cell_type_fet = [x[c] for c in cell_line]
        cell_type_fet = torch.cat(cell_type_fet, dim=0)

        if self.multi_pert == False:
            x = torch.cat([ctrl, cell_type_fet], dim=1)
            x = self.lin_predict(x)
        if self.multi_pert == True:
            x = torch.cat([ctrl, cell_type_fet], dim=1)
            x = torch.cat([x, self.embd_pert(pert.to(torch.float32))], dim=1)
            x = self.lin_predict(x)

        return x


class GNN_MLPOnly(torch.nn.Module):
    """
    MLP-only model (no graph encoding).
    Completely skips graph convolution layers.
    cell_type_fet is a zero tensor of the same dimension as GNN's output,
    ensuring the downstream MLP prediction head requires no modification.
    """

    def __init__(self, total_genes, num_perts, hidden_channels, in_head,
                 output_channels=1, act_fct=None, multi_pert=True):
        super().__init__()
        torch.manual_seed(42)

        self.total_genes = total_genes
        self.in_head = in_head
        self.hid = hidden_channels
        self.act_fct = act_fct
        self.multi_pert = multi_pert

        # Perturbation embedding network (identical to GNN)
        self.embd_pert = MLP([124, 124], self.act_fct)

        # Final prediction network (identical to GNN)
        self.lin_predict = None
        if multi_pert:
            self.lin_predict = MLP([self.total_genes + hidden_channels + 124,
                                    1024, self.total_genes], self.act_fct)
        else:
            self.lin_predict = MLP([self.total_genes + (hidden_channels * self.in_head),
                                    1024, self.total_genes], self.act_fct)

    def forward(self, x, edge_index, cell_line, cell_type_keys, ctrl, pert, pos):
        """
        Forward pass without any graph encoding.

        cell_type_fet is a zero tensor (no graph information used).
        This allows the model to rely solely on ctrl and (optionally) pert features,
        providing a baseline comparison for the importance of graph structure.
        """
        # Dimension matches GNN's cell_type_fet: hidden_channels * in_head
        cell_type_fet = torch.zeros(
            ctrl.shape[0], self.hid * self.in_head,
            device=ctrl.device, dtype=ctrl.dtype
        )

        if self.multi_pert == False:
            x = torch.cat([ctrl, cell_type_fet], dim=1)
            x = self.lin_predict(x)
        if self.multi_pert == True:
            x = torch.cat([ctrl, cell_type_fet], dim=1)
            x = torch.cat([x, self.embd_pert(pert.to(torch.float32))], dim=1)
            x = self.lin_predict(x)

        return x
