import os
from scipy import stats
import scanpy as sc
import numpy as np 
import pandas as pd
import networkx as nx
import anndata
import copy
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics
import torch_geometric
import tqdm.notebook as tq
from numpy.random import RandomState
from scipy import sparse
import torch
import seaborn as sns
import matplotlib.cm as cm
from scipy.sparse import csr_matrix
from torch.optim.lr_scheduler import StepLR
import anndata as ad
from torch_geometric.utils.convert import from_networkx
from sklearn.metrics import mean_squared_error
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected, is_undirected
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip, HeteroData, Batch
from torch_geometric.utils import *
import ot
from torch import nn
device = 'cuda'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

def loss_fct(pred, y, perts):
    """
    EMD (Earth Mover’s Distance) loss to train the model.
    Computes distributional distance between predicted and true values,
    grouped by perturbations.
    """
    perts = np.array(perts)  # Convert perturbation labels to NumPy array
    losses = torch.tensor(0.0).to(pred.device)  # Initialize total loss on the same device as predictions
    
    # Loop over each unique perturbation
    for p in set(perts):
        pert_idx = np.where(perts == p)[0]   # Indices of samples for this perturbation
        y_p = y[pert_idx]                    # True values for this perturbation
        pred_p = pred[pert_idx]              # Predicted values for this perturbation
        
        # Uniform weights over samples in this perturbation
        ab = torch.ones(y_p.shape[0]) / y_p.shape[0]
        
        # Compute pairwise cost (Euclidean distance) between predicted and true samples
        M = ot.dist(pred_p, y_p, metric='euclidean').to(pred.device)
        
        # Compute Earth Mover’s Distance (optimal transport cost)
        loss = ot.lp.emd2(ab.to(pred.device), ab, M)
        
        # Accumulate loss across perturbations
        losses = losses + loss
        
        # Free memory by deleting cost matrix
        del M
    
    # Average loss over number of unique perturbations
    return losses / len(set(perts))
    
#--------------------------------------------------------------------------------------------------------------
def Correlation_matrix(adata, cell_type, cell_type_key,
                       hv_genes_cells = None, union_HVGs = False):
    # compute pairwise gene-gene correlation matrix for a given cell type
    
    # subset AnnData object to the given cell type
    if union_HVGs:
        ad = adata[ (adata.obs[cell_type_key] == cell_type), :].copy()  # use all genes
    else:
        ad = adata[ (adata.obs[cell_type_key] == cell_type), hv_genes_cells[cell_type] ].copy()  # use HVGs for this cell type
    
    # extract expression matrix (dense or sparse)
    try: 
        X = ad.X.A  # if sparse, convert to dense array
    except: 
        X = ad.X    # if already dense
    
    genes = ad.var.index.values.tolist()  # gene names
    
    # compute correlation matrix across genes
    out = np.corrcoef(X, rowvar= False)
    out[np.isnan(out)] = 0.0  # replace NaNs with zeros
    
    # flatten upper triangle values (optional, not used later)
    values = (out[np.triu_indices(len(genes), k = 1)].flatten())
    
    # store correlation matrix in DataFrame with gene names
    out = pd.DataFrame((out), index = genes, columns = genes)
    
    # reshape to long format (gene1, gene2, correlation)
    out = out.stack().reset_index()
    return out  # return correlation matrix in long-format DataFrame

#--------------------------------------------------------------------------------------------------
 
def create_coexpression_graph(adata, co_expr_net, cell_type, threshold,
                              gene_key = 'gene_name',  celltype_key = 'cell_type'):

    # Take absolute values of co-expression weights (to ignore sign of correlation)
    co_expr_net[0] = np.abs(co_expr_net[0])
    print('number_of_edges before the threshold (5000 x 5000): ', len(co_expr_net))

    # Keep only edges above the given threshold
    co_expr_net = co_expr_net.loc[co_expr_net[0] >= threshold]
    print('number_of_edges after the threshold: ', len(co_expr_net))

    # Remove self-loops (edges where source == target)
    co_expr_net = co_expr_net.loc[co_expr_net.level_0 != co_expr_net.level_1]
    print('number_of_edges after removing self loops: ', len(co_expr_net))

    # Convert filtered dataframe into a directed graph with edge weights
    co_expr_net = nx.from_pandas_edgelist(
        co_expr_net,
        source='level_0', target='level_1',
        edge_attr=0, create_using=nx.DiGraph()
    )
    print('final number_of_edges: ', co_expr_net.number_of_edges())

    # Get all weakly connected components in the graph
    connected_components = nx.weakly_connected_components(co_expr_net)

    # Identify the largest connected component
    largest_component = max(connected_components, key=len)
    
    # Optional: restrict graph to the largest connected component
    # co_expr_net = co_expr_net.subgraph(largest_component).copy()
    
    # Map gene names to their positions in adata.var
    nodes_list = adata.var.reset_index()
    nodes_list = nodes_list.loc[nodes_list[gene_key].isin(list(co_expr_net.nodes))]
    nodes_list = pd.DataFrame({
        'gene_loc': nodes_list.index.values,
        'gene_id': nodes_list[gene_key].values
    })

    # Dictionary: gene_id → index in nodes_list
    dic_nodes = dict(zip(nodes_list.gene_id, nodes_list.index))

    # Convert networkx graph back to edge list dataframe
    edges = nx.to_pandas_edgelist(co_expr_net)
    edges['source'] = edges['source'].map(dic_nodes)
    edges['target'] = edges['target'].map(dic_nodes)
    edges.sort_values(['source', 'target'], inplace=True)

    # Select control samples for the given cell type
    ctrl = adata[(adata.obs[celltype_key] == cell_type)].copy()
    ctrl = ctrl[:, nodes_list.gene_loc]

    # Convert control expression matrix to PyTorch tensor (handle sparse matrices)
    try:
        if hasattr(ctrl.X, 'toarray'):
            x = torch.tensor(ctrl.X.toarray()).float()
        elif hasattr(ctrl.X, 'A'):
            x = torch.tensor(ctrl.X.A).float()
        else:
            x = torch.tensor(np.array(ctrl.X)).float()
    except:
        x = torch.tensor(ctrl.X).float()

    # Build PyTorch Geometric graph object
    G = Data(
        x=x.T,  # features: gene expression (genes × cells → transposed)
        edge_index=torch.tensor(edges[['source', 'target']].to_numpy().T),
        pos=list(nodes_list.gene_loc.values),
        edge_attr=torch.tensor(edges[0])  # edge weights
    )
    return G

#-----------------------------------------------------------------------------------------------------

def _get_sample_ctrl_x(sample):
    try:
        return torch.tensor(sample.layers['ctrl_x'])
    except KeyError:
        pass
    try:
        return torch.tensor(sample.obsm['ctrl_x'])
    except (KeyError, AttributeError):
        pass
    warnings.warn(
        "'ctrl_x' is missing from sample.layers/obsm; falling back to sample.X. "
        "Please ensure the input AnnData contains matched control expressions.",
        UserWarning,
    )
    x = sample.X.toarray() if hasattr(sample.X, 'toarray') else np.asarray(sample.X)
    return torch.tensor(x)


def create_cells(stim_data, cell_type_network, canonical_smiles,
                use_substate=False,
                substate_feature_key="substate_features",
                substate_id_key="substate_id",
                state_feature_mode="none",
                state_features_key="state_features_final",
                substate_label_key="substate_label",
                shuffle_state_features=False,
                seed=42):
    """
    Create PyG Data objects from AnnData for training/inference.

    Parameters
    ----------
    stim_data : AnnData
        Input data containing cells to process.
    cell_type_network : dict
        Cell type specific graphs.
    canonical_smiles : dict
        Drug SMILES fingerprints.
    use_substate : bool, default=False
        Whether to use substate features (legacy parameter, use state_feature_mode instead).
    substate_feature_key : str, default="substate_features"
        Key for substate features in obsm (used when use_substate=True).
    substate_id_key : str, default="substate_id"
        Key for substate IDs in obs.
    state_feature_mode : str, default="none"
        Mode for state features:
        - "none": baseline, no state features added
        - "state_final": use adata.obsm["state_features_final"]
        - "substate_only": use adata.obsm["substate_features"]
        - "bio_only": use adata.obsm["bio_state_features_control"]
    state_features_key : str, default="state_features_final"
        Key for state features in obsm when state_feature_mode != "none".
    substate_label_key : str, default="substate_label"
        Key for substate labels in obs.
    shuffle_state_features : bool, default=False
        If True, shuffle state features within each cell_type.
    seed : int, default=42
        Random seed for reproducibility.
    """
    cells = []
    obs = stim_data.obs
    print(obs.cell_type.unique(), obs.condition.unique())

    # Validate state_features_key - FORBIDDEN: bio_state_features_observed
    if state_feature_mode != "none" and state_features_key == "bio_state_features_observed":
        raise ValueError(
            "Using 'bio_state_features_observed' as model input is FORBIDDEN. "
            "This contains observed (non-control) biological state features which would cause data leakage. "
            "Please use 'state_features_final' (substate_features + bio_state_features_control) instead."
        )

    # Determine which feature key to use based on mode
    effective_state_feature_key = None
    if state_feature_mode != "none":
        if state_feature_mode == "state_final":
            effective_state_feature_key = "state_features_final"
        elif state_feature_mode == "substate_only":
            effective_state_feature_key = "substate_features"
        elif state_feature_mode == "bio_only":
            effective_state_feature_key = "bio_state_features_control"
        else:
            raise ValueError(
                f"Unknown state_feature_mode: '{state_feature_mode}'. "
                f"Valid options: 'none', 'state_final', 'substate_only', 'bio_only'."
            )

    # Validate obsm key exists
    if effective_state_feature_key is not None:
        if effective_state_feature_key not in stim_data.obsm:
            raise ValueError(
                f"Required obsm key '{effective_state_feature_key}' not found in stim_data.obsm. "
                f"Available keys: {list(stim_data.obsm.keys())}. "
                f"Please run the appropriate preprocessing script first:\n"
                f"  - For 'state_features_final': run merge_substate_and_bio_features.py\n"
                f"  - For 'substate_features': run prepare_substates.py\n"
                f"  - For 'bio_state_features_control': run compute_biological_state_scores.py"
            )

    # Handle legacy use_substate parameter
    if use_substate and state_feature_mode == "none":
        # Convert legacy use_substate to new mode
        if substate_feature_key not in stim_data.obsm:
            raise ValueError(
                f"Substate features missing: '{substate_feature_key}' not in stim_data.obsm. "
                f"Please run prepare_substates.py first."
            )
        if substate_id_key not in obs.columns:
            raise ValueError(
                f"Substate labels missing: '{substate_id_key}' not in stim_data.obs. "
                f"Please run prepare_substates.py first."
            )

    # Load state features if needed
    state_feature_matrix = None
    if effective_state_feature_key is not None:
        state_feature_matrix = stim_data.obsm[effective_state_feature_key]
        if state_feature_matrix.shape[0] != stim_data.shape[0]:
            raise ValueError(
                f"State feature matrix length ({state_feature_matrix.shape[0]}) does not match "
                f"number of cells in stim_data ({stim_data.shape[0]})."
            )
        print(f"Loaded state features: {effective_state_feature_key}, shape={state_feature_matrix.shape}")

    # Prepare shuffle indices if needed
    shuffle_indices = None
    if shuffle_state_features and state_feature_matrix is not None:
        np.random.seed(seed)
        all_indices = np.arange(stim_data.shape[0])
        cell_types = obs["cell_type"].values

        # Create mapping from cell_type to shuffled indices within that cell_type
        unique_cell_types = np.unique(cell_types)
        shuffle_map = {}

        for ct in unique_cell_types:
            ct_mask = cell_types == ct
            ct_indices = all_indices[ct_mask]
            shuffled_ct_indices = ct_indices.copy()
            np.random.shuffle(shuffled_ct_indices)
            shuffle_map[ct] = dict(zip(ct_indices, shuffled_ct_indices))

        # Build full shuffle index array
        shuffle_indices = np.array([shuffle_map[ct][idx] for idx, ct in zip(all_indices, cell_types)])
        print(f"Shuffled state features within {len(unique_cell_types)} cell types (seed={seed})")

    # Group the AnnData object by cov_drug (cell_type + drug) to avoid repeated filtering
    cov_drug_groups = {
        cov_drug: stim_data[obs.cov_drug == cov_drug, :].copy()
        for cov_drug in obs.cov_drug.unique()
    }

    # Iterate over each cov_drug group
    for cov_drug, adata_cov_drug in tq.tqdm(cov_drug_groups.items(), desc="Processing cov_drugs"):
        try:
            # Split cov_drug string into cell_type and drug
            cell_type, drug = cov_drug.split("_", 1)
        except ValueError:
            # Skip this group if splitting fails
            continue

        # Iterate over each sample (row) in the current group
        for sample in tq.tqdm(adata_cov_drug, leave=False, desc=f"Processing {cov_drug} samples"):
            # Convert control expression layer to tensor
            x = _get_sample_ctrl_x(sample)
            # Convert perturbed expression to tensor (dense if sparse)
            y = torch.tensor(sample.X.toarray() if hasattr(sample.X, 'toarray') else np.asarray(sample.X))

            if canonical_smiles is None:
                # If no drug fingerprints are provided, store only cell/drug info
                cell_kwargs = dict(
                    x=x,
                    y=y,
                    cell_type=cell_type,
                    cov_drug=cov_drug,
                    drug=drug
                )
            else:
                # Retrieve condition and get its drug fingerprint (SMILES → tensor)
                condition = sample.obs['condition'].values[0]
                pert = torch.tensor(canonical_smiles[condition]).unsqueeze(0)
                # Include fingerprint in the graph data object
                cell_kwargs = dict(
                    x=x,
                    y=y,
                    pert_label=pert,
                    cell_type=cell_type,
                    cov_drug=cov_drug,
                    drug=drug
                )

            # Handle state features (new unified approach)
            if state_feature_matrix is not None:
                sample_name = sample.obs_names[0]
                position = list(stim_data.obs_names).index(sample_name)

                # Get shuffled position if shuffle is enabled
                if shuffle_indices is not None:
                    position = shuffle_indices[position]

                # Add state features
                state_feat = torch.tensor(state_feature_matrix[position]).float().unsqueeze(0)
                cell_kwargs["substate_feat"] = state_feat

                # Add substate_id if available
                if substate_id_key in obs.columns:
                    substate_id = int(obs.loc[sample_name, substate_id_key])
                    cell_kwargs["substate_id"] = torch.tensor([substate_id], dtype=torch.long)

                # Add substate_label if available
                if substate_label_key in obs.columns:
                    substate_label = str(obs.loc[sample_name, substate_label_key])
                    cell_kwargs["substate_label"] = substate_label

            # Handle legacy use_substate (only if state_feature_mode is "none")
            elif use_substate and state_feature_mode == "none":
                sample_name = sample.obs_names[0]
                position = list(stim_data.obs_names).index(sample_name)
                substate_id = int(obs.loc[sample_name, substate_id_key])
                substate_feat = torch.tensor(
                    stim_data.obsm[substate_feature_key][position]
                ).float().unsqueeze(0)
                cell_kwargs["substate_id"] = torch.tensor([substate_id], dtype=torch.long)
                cell_kwargs["substate_feat"] = substate_feat

            cell = Data(**cell_kwargs)
            cells.append(cell)

    example_cell = cells[0] if cells else None

    # Print summary information
    print("\n" + "=" * 60)
    print("create_cells Summary")
    print("=" * 60)
    if example_cell is not None:
        print(f"Total cells created: {len(cells)}")
        print(f"Has substate_feat: {hasattr(example_cell, 'substate_feat')}")
        if hasattr(example_cell, 'substate_feat'):
            print(f"substate_feat shape: {tuple(example_cell.substate_feat.shape)}")
        print(f"Has substate_id: {hasattr(example_cell, 'substate_id')}")
        print(f"Has substate_label: {hasattr(example_cell, 'substate_label')}")
        print(f"state_feature_mode: {state_feature_mode}")
        if shuffle_state_features:
            print(f"shuffle_state_features: True (seed={seed})")
    print("=" * 60)

    return cells



#-------------------------------------------------------------------------------------------------------------

def rank_genes(dedf): 
    # Compute absolute log fold changes (magnitude of change regardless of direction)
    dedf['abs_logfoldchanges'] = dedf['logfoldchanges'].abs()

    # Rank genes by adjusted p-values (smaller p-value = higher rank)
    dedf["Rank_pvals_adj"] = dedf["pvals_adj"].rank(method='dense')

    # Rank genes by absolute log fold changes (larger change = higher rank)
    dedf["Rank_abs_logfoldchanges"] = dedf["abs_logfoldchanges"].rank(method='dense', ascending=False)

    # Combine ranks using geometric mean of the two ranks
    dedf['Final_rank'] = (dedf["Rank_pvals_adj"] * dedf["Rank_abs_logfoldchanges"]) ** (1/2)

    # Sort genes by the final combined ranking score
    dedf = dedf.sort_values('Final_rank')

    # Select the top 100 genes as DEGs
    num_genes = 100
    DEGs_name = dedf.head(num_genes).names.values 

    # Return the list of top-ranked gene names
    return list(DEGs_name)


#--------------------------------------------------------------------------------------------------------

def balance_subsample(data, labels, total_samples, seed=None):
    # Set random seed for reproducibility (if provided)
    if seed is not None:
        np.random.seed(seed)

    # Get unique class labels and their counts
    unique_labels, class_counts = np.unique(labels, return_counts=True)

    # Sort labels by their class size (smallest to largest)
    # This ensures remainder samples are assigned starting from the largest group
    sorted_indices = np.argsort(class_counts)
    unique_labels = unique_labels[sorted_indices]

    # Base number of samples to draw per class
    samples_per_class = np.floor_divide(total_samples, len(unique_labels))

    # Remainder after equal distribution across classes
    rem = total_samples % len(unique_labels)

    total = total_samples
    balanced_data = []

    # Iterate over each class
    for count, label in enumerate(unique_labels):
        # Initialize deterministic random generator
        prng = RandomState(1234567890)

        # Get indices of all samples belonging to this class
        indices = np.where(labels == label)[0]

        # Sample from class (without replacement if enough samples, else with replacement)
        if int(samples_per_class) <= len(indices): 
            selected_indices = prng.choice(indices, int(samples_per_class), replace=False)
        else:
            selected_indices = prng.choice(indices, int(samples_per_class), replace=True)

        # Add sampled data to the balanced dataset
        balanced_data.extend(data[selected_indices])

        # Update remaining sample count
        total = total - int(samples_per_class)

        # For the last class, handle the remainder distribution
        if count == (len(unique_labels)-1):
            if rem <= len(indices): 
                selected_indices = prng.choice(indices, rem, replace=False)
            else: 
                selected_indices = prng.choice(indices, rem, replace=True)
            balanced_data.extend(data[selected_indices])

    # Return the final balanced dataset
    return balanced_data


#---------------------------------------------------------------------------------------------------------
def train(model, num_epochs, lr, weight_decay, cell_type_network, train_loader, multi_pert = True, use_state_features=False):
    """
    The training function

    Parameters
    ----------
    model : torch.nn.Module
        The model to train (GNN or SubstateAwareGNN).
    num_epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    weight_decay : float
        Weight decay for optimizer.
    cell_type_network : dict
        Cell type specific graphs.
    train_loader : DataLoader
        Training data loader.
    multi_pert : bool, default=True
        Whether to use multi-perturbation mode.
    use_state_features : bool, default=False
        Whether to use state features (substate_feat) from samples.
        If True, model should be SubstateAwareGNN and samples must have substate_feat attribute.
        If False, model should be GNN and original behavior is preserved.
    """
    
    print('Training Starts')
    print(f'use_state_features={use_state_features}')
    # Define mean squared error loss
    mse_loss = torch.nn.MSELoss(reduction='mean')
    
    # Use GPU if available
    device = 'cuda'
    
    # Adam optimizer with learning rate and weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Move model to GPU and ensure float precision
    model = model.to(device).float()
    
    num_epochs = num_epochs
    # Iterate through epochs
    for epoch in tq.tqdm(range(num_epochs), leave=False):
        running_loss = 0.0
        train_epoch_loss = 0.0
        count = 0
        
        # Iterate through batches from train_loader
        for sample in tq.tqdm(train_loader, leave=False):
            model.train()  # set model to training mode
            sample = sample.to(device)  # move batch to GPU
            cell_type = sample.cell_type  # cell type(s) in the batch
            ctrl = sample.x               # control expression features
            
            # Perturbation label (if multiple perturbations supported and available)
            if multi_pert and hasattr(sample, 'pert_label'):
                pert_label = sample.pert_label
            else: 
                pert_label = None
            
            batch = sample.batch   # batch indices
            y = sample.y           # ground truth expression values
            
            # Collect cell-type-specific graph features (nodes, positions, edges)
            cell_graphs_x = {Cell: cell_type_network[Cell].x.to(device) for Cell in np.unique(cell_type)}
            cell_graphs_pos = {Cell: cell_type_network[Cell].pos.to(device) for Cell in np.unique(cell_type)}
            cell_graphs_edges = {Cell: cell_type_network[Cell].edge_index.to(device) for Cell in np.unique(cell_type)}
            
            # Forward pass through the model
            if use_state_features:
                if not hasattr(sample, 'substate_feat'):
                    raise ValueError(
                        "use_state_features=True but the current batch does not contain substate_feat. "
                        "Please regenerate cells with state_feature_mode != 'none'."
                    )
                substate_feat = sample.substate_feat
                out = model(cell_graphs_x, cell_graphs_edges,
                            cell_type, cell_graphs_edges.keys(), ctrl, pert_label, cell_graphs_pos,
                            substate_feat=substate_feat)
            else:
                out = model(cell_graphs_x, cell_graphs_edges, 
                            cell_type, cell_graphs_edges.keys(), ctrl, pert_label, cell_graphs_pos)
            
            # Compute loss (custom loss function loss_fct used here)
            loss = loss_fct(out, y, sample.cov_drug) 
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate batch loss
            running_loss += loss.item()
        
        # Compute average loss per epoch
        train_epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch}, train loss: {train_epoch_loss}")
    
    # Return trained model
    return model


#-----------------------------------------------------------------------------------------------------------------------------------------------

def create_anndata(pred_p, truth_p, adata, cell_type_network, p):
    # Extract cell type (c) and drug (d) from the input parameter `p`
    c = p.split('_')[0]
    d = p.split('_')[1]

    # Retrieve the positions of the genes for the specified cell type
    pos_genes = cell_type_network[c].pos

    # Get control data from the AnnData object (for the same cell type, control condition)
    ctrl_p = adata[adata.obs.cov_drug == c + '_control', pos_genes.tolist()].X.A

    # Stack true response, predicted response, and control into one matrix
    combined_data = np.vstack([truth_p, pred_p, ctrl_p])

    # Cell type labels for all rows (truth, prediction, control)
    cell_type = np.array([c] * combined_data.shape[0])

    # Assign conditions:
    condition_truth = np.array([d] * truth_p.shape[0])              # truth rows → drug label
    condition_pred = np.array(['pred_' + d] * pred_p.shape[0])      # prediction rows → "pred_drug"
    condition_ctrl = np.array(['control'] * ctrl_p.shape[0])        # control rows → "control"

    # Combine all conditions into a single array
    condition = np.concatenate([condition_truth, condition_pred, condition_ctrl])

    # Build the observations DataFrame (obs)
    obs_df = pd.DataFrame({
        'cell_type': np.concatenate([
            cell_type[:truth_p.shape[0]], 
            cell_type[:pred_p.shape[0]], 
            cell_type[:ctrl_p.shape[0]]
        ]),
        'condition': condition
    })

    # Create the AnnData object with expression matrix and observations
    adata_combined = sc.AnnData(X=combined_data, obs=obs_df)

    # Assign gene names as variable names
    adata_combined.var_names = pos_genes.tolist()

    # Return the combined AnnData object
    return adata_combined

#-----------------------------------------------------------------------------------------------------------------------------------------------

def evaluate_substate_mixture(pred, truth, perts, substate_ids, save_path, degs_dict=None, cell_type_network=None):
    """
    Evaluate substate-level and mixture-level predictions.

    For each cov_drug:
    1. Group cells by substate_id.
    2. For each substate k with at least 3 samples, compute:
       - pred_mu_k, true_mu_k
       - pred_var_k, true_var_k
       - substate proportion pi_k
    3. Aggregate cell-type-level mean and variance using mixture formulas.
    4. Compute mixture and substate-level R²/MSE metrics.
    5. Save per-substate and mixture summary CSVs.
    """
    if pred.shape != truth.shape:
        raise ValueError(
            f"pred and truth must have the same shape, got {pred.shape} and {truth.shape}."
        )
    if not (len(perts) == len(substate_ids) == pred.shape[0]):
        raise ValueError(
            "perts and substate_ids must have the same length as the cell dimension of pred/truth."
        )

    os.makedirs(save_path, exist_ok=True)
    perts = np.asarray(perts)
    substate_ids = np.asarray(substate_ids)
    mixture_summary_rows = []
    substate_rows = []

    for p in sorted(set(perts)):
        pert_idx = np.where(perts == p)[0]
        p_pred = pred[pert_idx]
        p_truth = truth[pert_idx]
        p_substates = substate_ids[pert_idx]

        unique_substates, counts = np.unique(p_substates, return_counts=True)
        valid_substates = unique_substates[counts >= 3]
        skipped_substates = unique_substates[counts < 3]
        if len(skipped_substates):
            warnings.warn(
                f"cov_drug='{p}' skipped substates {sorted(skipped_substates.tolist())} "
                f"because they have fewer than 3 samples."
            )

        if valid_substates.size == 0:
            warnings.warn(
                f"cov_drug='{p}' has no valid substate groups; skipping substate/mixture evaluation."
            )
            continue

        pred_mu_list = []
        true_mu_list = []
        pred_var_list = []
        true_var_list = []
        pi_list = []
        substate_metrics = []

        cell_type = p.split('_')[0]
        deg_indices = None
        if degs_dict is not None and p in degs_dict:
            deg_indices = np.asarray(degs_dict[p])
            if cell_type_network is not None and cell_type in cell_type_network:
                graph_pos = cell_type_network[cell_type].pos
                valid_degs = deg_indices[deg_indices < len(graph_pos)]
                if len(valid_degs) == 0:
                    valid_degs = None
            else:
                valid_degs = None if len(deg_indices) == 0 else deg_indices
        else:
            valid_degs = None

        for k in valid_substates:
            mask = p_substates == k
            pred_mu_k = np.mean(p_pred[mask], axis=0)
            true_mu_k = np.mean(p_truth[mask], axis=0)
            pred_var_k = np.var(p_pred[mask], axis=0)
            true_var_k = np.var(p_truth[mask], axis=0)

            pred_var_k = np.clip(pred_var_k, 0.0, None)
            true_var_k = np.clip(true_var_k, 0.0, None)

            pred_mu_list.append(pred_mu_k)
            true_mu_list.append(true_mu_k)
            pred_var_list.append(pred_var_k)
            true_var_list.append(true_var_k)
            pi_list.append(mask.sum() / mask.shape[0])

            substate_entry = {
                'cov_drug': p,
                'substate_id': int(k),
                'n_cells': int(mask.sum()),
                'r2_mean': metrics.r2_score(true_mu_k, pred_mu_k),
                'r2_std': metrics.r2_score(true_var_k, pred_var_k),
            }

            if valid_degs is not None:
                deg_pred_mu_k = pred_mu_k[valid_degs]
                deg_true_mu_k = true_mu_k[valid_degs]
                deg_pred_var_k = pred_var_k[valid_degs]
                deg_true_var_k = true_var_k[valid_degs]
                substate_entry['r2_top100_deg_mean'] = metrics.r2_score(deg_true_mu_k, deg_pred_mu_k)
                deg_pred_sd_k = np.sqrt(deg_pred_var_k + 1e-8)
                deg_true_sd_k = np.sqrt(deg_true_var_k + 1e-8)
                substate_entry['r2_top100_deg_std'] = metrics.r2_score(deg_true_sd_k, deg_pred_sd_k)

            substate_metrics.append(substate_entry)

        pred_mu_list = np.stack(pred_mu_list, axis=0)
        true_mu_list = np.stack(true_mu_list, axis=0)
        pred_var_list = np.stack(pred_var_list, axis=0)
        true_var_list = np.stack(true_var_list, axis=0)
        pi_array = np.array(pi_list).reshape(-1, 1)

        pred_mu = np.sum(pi_array * pred_mu_list, axis=0)
        true_mu = np.sum(pi_array * true_mu_list, axis=0)
        pred_var = np.sum(pi_array * (pred_var_list + pred_mu_list ** 2), axis=0) - pred_mu ** 2
        true_var = np.sum(pi_array * (true_var_list + true_mu_list ** 2), axis=0) - true_mu ** 2

        pred_var = np.clip(pred_var, 0.0, None)
        true_var = np.clip(true_var, 0.0, None)

        pred_sd = np.sqrt(pred_var + 1e-8)
        true_sd = np.sqrt(true_var + 1e-8)

        r2_substate_mean = float(np.mean([m['r2_mean'] for m in substate_metrics])) if substate_metrics else np.nan
        r2_substate_std = float(np.mean([m['r2_std'] for m in substate_metrics])) if substate_metrics else np.nan
        r2_mixture_mean = metrics.r2_score(true_mu, pred_mu)
        r2_mixture_sd = metrics.r2_score(true_sd, pred_sd)
        mse_mixture_sd = mean_squared_error(true_sd, pred_sd)

        # Substate-level metrics on top 100 DEGs
        top100_deg_mean = np.nan
        top100_deg_std = np.nan
        if valid_degs is not None:
            top100_deg_mean = metrics.r2_score(true_mu[valid_degs], pred_mu[valid_degs])
            top100_deg_std = metrics.r2_score(true_sd[valid_degs], pred_sd[valid_degs])

        # Mixture-level metrics on top 100 DEGs
        r2_mixture_top100_deg_mean = np.nan
        r2_mixture_top100_deg_std = np.nan
        if valid_degs is not None:
            r2_mixture_top100_deg_mean = metrics.r2_score(true_mu[valid_degs], pred_mu[valid_degs])
            r2_mixture_top100_deg_std = metrics.r2_score(true_sd[valid_degs], pred_sd[valid_degs])

        mixture_row = {
            'cov_drug': p,
            'n_valid_substates': len(valid_substates),
            'n_skipped_substates': len(skipped_substates),
            'r2_substate_mean': r2_substate_mean,
            'r2_substate_std': r2_substate_std,
            'r2_substate_top100_deg_mean': top100_deg_mean,
            'r2_substate_top100_deg_std': top100_deg_std,
            'r2_mixture_mean': r2_mixture_mean,
            'r2_mixture_sd': r2_mixture_sd,
            'r2_mixture_top100_deg_mean': r2_mixture_top100_deg_mean,
            'r2_mixture_top100_deg_std': r2_mixture_top100_deg_std,
            'mse_mixture_sd': mse_mixture_sd,
        }
        mixture_summary_rows.append(mixture_row)

        substate_rows.extend(substate_metrics)

        print(
            f"[SubstateEval] {p}: valid_substates={len(valid_substates)}, "
            f"R2_substate_mean={r2_substate_mean:.4f}, "
            f"R2_substate_std={r2_substate_std:.4f}, "
            f"R2_substate_top100_DEG_mean={top100_deg_mean:.4f}, "
            f"R2_substate_top100_DEG_std={top100_deg_std:.4f}, "
            f"R2_mixture_mean={r2_mixture_mean:.4f}, "
            f"R2_mixture_sd={r2_mixture_sd:.4f}, "
            f"R2_mixture_top100_DEG_mean={r2_mixture_top100_deg_mean:.4f}, "
            f"R2_mixture_top100_DEG_std={r2_mixture_top100_deg_std:.4f}, "
            f"MSE_mixture_sd={mse_mixture_sd:.4f}"
        )

    if mixture_summary_rows:
        mixture_df = pd.DataFrame(mixture_summary_rows)
        mixture_df.to_csv(os.path.join(save_path, 'substate_mixture_summary.csv'), index=False)

    if substate_rows:
        substate_df = pd.DataFrame(substate_rows)
        substate_df.to_csv(os.path.join(save_path, 'substate_metrics_per_cov_drug.csv'), index=False)


def Inference_multi_pert(cell_type_network, model, save_path_res,
              ood_loader, adata, degs_dict, device = 'cuda', mean_or_std = True, plot = True, multi_pert = True, use_state_features=False):
    """
    The Inference function

    Parameters
    ----------
    cell_type_network : dict
        Cell type specific graphs.
    model : torch.nn.Module
        The model to use for inference (GNN or SubstateAwareGNN).
    save_path_res : str
        Path to save results.
    ood_loader : DataLoader
        Out-of-distribution data loader.
    adata : AnnData
        Original AnnData object.
    degs_dict : dict
        Dictionary of DEGs per perturbation.
    device : str, default='cuda'
        Device to use.
    mean_or_std : bool, default=True
        If True, evaluate on mean; if False, evaluate on std.
    plot : bool, default=True
        Whether to generate plots.
    multi_pert : bool, default=True
        Whether to use multi-perturbation mode.
    use_state_features : bool, default=False
        Whether to use state features (substate_feat) from samples.
        If True, model should be SubstateAwareGNN and samples must have substate_feat attribute.
        If False, model should be GNN and original behavior is preserved.
    """
    pred = []   # list to collect predictions
    truth = []  # list to collect ground truth values
    substate_ids = []  # list to collect substate identifiers
    print(f'Inference use_state_features={use_state_features}')
    with torch.no_grad():  # disable gradient computation during inference
        model.eval()       # set model to evaluation mode
        cov_drugs = []     # list to store perturbation labels
        for sample in tq.tqdm(ood_loader, leave=False):  # iterate over OOD data
            sample = sample.to(device)   # move batch to device (GPU/CPU)
            cell_type = sample.cell_type
            ctrl = sample.x              # control expression values
            
            # Perturbation labels (if multiple perturbations are supported and available)
            if multi_pert and hasattr(sample, 'pert_label'):
                pert_label = sample.pert_label
            else: 
                pert_label = None
            
            batch = sample.batch
            y = sample.y   # true perturbed expression values
            
            # Build dictionaries for cell-type-specific graphs
            cell_graphs_x = {Cell: cell_type_network[Cell].x.to(device) for Cell in np.unique(cell_type)}
            cell_graphs_pos = {Cell: cell_type_network[Cell].pos.to(device) for Cell in np.unique(cell_type)}
            cell_graphs_edges = {Cell: cell_type_network[Cell].edge_index.to(device) for Cell in np.unique(cell_type)}
            
            if use_state_features:
                if not hasattr(sample, 'substate_feat'):
                    raise ValueError(
                        "use_state_features=True but inference batch does not contain substate_feat. "
                        "Please regenerate cells with state_feature_mode != 'none'."
                    )
                out = model(cell_graphs_x, cell_graphs_edges,
                            cell_type, cell_graphs_edges.keys(), ctrl, pert_label, cell_graphs_pos,
                            substate_feat=sample.substate_feat)
                substate_ids.append(sample.substate_id.detach().cpu().numpy())
            else:
                # Forward pass
                out = model(cell_graphs_x, cell_graphs_edges, 
                            cell_type, cell_graphs_edges.keys(), ctrl, pert_label, cell_graphs_pos)
            
            # Collect predictions, truths, and perturbation identifiers
            pred.extend(out)
            truth.extend(y)
            cov_drugs.extend(sample.cov_drug)
   
    # Convert prediction and truth lists into numpy arrays
    pred = torch.stack(pred)
    truth = torch.stack(truth)
    pred = (pred).cpu().numpy()
    truth = (truth).cpu().numpy()
    
    # Perturbation identifiers for all samples
    perts = np.array(cov_drugs)
    
    if use_state_features:
        if not substate_ids:
            warnings.warn("use_state_features=True but no substate_ids were collected during inference; skipping substate evaluation.")
        else:
            substate_ids = np.concatenate(substate_ids, axis=0)
            evaluate_substate_mixture(
                pred,
                truth,
                perts,
                substate_ids,
                save_path_res,
                degs_dict=degs_dict,
                cell_type_network=cell_type_network,
            )
    
    # Loop through each unique perturbation
    for p in (set(perts)):
        # Split into cell type and drug
        c = p.split('_')[0]
        d = p.split('_')[1]
        
        # Select positions of genes for this cell type
        pos_genes = cell_type_network[c].pos
        
        # Subset predictions and truths to only those genes
        p_pred = pred[:, pos_genes.tolist()] 
        p_truth = truth[:, pos_genes.tolist()]
        
        # Get indices of samples belonging to this perturbation
        pert_idx = np.where(perts == p)[0]
        y_p = p_truth[pert_idx]   # ground truth for this perturbation
        pred_p = p_pred[pert_idx] # predictions for this perturbation
        
        # Create AnnData object with predictions, truth, and control
        Ann_Data = create_anndata(pred_p, y_p, adata, cell_type_network, p)
        
        # Attach DEGs for this perturbation
        DEGs = degs_dict[p]
        Ann_Data.uns['DEGs'] = DEGs
        
        # Save results to disk
        Ann_Data.write(save_path_res+p+'_pred.h5ad')
        
        # Compute mean squared error
        mse = np.mean((y_p - pred_p) ** 2)
        print(p, " mse: ", mse)
        
        # Compute R² either on mean or standard deviation of expression
        if mean_or_std:
            # Mean expression comparison
            x = np.mean(y_p, axis=0)
            y = np.mean(pred_p, axis=0) 
            r2_all = metrics.r2_score(x, y)
            print(f"R² value for predicting the **mean** expression of all genes for perturbation '{p}': {r2_all:.4f}")
            r2_DEGs = metrics.r2_score(x[DEGs], y[DEGs])
            print(f"R² value for predicting the **mean** of the top 100 DEGs for perturbation '{p}': {r2_DEGs:.4f}")
        else: 
            # Standard deviation comparison
            x = np.std(y_p, axis=0) 
            y = np.std(pred_p, axis=0) 
            data_to_plot = np.vstack([x, y])
            r2_all = metrics.r2_score(x, y)
            print(f"R² value for predicting the **standard deviation** expression of all genes for perturbation '{p}': {r2_all:.4f}")
            r2_DEGs = metrics.r2_score(x[DEGs], y[DEGs])
            print(f"R² value for predicting the **standard deviation** of the top 100 DEGs for perturbation '{p}': {r2_DEGs:.4f}")
    
        # Plotting settings
        sns.set_style("darkgrid")
        x_coeff = 0.35
        
        # Generate scatter plot of predicted vs truth statistics
        if plot:
            fig, ax = plt.subplots(figsize=(6,6))
            sns.regplot(x=x, y=y, ci=None, color="#1C2E54")
            
            # Annotate R² for all genes
            y_coeff = 0.8
            ax.text(x.max() - x.max() * x_coeff, y.max() - y_coeff * y.max(),
                    r'$\mathrm{R^2_{\mathrm{\mathsf{all\ genes}}}}$= '+ f"{r2_all:.4f}",
                    fontsize='large')
            
            # Annotate R² for top 100 DEGs
            y_coeff = 0.9
            ax.text(x.max() - x.max() * x_coeff, y.max() - y_coeff * y.max(),
                   r'$\mathrm{R^2_{\mathrm{\mathsf{top\ 100 \ DEGs}}}}$= ' + f"{r2_DEGs:.4f}",
                   fontsize='large')
