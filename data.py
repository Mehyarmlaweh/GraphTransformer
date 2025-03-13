# data.py
# Documentation from ; https://ogb.stanford.edu/docs/lsc/pcqm4mv2/
import torch
from ogb.lsc import PygPCQM4Mv2Dataset
from ogb.utils import smiles2graph as ogb_smiles2graph
from torch_geometric.utils import degree, to_dense_adj
from torch_geometric.data import DataLoader
import numpy as np
from scipy.sparse.csgraph import shortest_path, reconstruct_path
from rdkit import Chem


#custom SMILES processing
def custom_smiles2graph(smiles):
    """
    Wrapper around ogb's smiles2graph to handle invalid SMILES strings.
    Returns an empty graph for invalid SMILES to avoid crashing the dataset processing.
    """
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            print(f"Invalid SMILES '{smiles}' encountered; returning empty graph.")
            # Return an empty graph dictionary compatible with PygPCQM4Mv2Dataset
            return {
                'edge_index': np.array([[], []], dtype=np.int64),  # Shape (2, 0)
                'edge_feat': np.array([], dtype=np.int64),         # Shape (0, edge_dim)
                'node_feat': np.array([], dtype=np.int64),         # Shape (0, node_dim)
                'num_nodes': 0
            }
        return ogb_smiles2graph(smiles)
    except Exception as e:
        print(f"Error processing SMILES '{smiles}': {e}")
        return {
            'edge_index': np.array([[], []], dtype=np.int64),
            'edge_feat': np.array([], dtype=np.int64),
            'node_feat': np.array([], dtype=np.int64),
            'num_nodes': 0
        }

def load_pcqm4m_lsc(batch_size=32):
    # Load PCQM4M-v2 dataset with custom SMILES processing
    dataset = PygPCQM4Mv2Dataset(root="dataset/", smiles2graph=custom_smiles2graph)
    split_idx = dataset.get_idx_split()
    train_dataset = dataset[split_idx["train"]]
    val_dataset = dataset[split_idx["valid"]]

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def get_shortest_path_edges(edge_index, edge_attr, sp_matrix, num_nodes, batch):
    batch_size = batch.max().item() + 1
    edge_dict = {(e[0].item(), e[1].item()): attr for e, attr in zip(edge_index.t(), edge_attr)}
    sp_edges = []

    for b in range(batch_size):
        start_idx = torch.where(batch == b)[0][0].item()
        end_idx = torch.where(batch == b)[0][-1].item() + 1
        nodes_in_batch = end_idx - start_idx
        sub_edge_index = edge_index[:, (edge_index[0] >= start_idx) & (edge_index[0] < end_idx)]
        sub_sp = sp_matrix[b, :nodes_in_batch, :nodes_in_batch]

        adj = to_dense_adj(sub_edge_index - start_idx, max_num_nodes=nodes_in_batch).squeeze(0).numpy()
        predecessors = shortest_path(adj, directed=False, unweighted=True, return_predecessors=True)[1]

        for i in range(nodes_in_batch):
            path_row = []
            for j in range(nodes_in_batch):
                if sub_sp[i, j] == -1 or sub_sp[i, j] == np.inf:
                    path_row.append(torch.zeros(edge_attr.size(1)))
                else:
                    path = reconstruct_path(i, j, predecessors)
                    if len(path) <= 1:
                        if (i + start_idx, j + start_idx) in edge_dict:
                            path_row.append(edge_dict[(i + start_idx, j + start_idx)])
                        elif (j + start_idx, i + start_idx) in edge_dict:
                            path_row.append(edge_dict[(j + start_idx, i + start_idx)])
                        else:
                            path_row.append(torch.zeros(edge_attr.size(1)))
                    else:
                        edge_feats = []
                        for k in range(len(path) - 1):
                            u, v = path[k] + start_idx, path[k + 1] + start_idx
                            if (u, v) in edge_dict:
                                edge_feats.append(edge_dict[(u, v)])
                            elif (v, u) in edge_dict:
                                edge_feats.append(edge_dict[(v, u)])
                        if edge_feats:
                            edge_feats = torch.stack(edge_feats).mean(dim=0)
                            path_row.append(edge_feats)
                        else:
                            path_row.append(torch.zeros(edge_attr.size(1)))
            sp_edges.append(torch.stack(path_row))
    
    return torch.stack(sp_edges)

def preprocess_batch(batch):
    edge_index = batch.edge_index
    edge_attr = batch.edge_attr
    num_nodes = batch.num_nodes
    batch_size = batch.num_graphs
    y = batch.y

    deg = degree(edge_index[0], num_nodes=num_nodes).long()
    max_degree = 100
    deg = torch.clamp(deg, max=max_degree)

    adj = to_dense_adj(edge_index, batch=batch.batch, max_num_nodes=num_nodes)
    adj = adj.numpy()
    sp_matrix = np.zeros((batch_size, num_nodes, num_nodes))
    for i in range(batch_size):
        sp_matrix[i] = shortest_path(adj[i], directed=False, unweighted=True)
    sp_matrix[sp_matrix == np.inf] = -1
    sp_matrix = torch.tensor(sp_matrix, dtype=torch.long)

    edge_encoding = get_shortest_path_edges(edge_index, edge_attr, sp_matrix, num_nodes, batch.batch)

    return {
        "x": batch.x,
        "edge_index": edge_index,
        "edge_attr": edge_encoding,
        "deg": deg,
        "sp": sp_matrix,
        "y": y,
        "batch": batch.batch,
        "num_nodes": num_nodes
    }

if __name__ == "__main__":
    train_loader, val_loader = load_pcqm4m_lsc(batch_size=2)
    for batch in train_loader:
        data = preprocess_batch(batch)
        print(data["edge_attr"].shape)
        break