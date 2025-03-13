# data.py
import torch
from ogb.lsc import PygPCQM4Mv2Dataset  # Correct class for PCQM4Mv2
from torch_geometric.utils import degree, to_dense_adj
from torch_geometric.data import DataLoader
import numpy as np
from scipy.sparse.csgraph import shortest_path, reconstruct_path

def load_pcqm4m_lsc(batch_size=32):
    # Load PCQM4M-v2 dataset with default graph conversion
    dataset = PygPCQM4Mv2Dataset(root="dataset/")
    split_idx = dataset.get_idx_split()
    train_dataset = dataset[split_idx["train"]]
    val_dataset = dataset[split_idx["valid"]]

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def get_shortest_path_edges(edge_index, edge_attr, sp_matrix, num_nodes, batch):
    """
    Reconstruct shortest paths and collect edge features for each node pair.
    Returns a tensor of edge features along shortest paths.
    """
    batch_size = batch.max().item() + 1
    edge_dict = {(e[0].item(), e[1].item()): attr for e, attr in zip(edge_index.t(), edge_attr)}
    sp_edges = []

    for b in range(batch_size):
        start_idx = torch.where(batch == b)[0][0].item()
        end_idx = torch.where(batch == b)[0][-1].item() + 1
        nodes_in_batch = end_idx - start_idx
        sub_edge_index = edge_index[:, (edge_index[0] >= start_idx) & (edge_index[0] < end_idx)]
        sub_sp = sp_matrix[b, :nodes_in_batch, :nodes_in_batch]

        # Convert to dense adjacency for path reconstruction
        adj = to_dense_adj(sub_edge_index - start_idx, max_num_nodes=nodes_in_batch).squeeze(0).numpy()
        predecessors = shortest_path(adj, directed=False, unweighted=True, return_predecessors=True)[1]

        # For each node pair, reconstruct path and collect edge features
        for i in range(nodes_in_batch):
            path_row = []
            for j in range(nodes_in_batch):
                if sub_sp[i, j] == -1 or sub_sp[i, j] == np.inf:  # Disconnected
                    path_row.append(torch.zeros(edge_attr.size(1)))
                else:
                    path = reconstruct_path(i, j, predecessors)
                    if len(path) <= 1:  # Same node or adjacent
                        if (i + start_idx, j + start_idx) in edge_dict:
                            path_row.append(edge_dict[(i + start_idx, j + start_idx)])
                        elif (j + start_idx, i + start_idx) in edge_dict:
                            path_row.append(edge_dict[(j + start_idx, i + start_idx)])
                        else:
                            path_row.append(torch.zeros(edge_attr.size(1)))
                    else:
                        # Collect edge features along the path
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

    # Centrality Encoding: Degree of nodes
    deg = degree(edge_index[0], num_nodes=num_nodes).long()
    max_degree = 100  # Cap degree
    deg = torch.clamp(deg, max=max_degree)

    # Spatial Encoding: Shortest Path Distances
    adj = to_dense_adj(edge_index, batch=batch.batch, max_num_nodes=num_nodes)
    adj = adj.numpy()
    sp_matrix = np.zeros((batch_size, num_nodes, num_nodes))
    for i in range(batch_size):
        sp_matrix[i] = shortest_path(adj[i], directed=False, unweighted=True)
    sp_matrix[sp_matrix == np.inf] = -1  # Disconnected nodes
    sp_matrix = torch.tensor(sp_matrix, dtype=torch.long)

    # Edge Encoding: Aggregate edge features along shortest paths
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
        print(data["edge_attr"].shape)  # [batch_size, num_nodes, num_nodes, edge_dim]
        break