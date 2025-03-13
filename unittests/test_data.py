# test_data.py
import pytest
import torch
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader
from ..data import load_pcqm4m_lsc, preprocess_batch  # Relative import

# Override load_pcqm4m_lsc to use a smaller dataset for testing
def load_test_dataset(batch_size=2):
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root="dataset/")
    split_idx = dataset.get_idx_split()
    train_dataset = dataset[split_idx["train"]]
    val_dataset = dataset[split_idx["valid"]]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

@pytest.fixture
def data_loader():
    train_loader, val_loader = load_test_dataset(batch_size=2)
    return train_loader, val_loader

def test_load_pcqm4m_lsc(data_loader):
    train_loader, val_loader = data_loader
    assert len(train_loader) > 0, "Train loader is empty"
    assert len(val_loader) > 0, "Validation loader is empty"
    for batch in train_loader:
        assert hasattr(batch, "x"), "Batch missing node features"
        assert hasattr(batch, "edge_index"), "Batch missing edge index"
        assert hasattr(batch, "edge_attr"), "Batch missing edge attributes"
        assert hasattr(batch, "y"), "Batch missing labels"
        assert hasattr(batch, "batch"), "Batch missing batch tensor"
        assert batch.num_graphs == 2, f"Expected batch size 2, got {batch.num_graphs}"
        break

def test_preprocess_batch(data_loader):
    train_loader, _ = data_loader
    for batch in train_loader:
        data = preprocess_batch(batch)
        expected_keys = {"x", "edge_index", "edge_attr", "deg", "sp", "y", "batch", "num_nodes"}
        assert set(data.keys()) == expected_keys, f"Missing keys: {expected_keys - set(data.keys())}"
        assert isinstance(data["x"], torch.Tensor), "x should be a tensor"
        assert isinstance(data["edge_index"], torch.Tensor), "edge_index should be a tensor"
        assert isinstance(data["edge_attr"], torch.Tensor), "edge_attr should be a tensor"
        assert isinstance(data["deg"], torch.Tensor), "deg should be a tensor"
        assert isinstance(data["sp"], torch.Tensor), "sp should be a tensor"
        assert isinstance(data["y"], torch.Tensor), "y should be a tensor"
        assert isinstance(data["batch"], torch.Tensor), "batch should be a tensor"
        assert isinstance(data["num_nodes"], int), "num_nodes should be an int"
        num_nodes = data["num_nodes"]
        batch_size = batch.num_graphs
        assert data["x"].shape[0] == batch.x.shape[0], "Node features shape mismatch"
        assert data["edge_index"].shape == batch.edge_index.shape, "Edge index shape mismatch"
        assert data["edge_attr"].shape[0] == batch_size, "Edge attr batch size mismatch"
        assert data["edge_attr"].shape[1] == num_nodes, "Edge attr num_nodes mismatch"
        assert data["edge_attr"].shape[2] == num_nodes, "Edge attr num_nodes mismatch"
        assert data["edge_attr"].shape[3] == batch.edge_attr.shape[1], "Edge attr feature dim mismatch"
        assert data["deg"].shape[0] == num_nodes, "Degree shape mismatch"
        assert data["sp"].shape == (batch_size, num_nodes, num_nodes), "Shortest path matrix shape mismatch"
        assert data["y"].shape == (batch_size, 1), "Labels shape mismatch"
        assert data["batch"].shape[0] == num_nodes, "Batch tensor shape mismatch"
        break

def test_data_integrity(data_loader):
    train_loader, _ = data_loader
    for batch in train_loader:
        data = preprocess_batch(batch)
        assert data["x"].numel() > 0, "Node features are empty"
        assert data["edge_index"].numel() > 0, "Edge index is empty"
        assert data["edge_attr"].numel() > 0, "Edge attributes are empty"
        assert data["deg"].numel() > 0, "Degrees are empty"
        assert data["sp"].numel() > 0, "Shortest path matrix is empty"
        assert data["y"].numel() > 0, "Labels are empty"
        assert torch.all(data["deg"] >= 0), "Degrees contain negative values"
        assert torch.all(data["sp"] >= -1), "Shortest path matrix contains invalid values"
        break

if __name__ == "__main__":
    pytest.main(["-v", __file__])