# unittests/test_model.py
import pytest
import torch
from ..data import load_pcqm4m_lsc, preprocess_batch
from ..model import Graphormer

@pytest.fixture
def model_and_data():
    train_loader, _ = load_pcqm4m_lsc(batch_size=2)
    model = Graphormer(input_dim=9, hidden_dim=128, num_layers=2, num_heads=4, edge_dim=3)
    for batch in train_loader:
        data = preprocess_batch(batch)
        return model, data

def test_model_forward(model_and_data):
    model, data = model_and_data
    out = model(data)
    assert out.shape == (2, 1), f"Expected output shape [2, 1], got {out.shape}"
    assert torch.isfinite(out).all(), "Output contains NaN or Inf"

def test_model_empty_graph(model_and_data):
    model, _ = model_and_data
    empty_data = {
        "x": torch.empty(0, 9),
        "edge_index": torch.empty(2, 0, dtype=torch.long),
        "edge_attr": torch.zeros(1, 1, 1, 3),
        "deg": torch.zeros(0, dtype=torch.long),
        "sp": torch.zeros(1, 1, 1, dtype=torch.long),
        "y": torch.zeros(1, 1),
        "batch": torch.empty(0, dtype=torch.long),
        "num_nodes": 0
    }
    out = model(empty_data)
    assert out.shape == (1, 1), f"Expected output shape [1, 1] for empty graph, got {out.shape}"
    assert torch.all(out == 0), "Empty graph output should be zero"

if __name__ == "__main__":
    pytest.main(["-v", __file__])