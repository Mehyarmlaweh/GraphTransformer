# unittests/test_train.py
import pytest
import torch
from ..data import load_pcqm4m_lsc
from ..model import Graphormer
from ..train import train_epoch, evaluate

@pytest.fixture
def setup_training():
    train_loader, val_loader = load_pcqm4m_lsc(batch_size=2)
    model = Graphormer(input_dim=9, hidden_dim=64, num_layers=2, num_heads=4, edge_dim=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    device = torch.device("cpu")
    from ogb.lsc import PCQM4Mv2Evaluator
    evaluator = PCQM4Mv2Evaluator()
    return model, train_loader, val_loader, optimizer, device, evaluator

def test_train_epoch(setup_training):
    model, train_loader, _, optimizer, device, _ = setup_training
    initial_loss = train_epoch(model, train_loader, optimizer, device)
    assert initial_loss > 0, "Training loss should be positive"
    assert torch.isfinite(torch.tensor(initial_loss)), "Training loss should be finite"

def test_evaluate(setup_training):
    model, _, val_loader, _, device, evaluator = setup_training
    mae = evaluate(model, val_loader, device, evaluator)
    assert mae > 0, "Validation MAE should be positive"
    assert torch.isfinite(torch.tensor(mae)), "Validation MAE should be finite"

if __name__ == "__main__":
    pytest.main(["-v", __file__])