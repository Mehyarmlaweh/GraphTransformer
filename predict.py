# predict.py
import torch
from data import load_pcqm4m_lsc, preprocess_batch
from model import Graphormer
from ogb.lsc import PCQM4Mv2Evaluator
from torch_geometric.data import DataLoader
from tqdm import tqdm

def predict(model, loader, device):
    model.eval()
    y_pred = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            data = preprocess_batch(batch)
            data = {k: v.to(device) for k, v in data.items()}
            out = model(data)
            y_pred.append(out.cpu())
    return torch.cat(y_pred).numpy()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate predictions with trained Graphormer")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--output_dir", type=str, default="submission", help="Output directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Graphormer(input_dim=9, hidden_dim=128, num_layers=6, num_heads=8, edge_dim=3).to(device)
    model.load_state_dict(torch.load(args.model_path))
    _, val_loader = load_pcqm4m_lsc(batch_size=args.batch_size)
    evaluator = PCQM4Mv2Evaluator()

    y_pred = predict(model, val_loader, device)
    evaluator.save_test_submission({"y_pred": y_pred}, args.output_dir, mode="test-dev")
    print(f"Predictions saved to {args.output_dir}/y_pred_pcqm4m-v2_test-dev.npz")