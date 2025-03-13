# train.py
import argparse
import logging
import torch
import torch.nn as nn
from data import load_pcqm4m_lsc, preprocess_batch
from model import Graphormer
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from ogb.lsc import PCQM4Mv2Evaluator
from torch_geometric.data import DataLoader
from tqdm import tqdm

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        data = preprocess_batch(batch)
        data = {k: v.to(device) for k, v in data.items()}
        optimizer.zero_grad()
        out = model(data)
        loss = nn.L1Loss()(out, data["y"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device, evaluator):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            data = preprocess_batch(batch)
            data = {k: v.to(device) for k, v in data.items()}
            out = model(data)
            y_true.append(data["y"].cpu())
            y_pred.append(out.cpu())
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    result = evaluator.eval({"y_true": y_true, "y_pred": y_pred})
    return result["mae"]

def predict_test(model, loader, device, evaluator, dir_path="submission"):
    model.eval()
    y_pred = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting Test Set", leave=False):
            data = preprocess_batch(batch)
            data = {k: v.to(device) for k, v in data.items()}
            out = model(data)
            y_pred.append(out.cpu())
    y_pred = torch.cat(y_pred).numpy()
    evaluator.save_test_submission({"y_pred": y_pred}, dir_path, mode="test-dev")
    logging.info(f"Test predictions saved to {dir_path}/y_pred_pcqm4m-v2_test-dev.npz")

# read comments to understand the command-line arguments
def main():
    # cmd-line argument parser
    parser = argparse.ArgumentParser(description="Train Graphormer on PCQM4Mv2")

    # Batch size: Controls how many graphs are processed at once (default: 32)
    # Example: --batch_size 16 
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

    # Epochs: Number of training iterations over the dataset (default: 10)
    # Example: --epochs 20 
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")

    # Hidden dimension: Size of the model's hidden layers (default: 128)
    # Example: --hidden_dim 256 
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")

    # Number of layers: Depth of the transformer (default: 6)
    # Example: --num_layers 8 
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers")

    # Number of attention heads: Parallel attention mechanisms (default: 8)
    # Example: --num_heads 4 
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")

    # Learning rate: Step size for optimization (default: 0.0002)
    # Example: --lr 1e-4 for a smaller learning rate
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")

    # Output directory: Where test predictions are saved (default: "submission")
    # Example: --output_dir "results" 
    parser.add_argument("--output_dir", type=str, default="submission", help="Directory for test submission")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        filename="training.log",
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        filemode="a"  # log file
    )
    

    logging.info(f"Starting training with batch_size={args.batch_size}, epochs={args.epochs}, "
                 f"hidden_dim={args.hidden_dim}, num_layers={args.num_layers}, num_heads={args.num_heads}, "
                 f"lr={args.lr}")

    # GPU if available else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load training and validation data
    train_loader, val_loader = load_pcqm4m_lsc(batch_size=args.batch_size)

    # Load full dataset to access test split for predictions
    dataset = train_loader.dataset.dataset  # Access the full PygPCQM4Mv2Dataset
    split_idx = dataset.get_idx_split()
    test_loader = DataLoader(dataset[split_idx["test-dev"]], batch_size=args.batch_size, shuffle=False)

    # Initialize model with command-line parameters
    model = Graphormer(
        input_dim=9,  # Fixed for PCQM4Mv2 node features
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        edge_dim=3   
    ).to(device)

    # Setup optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = LinearLR(optimizer, start_factor=1e-4, end_factor=1.0, total_iters=60000)
    evaluator = PCQM4Mv2Evaluator()  # OGB evaluator for PCQM4Mv2

    # Training loop
    best_val_mae = float("inf")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_mae = evaluate(model, val_loader, device, evaluator)
        scheduler.step()

        log_msg = f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val MAE: {val_mae:.4f}"
        print(log_msg)
        logging.info(log_msg)

        # Save the best model based on validation MAE
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), "graphormer_best.pth")
            logging.info(f"Saved best model with Val MAE: {best_val_mae:.4f}")
            print(f"Saved best model with Val MAE: {best_val_mae:.4f}")

    torch.save(model.state_dict(), "graphormer_final.pth")
    logging.info("Saved final model to graphormer_final.pth")

    predict_test(model, test_loader, device, evaluator, dir_path=args.output_dir)
    print(f"Test predictions saved to {args.output_dir}/y_pred_pcqm4m-v2_test-dev.npz")

if __name__ == "__main__":
    main()