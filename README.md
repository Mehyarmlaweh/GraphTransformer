# GraphTransformer Project
This project implements a Graphormer model for the PCQM4Mv2 dataset from the Open Graph Benchmark (OGB), designed for predicting molecular properties (HOMO-LUMO gap). It includes data loading, model definition, training, and testing components.

## Project Overview

- **Goal**: Train a Graphormer model to achieve competitive Mean Absolute Error (MAE) on the PCQM4Mv2 dataset.
- **Features**:
  - Custom SMILES preprocessing to handle invalid molecules.
  - Multi-head attention with spatial and edge encodings.
  - Training with logging, checkpointing, and test prediction generation.
  - Unit tests for data loading and model functionality.

## Directory Structure
```
    GraphTransformer/
    ├── data.py           # Data loading and preprocessing for PCQM4Mv2
    ├── model.py          # Graphormer model definition
    ├── train.py          # Training script with logging and command-line options
    ├── requirements.txt  # Python dependencies
    ├── unittests/        # Unit tests directory
    │   └── test_data.py  # Tests for data loader
    ├── graphvenv/        # Virtual environment
    └── dataset/          # PCQM4Mv2 dataset (auto-downloaded on first run)
```

## Prerequisites

- **Operating System**: Windows (tested), Linux/macOS compatible.
- **Python**: 3.11+ (tested with 3.11.9).
- **Hardware**: CPU sufficient for small-scale testing; GPU recommended for full training.

## Setup Instructions

1. **Clone or Setup Directory**:
   - If using a repo: `git clone https://github.com/Mehyarmlaweh/GraphTransformer.git`
   - Otherwise, ensure all files are in `GraphTransformer/`.

2. **Create Virtual Environment**:
```   
    cd "\GraphTransformer"
    python -m venv graphvenv
```

3. **Activate Virtual Environment**:
```   
    .\graphvenv\Scripts\activate
```
    - Prompt should show `(graphvenv)`.

3. **Install Dependencies**:
```
    pip install -r requirements.txt
```


## Files and Their Purpose

- **`data.py`**:
- Loads the PCQM4Mv2 dataset using `PygPCQM4Mv2Dataset`.
- Preprocesses batches with shortest path and edge encodings.
- Handles invalid SMILES strings with a custom `smiles2graph` function.
- Test: `python data.py` (outputs tensor shapes).

- **`model.py`**:
- Defines the `Graphormer` model with multi-head attention, centrality, spatial, and edge encodings.
- Includes a virtual node for graph-level predictions.
- Test: `python model.py` (processes a batch and outputs `[batch_size, 1]`).

- **`train.py`**:
- Trains the model with configurable parameters via command-line arguments.
- Logs progress to `training.log`, saves best and final models, and generates test predictions.
- Uses `PCQM4Mv2Evaluator` for evaluation.

- **`unittests/test_data.py`**:
- Tests data loading and preprocessing using `ogbg-molhiv` (smaller dataset for speed).

- **`requirements.txt`**:
- Lists all required Python packages.

## Usage

### Running Tests
- Test data loading:
```
    pytest unittests/test_data.py -v
    - Downloads `ogbg-molhiv` (~1.5 GB) on first run.
```
### Training the Model
- **Default Run** (10 epochs, batch size 32):
   ```
    python train.py
    ```
- **Custom Run**:
- Example: 20 epochs, batch size 16, larger model:
```
    python train.py --batch_size 16 --epochs 20 --hidden_dim 256 --num_layers 8 --num_heads 8 --lr 1e-4 --output_dir "experiment1"
```

    
- **Command-Line Options**:
- `--batch_size`: Number of graphs per batch (default: 32).
- `--epochs`: Training iterations (default: 10).
- `--hidden_dim`: Model hidden layer size (default: 128).
- `--num_layers`: Transformer layers (default: 6).
- `--num_heads`: Attention heads (default: 8).
- `--lr`: Learning rate (default: 2e-4).
- `--output_dir`: Directory for test predictions (default: "submission").

### Outputs
- **`training.log`**: Logs training progress (e.g., `2025-03-12 10:00:00 - Epoch 1/10, Train Loss: 1.2345, Val MAE: 0.5678`).
- **`graphormer_best.pth`**: Best model weights based on validation MAE.
- **`graphormer_final.pth`**: Final model weights after training.
- **`<output_dir>/y_pred_pcqm4m-v2_test-dev.npz`**: Test predictions in OGB submission format.

### First Run Notes
- **Dataset Download**: `data.py` downloads `pcqm4m-v2.zip` (~60 MB) on first run.
- **Preprocessing**: Converts ~3.7M SMILES strings to graphs (takes hours, ~14 GB disk space). Subsequent runs use preprocessed data.

## Example Workflow

1. **Setup**:
```
    .\graphvenv\Scripts\activate
    pip install -r requirements.txt
```
2. **Test Data**:
```
    pytest unittests/test_data.py -v
```

3. **Train Model**:

```
python train.py --batch_size 16 --epochs 20
```


4. **Check Results**:
- View `training.log` for training history.
- Use `graphormer_best.pth` for further analysis or submission.

## Tips for Full Training

- **Epochs**: Increase to 300+ for competitive MAE (~0.12), requires GPU.
- **Hardware**: Use CUDA-enabled GPU for faster training:
```
    python train.py --batch_size 64 --epochs 300 --hidden_dim 768
```

- **Memory**: Reduce batch size (e.g., 8) if running on CPU to avoid memory errors.

## Troubleshooting

- **ModuleNotFoundError**: Ensure `graphvenv` is active and all dependencies are installed.
- **SMILES Errors**: `data.py` handles invalid SMILES; check console for skipped molecules.
- **Memory Issues**: Lower `--batch_size` or use a machine with more RAM/GPU.
- **Test Failures**: Verify `ogbg-molhiv` downloaded correctly in `dataset/`.

## Extending the Project
- **Tuning**: Integrate `optuna` for hyperparameter optimization.

## License
This code is an implementation of the Graphormer model described in the paper : [Do Transformers Really Perform Badly for Graph Representation ?](https://arxiv.org/abs/2106.05234). For licensing details, see the [LICENSE](LICENSE).
