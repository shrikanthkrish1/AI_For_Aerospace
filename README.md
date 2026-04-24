# 📚 Project Notebooks — Overview

This repository contains four independent machine learning and simulation notebooks spanning physics-informed neural networks, graph deep learning, and classical ML regression.

---

## 📁 Contents

| # | Notebook | Domain | Model / Method |
|---|----------|--------|----------------|
| 1 | `predictive-maintainance-file.ipynb` | Industrial IoT | LightGBM Classifier |
| 2 | `Warp_Bubble_Simulation.ipynb` | Theoretical Physics | Physics-Informed Neural Network (PINN) |
| 3 | `Predicting_Speed_of_gas_particles_using_LGBM_Model.ipynb` | Experimental Physics | LightGBM Regressor |
| 4 | `Mesh_Transformer.ipynb` | Scientific ML / Simulation | Graph Neural Network + Transformer |

---

## 1. 🔧 Predictive Maintenance — Machine Failure Forecasting

**File:** `predictive-maintainance-file.ipynb`

### What it does
Predicts whether an industrial device will **fail within the next 45 days** using historical sensor readings. The goal is to flag high-risk devices early so maintenance teams can intervene before a breakdown occurs.

### Dataset
- Source: [Kaggle — Predictive Maintenance Dataset](https://www.kaggle.com/datasets/hiimanshuagarwal/predictive-maintenance-dataset)
- Columns: `date`, `device`, `metric1`–`metric9`, `failure`

### Approach
1. **Target engineering** — creates a rolling 45-day failure horizon label per device
2. **Feature engineering** — for each of 9 sensor metrics, generates lag-1, day-over-day diff, 7-day rolling mean/std, and device-normalised z-score
3. **Time-based train/test split** — cutoff at `2015-07-15` to prevent data leakage
4. **Model** — `LGBMClassifier` with inverse-frequency class weighting for imbalanced data
5. **Probability calibration** — isotonic regression calibration via `CalibratedClassifierCV`
6. **Evaluation** — Precision/Recall across thresholds, PR-AUC, and Top-K catch rate analysis

### Key libraries
`pandas`, `numpy`, `lightgbm`, `scikit-learn`

---

## 2. 🌌 Warp Bubble Simulation (Physics-Informed Neural Network)

**File:** `Warp_Bubble_Simulation.ipynb`

### What it does
Simulates the **Alcubierre warp drive metric** using a Physics-Informed Neural Network (PINN). Rather than solving the differential equations numerically, a neural network is trained to satisfy the underlying physics equations as its loss function.

### Physics background
The [Alcubierre metric](https://en.wikipedia.org/wiki/Alcubierre_drive) is a theoretical general-relativity solution that describes a "warp bubble" — a region of spacetime that contracts ahead of a spacecraft and expands behind it. This notebook models the warp field intensity and energy density across space and time.

### Approach
1. **Network** — a 2-layer MLP using `sin` activations (well-suited for oscillatory physical fields)
2. **Inputs** — spatial coordinate `x` and time `t`
3. **Outputs** — `warp_field` and `energy_density` at each (x, t) point
4. **Loss function** — residuals from the Alcubierre PDE and Einstein field equation constraint
5. **Training** — 10,000 epochs with Adam optimiser
6. **Visualisation** — animated plot of warp field shape evolving over time (rendered as HTML in-notebook)

### Key libraries
`torch`, `matplotlib`, `numpy`, `IPython`

---

## 3. ⚛️ Predicting Speed of Gas Particles (LightGBM Regressor)

**File:** `Predicting_Speed_of_gas_particles_using_LGBM_Model.ipynb`

### What it does
Predicts the **speed of gas particles** from physical experiment measurements using a gradient boosting regression model.

### Dataset
- File: `Physical experiments.csv` (semicolon-separated, comma decimals)
- Features: `X Axis`, `Y Axis`, `Z Axis`, `std speed`, `flashes`, `height`, `flash time`, `number of flashes`, `time`, `direction`
- Target: `speed`

### Approach
1. **EDA** — histogram and box-plot inspection for all numeric features, correlation heatmap
2. **Preprocessing** — null imputation with zeros, label encoding for the `direction` categorical column
3. **Train/test split** — 70/30 random split
4. **Model** — `LGBMRegressor` with default hyperparameters
5. **Evaluation** — R² score and Mean Squared Error; predicted vs actual speed line plot

### Key libraries
`pandas`, `numpy`, `lightgbm`, `scikit-learn`, `seaborn`, `matplotlib`

---

## 4. 🕸️ Mesh Transformer — Graph Neural Network + Transformer

**File:** `Mesh_Transformer.ipynb`

### What it does
Implements a **Mesh Transformer** architecture designed for simulation tasks on graph-structured data (e.g. fluid dynamics, structural mechanics). It combines a Graph Neural Network for local message passing with a Transformer for global, long-range interactions across graph clusters.

### Architecture

```
Input Graph (nodes + edges)
        ↓
  GraphEncoder          — embeds raw node and edge features into a latent space
        ↓
  GraphNetwork          — message-passing GNN (mean aggregation)
        ↓
  GraphPooling          — GRU-based cluster pooling to coarsen the graph
        ↓
  TransformerLayers     — multi-head self-attention over cluster embeddings
        ↓
  GraphDecoder          — GNN + MLP to produce node-level predictions
        ↓
Output (per-node predictions)
```

### Components

| Module | Role |
|--------|------|
| `GraphEncoder` | Linear MLP to project node/edge features to `hidden_dim` |
| `GraphNetwork` | `MessagePassing` layer — updates edge and node embeddings |
| `GraphPooling` | Pools node groups into cluster-level representations via GRU |
| `TransformerLayer` | Multi-head attention + feed-forward network on cluster embeddings |
| `GraphDecoder` | Maps latent node features back to output predictions |
| `MeshTransformer` | Full end-to-end pipeline assembling all components |

### Example usage
```python
model = MeshTransformer(
    node_input_dim=3,       # e.g. position, velocity, pressure
    edge_input_dim=2,       # e.g. relative position and distance
    hidden_dim=16,
    output_dim=2,           # e.g. future velocity and pressure
    num_heads=4,
    num_transformer_layers=2
)
output = model(x, edge_index, edge_attr, clusters)
# output shape: (num_nodes, output_dim)
```

### Key libraries
`torch`, `torch_geometric`

---

## 🛠️ Requirements

```bash
# Core
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm

# Deep learning
pip install torch torchvision

# Graph neural networks
pip install torch-geometric

# Notebook display
pip install ipython
```

> **Note:** `torch-geometric` installation depends on your CUDA version. See the [official install guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

---

## 📝 Notes

- The **Predictive Maintenance** and **Gas Particles** notebooks are Kaggle/CSV-dependent — update the file paths before running locally.
- The **Warp Bubble** notebook produces an in-notebook HTML animation; run in Jupyter or Colab for full output.
- The **Mesh Transformer** notebook uses synthetic random data for demonstration and can be plugged into any real graph-structured simulation dataset.
