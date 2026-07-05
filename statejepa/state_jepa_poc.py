#!/usr/bin/env python3
"""
State-JEPA Proof of Concept (POC).
This script:
1. Loads S&P 500 ETF minute data (IVE_bidask1min.txt).
2. Engineers stationary normalized state sequences: [Log_Return, Spread_Relative, Rolling_Volatility, High_Low_Range, Time_Sin, Time_Cos].
3. Defines a JEPA architecture:
   - Context Encoder (processes historical sequence)
   - Target Encoder (processes future sequence, updated via EMA, no gradients)
   - Predictor (predicts future representations from historical representations)
4. Trains the JEPA model using self-supervised representation learning.
5. Evaluates the learned representations using a downstream 3-way Classification Probe (UP, FLAT, DOWN) to predict future price direction.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# =====================================================================
# 1. Dataset Preprocessing
# =====================================================================

def load_and_preprocess_data(file_path, limit_rows=250000):
    print(f"Loading data from {file_path} (loading first {limit_rows:,} rows)...")
    columns = [
        'Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'BidOpen', 'BidHigh', 'BidLow', 'BidClose',
        'AskOpen', 'AskHigh', 'AskLow', 'AskClose'
    ]
    
    # Read first line to detect number of columns in the actual file
    with open(file_path, 'r') as f:
        num_cols = len(f.readline().split(','))
        
    if num_cols == 10:
        print("Detected 10-column file format. Mapping Bid/Ask fields and creating trade placeholders.")
        actual_cols = [
            'Date', 'Time', 
            'AskOpen', 'AskHigh', 'AskLow', 'AskClose', 
            'BidOpen', 'BidHigh', 'BidLow', 'BidClose'
        ]
        df = pd.read_csv(file_path, header=None, names=actual_cols, nrows=limit_rows)
        # Fill trade column placeholders
        df['Open'] = df['AskOpen']
        df['High'] = df['AskHigh']
        df['Low'] = df['AskLow']
        df['Close'] = (df['AskClose'] + df['BidClose']) / 2.0
        df['Volume'] = 0
    else:
        print("Detected 15-column file format.")
        df = pd.read_csv(file_path, header=None, names=columns, nrows=limit_rows)
    
    # Combine Date and Time to create a Datetime index for temporal features
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df.set_index('Datetime', inplace=True)
    df.drop(columns=['Date', 'Time'], inplace=True)
    
    # Calculate Mid-Price and Spread
    df['Mid_Price'] = (df['AskClose'] + df['BidClose']) / 2.0
    df['Spread'] = df['AskClose'] - df['BidClose']
    df['Log_Return'] = np.log(df['Mid_Price'] / df['Mid_Price'].shift(1))
    
    # 1. Feature: Relative Spread (spread normalized by mid price)
    df['Spread_Relative'] = df['Spread'] / df['Mid_Price']
    
    # 2. Feature: Rolling Volatility (15 minutes)
    df['Rolling_Volatility'] = df['Log_Return'].rolling(window=15).std()
    
    # 3. Feature: High-Low Range normalized by Mid-Price
    df['High_Low_Range'] = ((df['AskHigh'] - df['AskLow']) + (df['BidHigh'] - df['BidLow'])) / (2.0 * df['Mid_Price'])
    
    # 4. Feature: Time-of-day sin/cos features
    df['Minute_of_Day'] = df.index.hour * 60 + df.index.minute
    df['Time_Sin'] = np.sin(2 * np.pi * df['Minute_of_Day'] / 1440.0)
    df['Time_Cos'] = np.cos(2 * np.pi * df['Minute_of_Day'] / 1440.0)
    
    df_clean = df[['Mid_Price', 'Log_Return', 'Spread_Relative', 'Rolling_Volatility', 'High_Low_Range', 'Time_Sin', 'Time_Cos']].dropna()
    
    # Return features and raw mid prices
    features = df_clean[['Log_Return', 'Spread_Relative', 'Rolling_Volatility', 'High_Low_Range', 'Time_Sin', 'Time_Cos']].values
    mid_prices = df_clean['Mid_Price'].values
    return features, mid_prices


def create_jepa_tensors(features, mid_prices, context_len=60, target_len=15):
    # Normalize features using Z-score normalization
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1.0
    normalized = (features - mean) / std

    X, Y = [], []
    future_returns = []
    
    for i in range(len(normalized) - context_len - target_len + 1):
        context = normalized[i : i + context_len]
        target = normalized[i + context_len : i + context_len + target_len]
        X.append(context)
        Y.append(target)
        
        # Continuous return over the next target_len minutes relative to the end of context
        p_end_context = mid_prices[i + context_len - 1]
        p_end_target = mid_prices[i + context_len + target_len - 1]
        ret = (p_end_target - p_end_context) / p_end_context
        future_returns.append(ret)
        
    return (
        torch.tensor(np.array(X), dtype=torch.float32), 
        torch.tensor(np.array(Y), dtype=torch.float32),
        torch.tensor(np.array(future_returns), dtype=torch.float32)
    )


# =====================================================================
# 2. State-JEPA Model Architecture
# =====================================================================

class StateEncoder(nn.Module):
    """
    Encodes a sequence of state vectors into a latent space.
    Input shape: [Batch, Seq_Len, State_Dim]
    Output shape: [Batch, Seq_Len, Embed_Dim]
    """
    def __init__(self, state_dim=6, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.projection = nn.Linear(state_dim, embed_dim)
        # Using a GRU as a simple temporal processor
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, num_layers=1)
        self.out_proj = nn.Linear(hidden_dim, embed_dim)
        
    def forward(self, x):
        # x: [B, S, D]
        proj = F.relu(self.projection(x))  # [B, S, E]
        gru_out, _ = self.gru(proj)        # [B, S, H]
        out = self.out_proj(gru_out)       # [B, S, E]
        return out


class LatentPredictor(nn.Module):
    """
    Predicts the future state representations from the history representation.
    Input: Context Representation [Batch, Embed_Dim] (taken from the end of the context sequence)
    Output: Predicted Future Representations [Batch, Target_Len, Embed_Dim]
    """
    def __init__(self, embed_dim=64, target_len=15, hidden_dim=128):
        super().__init__()
        self.target_len = target_len
        self.embed_dim = embed_dim
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, target_len * embed_dim)
        self.refine = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
    def forward(self, context_latent):
        # context_latent: [B, E]
        h = F.relu(self.fc1(context_latent))
        out = self.fc2(h)  # [B, M * E]
        out = out.view(-1, self.target_len, self.embed_dim)  # [B, M, E]
        out = self.refine(out)  # [B, M, E]
        return out


# Helper function to update the Target Encoder using EMA
def update_ema(context_encoder, target_encoder, decay=0.99):
    with torch.no_grad():
        for param_c, param_t in zip(context_encoder.parameters(), target_encoder.parameters()):
            param_t.data.mul_(decay).add_(param_c.data, alpha=1.0 - decay)


# =====================================================================
# 3. Training and Evaluation Routines
# =====================================================================

def pretrain_jepa(context_encoder, target_encoder, predictor, train_loader, epochs=3, lr=1e-3, device="cpu"):
    print("\n--- Starting JEPA Self-Supervised Pre-training ---")
    context_encoder.train()
    predictor.train()
    target_encoder.eval()  # Target encoder is always in eval mode

    optimizer = optim.Adam(list(context_encoder.parameters()) + list(predictor.parameters()), lr=lr)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (context_x, target_y, _) in enumerate(train_loader):
            context_x = context_x.to(device)
            target_y = target_y.to(device)

            optimizer.zero_grad()

            # 1. Encode context (history)
            context_latents = context_encoder(context_x)  # [B, C, E]
            z_context = context_latents[:, -1, :]          # [B, E]

            # 2. Encode target (true future) - No gradients flow here
            with torch.no_grad():
                z_target_true = target_encoder(target_y)   # [B, M, E]

            # 3. Predict the target representations
            z_target_pred = predictor(z_context)           # [B, M, E]

            # 4. Compute Loss
            loss = F.mse_loss(z_target_pred, z_target_true)

            loss.backward()
            optimizer.step()

            # 5. EMA Update for the target encoder weights
            update_ema(context_encoder, target_encoder, decay=0.99)

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.6f}")


def train_linear_probe(encoder, train_loader, val_loader, epochs=3, lr=1e-3, device="cpu"):
    """
    Evaluates the quality of JEPA representations using a downstream task:
    Predicting future price return direction (UP, FLAT, DOWN) over the next 15 minutes.
    """
    print("\n--- Training Downstream Linear Probe (Evaluation) ---")
    encoder.eval()  # Freeze the encoder

    # 3-way classification probe
    probe = nn.Linear(64, 3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(probe.parameters(), lr=lr)

    for epoch in range(epochs):
        probe.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for context_x, _, batch_labels in train_loader:
            context_x = context_x.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            # Extract frozen representations
            with torch.no_grad():
                context_latents = encoder(context_x)
                z_context = context_latents[:, -1, :]  # [B, E]

            outputs = probe(z_context)
            loss = criterion(outputs, batch_labels)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == batch_labels).sum().item()
            total_train += batch_labels.size(0)

        # Validation Evaluation
        probe.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        val_label_counts = torch.zeros(3)
        
        with torch.no_grad():
            for context_x, _, batch_labels in val_loader:
                context_x = context_x.to(device)
                batch_labels = batch_labels.to(device)
                
                context_latents = encoder(context_x)
                z_context = context_latents[:, -1, :]
                
                outputs = probe(z_context)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == batch_labels).sum().item()
                total_val += batch_labels.size(0)
                
                for label in batch_labels:
                    val_label_counts[label.item()] += 1

        avg_train = train_loss / len(train_loader)
        train_acc = (correct_train / total_train) * 100.0
        val_acc = (correct_val / total_val) * 100.0
        
        # Baseline accuracy: always predicting majority class in the validation set
        majority_class_count = val_label_counts.max().item()
        baseline_acc = (majority_class_count / total_val) * 100.0
        
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train:.6f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Baseline Acc: {baseline_acc:.2f}%")


# =====================================================================
# 4. Main Execution
# =====================================================================

def main():
    file_path = "/Users/loganchoi/Desktop/vjepa2/IVE_bidask1min.txt"
    if not os.path.exists(file_path):
        print(f"Error: Market data file not found at: {file_path}")
        return

    # 1. Load and process S&P 500 ETF data
    # Increase limit_rows to get a robust dataset
    features, mid_prices = load_and_preprocess_data(file_path, limit_rows=250000)
    
    # Create rolling windows (60 mins history -> 15 mins target)
    print("Generating context, target, and future return tensors...")
    X, Y, future_returns = create_jepa_tensors(features, mid_prices, context_len=60, target_len=15)
    print(f"Dataset summary - Context shape: {X.shape}, Target shape: {Y.shape}")

    # Split into Train / Validation sets (80% / 20%)
    split_idx = int(0.8 * len(X))
    
    # Compute 33.3% and 66.7% quantiles on the training set to prevent data leakage
    train_returns = future_returns[:split_idx].numpy()
    lower_threshold = np.percentile(train_returns, 33.33)
    upper_threshold = np.percentile(train_returns, 66.67)
    print(f"Quantile classification thresholds: Lower={lower_threshold:.6f}, Upper={upper_threshold:.6f}")
    
    # Convert returns to discrete labels (0: DOWN, 1: FLAT, 2: UP)
    labels = torch.zeros(len(future_returns), dtype=torch.long)
    labels[future_returns < lower_threshold] = 0
    labels[(future_returns >= lower_threshold) & (future_returns <= upper_threshold)] = 1
    labels[future_returns > upper_threshold] = 2
    
    train_dataset = TensorDataset(X[:split_idx], Y[:split_idx], labels[:split_idx])
    val_dataset = TensorDataset(X[split_idx:], Y[split_idx:], labels[split_idx:])

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # 2. Setup Device
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Selected device: {device}")

    # 3. Instantiate JEPA Models
    embed_dim = 64
    context_encoder = StateEncoder(state_dim=6, embed_dim=embed_dim).to(device)
    target_encoder = StateEncoder(state_dim=6, embed_dim=embed_dim).to(device)
    predictor = LatentPredictor(embed_dim=embed_dim, target_len=15).to(device)

    # Initialize Target Encoder with exact weights of Context Encoder
    target_encoder.load_state_dict(context_encoder.state_dict())

    # 4. Run pre-training
    try:
        pretrain_jepa(context_encoder, target_encoder, predictor, train_loader, epochs=3, lr=1e-3, device=device)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n[Warning] GPU Out of Memory error during pretraining: {e}")
            print("Restarting pretraining on CPU...")
            device = "cpu"
            context_encoder = context_encoder.to(device)
            target_encoder = target_encoder.to(device)
            predictor = predictor.to(device)
            pretrain_jepa(context_encoder, target_encoder, predictor, train_loader, epochs=3, lr=1e-3, device=device)
        else:
            raise e

    # 5. Run evaluation (Linear Probing)
    try:
        train_linear_probe(context_encoder, train_loader, val_loader, epochs=3, lr=1e-3, device=device)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n[Warning] GPU Out of Memory error during probing: {e}")
            print("Restarting probing on CPU...")
            device = "cpu"
            context_encoder = context_encoder.to(device)
            train_linear_probe(context_encoder, train_loader, val_loader, epochs=3, lr=1e-3, device=device)
        else:
            raise e

    print("\nPOC Execution Finished successfully!")


if __name__ == "__main__":
    main()
