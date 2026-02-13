# -*- coding: utf-8 -*-
"""
Script to visualize the attention mechanism of the trained Transformer model.

This script will:
1. Load a trained model.
2. Load a sample from the dataset.
3. Run a forward pass and extract attention weights.
4. Plot the input settlement curve, its gradient, and the attention weights
   to analyze which parts of the input the model "focuses" on.
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- Import necessary classes from train_model.py ---
# Note: This assumes the script is run from the 'dl_model' directory.
# If running from the root, paths might need adjustment.

# Paths
DATASET_DIR = "../final_dataset"
MODEL_PATH = "trained_models/best_model.pth"
OUTPUT_DIR = "evaluation_results"

# --- 1. Dataset and Model Definitions (Copied from train_model.py) ---

class NormalizeTransform:
    """A picklable transform class for normalization."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / (self.std + 1e-8)

class FractureDataset(Dataset):
    """Custom dataset for loading the .npz samples."""
    def __init__(self, npz_file_list, transform=None):
        self.file_list = npz_file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = self.file_list[idx]
        with np.load(filepath) as data:
            # Return original x for plotting, and transformed x for model input
            original_x = torch.from_numpy(data['x'].astype(np.float32))
            y = torch.from_numpy(data['y'].astype(np.float32))
        
        transformed_x = original_x
        if self.transform:
            transformed_x = self.transform(original_x)
            
        return original_x, transformed_x, y

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerSurrogateModel(nn.Module):
    def __init__(self, input_size, output_size, d_model=128, nhead=8, num_encoder_layers=4, dim_feedforward=512, dropout=0.1):
        super(TransformerSurrogateModel, self).__init__()
        self.d_model = d_model
        self.feature_embedder = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=input_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.output_layer = nn.Linear(d_model, output_size)

    def forward(self, x):
        # This is the standard forward pass. We will create a new one for attention.
        x = x.unsqueeze(-1)
        x = self.feature_embedder(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        output = self.output_layer(x)
        return output

    def forward_with_attention(self, x):
        """
        A modified forward pass that returns the attention weights from each layer.
        """
        x = x.unsqueeze(-1)
        x = self.feature_embedder(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        attention_maps = []
        output = x
        for layer in self.transformer_encoder.layers:
            # We need to manually replicate the forward pass of TransformerEncoderLayer
            # to get access to the attention weights.
            
            # Self-attention block
            attn_output, attn_weights = layer.self_attn(output, output, output, need_weights=True, average_attn_weights=False)
            attention_maps.append(attn_weights)
            output = output + layer.dropout1(attn_output)
            output = layer.norm1(output)
            
            # Feed-forward block
            ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(output))))
            output = output + layer.dropout2(ff_output)
            output = layer.norm2(output)

        # Continue with the rest of the original forward pass
        x_agg = output.mean(dim=1)
        final_output = self.output_layer(x_agg)
        
        return final_output, attention_maps


# --- 2. Visualization Function ---

def visualize_attention(model, dataset, device, sample_idx=0, save_name_suffix=None, show_plot=False):
    """
    Loads a sample, runs the model, and plots the attention weights.
    """
    NUM_PARAMS = 11 # Number of rock/geology parameters at the start of the vector

    # Get a single sample
    original_x, transformed_x, target_y = dataset[sample_idx]
    
    # Add batch dimension and send to device
    original_x = original_x.unsqueeze(0).to(device)
    transformed_x = transformed_x.unsqueeze(0).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Run forward pass to get attention
    with torch.no_grad():
        prediction, attention_maps = model.forward_with_attention(transformed_x)

    # --- Data Processing for Plotting ---
    original_x_np = original_x.squeeze(0).cpu().numpy()
    
    # Separate parameters from the subsidence curve
    subsidence_curve = original_x_np[NUM_PARAMS:]

    # We'll visualize the attention from the LAST layer, as it's most informative
    # Shape is (batch_size, num_heads, seq_len, seq_len)
    last_layer_attention = attention_maps[-1].squeeze(0).cpu().numpy()
    # Average over the heads
    avg_head_attention = np.mean(last_layer_attention, axis=0)
    # To find the importance of each input token, we sum the attention it receives
    # from all output tokens. This means summing over the rows (axis=0).
    attention_score_full = np.sum(avg_head_attention, axis=0)

    # Extract the part of attention corresponding to the subsidence curve
    attention_score_subsidence = attention_score_full[NUM_PARAMS:]
    
    # Handle cases where attention is uniform (max == min)
    min_score, max_score = np.min(attention_score_subsidence), np.max(attention_score_subsidence)
    if max_score > min_score:
        attention_score_subsidence = (attention_score_subsidence - min_score) / (max_score - min_score)
    else:
        attention_score_subsidence = np.zeros_like(attention_score_subsidence)

    # Calculate the gradient of the subsidence curve
    gradient = np.gradient(subsidence_curve)
    
    # --- Plotting ---
    fig, ax1 = plt.subplots(figsize=(18, 8))
    x_axis = np.arange(len(subsidence_curve))
    
    color = 'tab:blue'
    ax1.set_xlabel('Surface Position Index')
    ax1.set_ylabel('Settlement', color=color)
    ax1.plot(x_axis, subsidence_curve, color=color, label='Surface Settlement', lw=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Gradient / Attention', color=color)
    ax2.plot(x_axis, gradient, color=color, label='Settlement Gradient', linestyle='--', lw=2)
    
    color = 'tab:red'
    ax2.fill_between(x_axis, attention_score_subsidence, color=color, alpha=0.4, label='Attention Score')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    if save_name_suffix is None:
        save_name_suffix = f"sample_{sample_idx}"
    
    fig.suptitle(f'Transformer Attention Analysis ({save_name_suffix})', fontsize=16)
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, f'attention_visualization_{save_name_suffix}.png')
    plt.savefig(save_path)
    
    if show_plot:
        plt.show()
    
    plt.close(fig) # Close the figure to free up memory



# --- 3. Main Execution Block ---

def main():
    """Main function to load model and run visualization."""
    print("==========================================================")
    print("      Transformer Attention Visualization Script      ")
    print("==========================================================")

    if not os.path.exists(MODEL_PATH):
        print(f"FATAL: Model file not found at '{MODEL_PATH}'")
        print("Please train the model first (run train_model.py).")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_files = glob.glob(os.path.join(DATASET_DIR, "*.npz"))
    if not all_files:
        print(f"FATAL: No .npz files found in '{DATASET_DIR}'.")
        return
    
    print(f"Found {len(all_files)} total samples.")

    # --- Data Normalization & Splitting ---
    np.random.shuffle(all_files) # Shuffle for random validation set
    TRAIN_VAL_SPLIT_RATIO = 0.9
    split_idx = int(TRAIN_VAL_SPLIT_RATIO * len(all_files))
    train_files, val_files = all_files[:split_idx], all_files[split_idx:]

    print("Calculating normalization stats from training set...")
    stats_dataset = FractureDataset(train_files)
    all_x_for_stats = [stats_dataset[i][0] for i in range(len(stats_dataset))]
    x_tensor = torch.stack(all_x_for_stats, dim=0)
    mean = x_tensor.mean(dim=0)
    std = x_tensor.std(dim=0)
    transform = NormalizeTransform(mean, std)
    print("Normalization stats calculated.")

    # Create the validation dataset with the transform
    val_dataset = FractureDataset(val_files, transform=transform)
    print(f"Created validation set with {len(val_dataset)} samples.")

    # --- Model Loading ---
    print("Initializing and loading the model...")
    input_features = val_dataset[0][0].shape[0]
    output_features = 64 * 64 # As defined in training

    model = TransformerSurrogateModel(
        input_size=input_features, 
        output_size=output_features
    ).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Model state loaded from '{MODEL_PATH}'.")

    # --- Run Visualization on 100 Random Samples ---
    num_samples_to_visualize = 100
    if len(val_dataset) < num_samples_to_visualize:
        print(f"Warning: Number of validation samples ({len(val_dataset)}) is less than requested. Visualizing all {len(val_dataset)} samples.")
        num_samples_to_visualize = len(val_dataset)

    random_indices = np.random.choice(len(val_dataset), num_samples_to_visualize, replace=False)

    print(f"\nStarting visualization for {num_samples_to_visualize} random samples...")
    for idx in tqdm(random_indices, desc="Generating attention plots"):
        filepath = val_dataset.file_list[idx]
        filename_suffix = os.path.basename(filepath).replace('.npz', '')
        visualize_attention(model, val_dataset, device, sample_idx=idx, save_name_suffix=filename_suffix)

    print(f"\nFinished generating {num_samples_to_visualize} plots in '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()
