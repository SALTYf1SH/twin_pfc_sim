# -*- coding: utf-8 -*-
"""
Data Visualization Script for PFC Simulation Results

This script generates and displays several plots from the raw CSV data produced by a PFC simulation run.

Usage:
    python visualize_simulation.py --exp_dir <path_to_experiment_directory> --step <step_number>

Example:
    python visualize_simulation.py --exp_dir experiments/Geology_Sim_Panel_12401_sample_0046_sample_0009 --step 1
"""

import os
import argparse
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_fracture_distribution(exp_dir, step):
    """Plots the fracture distribution as a scatter plot."""
    fracture_file = os.path.join(exp_dir, 'csv', f'fractures_step_{step}.csv')
    if not os.path.exists(fracture_file):
        print(f"Warning: Fracture file not found for step {step}: {fracture_file}")
        return

    print(f"Plotting fracture distribution for step {step}...")
    df = pd.read_csv(fracture_file)

    plt.figure(figsize=(10, 8))
    plt.scatter(df['pos_x'], df['pos_y'], s=1, alpha=0.6, marker='.')
    plt.title(f'Fracture Distribution - Step {step}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def plot_block_centroids(exp_dir, step):
    """Plots the fragment centroids as a scatter plot."""
    fragments_file = os.path.join(exp_dir, 'csv', f'fragments_properties_step_{step}.csv')
    if not os.path.exists(fragments_file):
        print(f"Warning: Fragments file not found for step {step}: {fragments_file}")
        return

    print(f"Plotting block centroids for step {step}...")
    df = pd.read_csv(fragments_file)

    plt.figure(figsize=(10, 8))
    plt.scatter(df['centroid_x'], df['centroid_y'], s=10, alpha=0.8)
    plt.title(f'Block Centroids - Step {step}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def plot_displacement_map(exp_dir, step):
    """Plots the displacement data as a contour map."""
    displacement_files = glob.glob(os.path.join(exp_dir, 'csv', 'resampled_displacement_*.csv'))
    if not displacement_files:
        print(f"Warning: No displacement files found in {exp_dir}")
        return
    
    displacement_file = displacement_files[0]
    print(f"Plotting displacement map using {os.path.basename(displacement_file)}...")
    
    df = pd.read_csv(displacement_file)
    
    y_coords = df.iloc[:, 0].values
    x_coords = [float(c) for c in df.columns[1:]]
    displacement_values = df.iloc[:, 1:].values

    X, Y = np.meshgrid(x_coords, y_coords)

    plt.figure(figsize=(12, 8))
    contour = plt.contourf(X, Y, displacement_values, levels=50, cmap='viridis')
    plt.colorbar(contour, label='Displacement')
    plt.title(f'Displacement Map - Step {step} (from {os.path.basename(displacement_file)})')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.axis('equal')
    plt.show()

def plot_ground_fracture_curve(exp_dir, step):
    """Plots a curve representing the density of fractures along the X-axis."""
    fracture_file = os.path.join(exp_dir, 'csv', f'fractures_step_{step}.csv')
    if not os.path.exists(fracture_file):
        print(f"Warning: Fracture file not found for step {step}: {fracture_file}")
        return

    print(f"Plotting ground fracture curve for step {step}...")
    df = pd.read_csv(fracture_file)

    counts, bin_edges = np.histogram(df['pos_x'], bins=100)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.figure(figsize=(12, 6))
    plt.plot(bin_centers, counts, linestyle='-', marker='o', markersize=4)
    plt.title(f'Ground Fracture Density Curve - Step {step}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Fracture Count per Bin')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(bottom=0)
    plt.show()


def main():
    """Main function to parse arguments and run plotting functions."""
    parser = argparse.ArgumentParser(description="Visualize PFC simulation data.")
    parser.add_argument('--exp_dir', type=str, required=True, help='Path to the experiment directory.')
    parser.add_argument('--step', type=int, required=True, help='Simulation step number to visualize.')
    args = parser.parse_args()

    if not os.path.isdir(args.exp_dir):
        print(f"Error: Experiment directory not found at {args.exp_dir}")
        return

    print(f"Visualizing data for {args.exp_dir}, step {args.step}")

    # Run all plotting functions
    plot_fracture_distribution(args.exp_dir, args.step)
    plot_block_centroids(args.exp_dir, args.step)
    plot_displacement_map(args.exp_dir, args.step)
    plot_ground_fracture_curve(args.exp_dir, args.step)
    
    print("\nAll plotting tasks complete.")

if __name__ == '__main__':
    main()
