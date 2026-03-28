#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_substrate.py - Substrate Occupancy Visualization Script

Convert the final substrate occupancy matrix (after translation)
into black-and-white bitmap images.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys

def load_substrate_matrix(filepath="data/intermediate/final_substrate/final_substrate.npy"):
    """Load substrate matrix"""
    if not os.path.exists(filepath):
        print(f"Error: File does not exist - {filepath}")
        print("Please run computation_core.py first to generate substrate matrix")
        sys.exit(1)
    
    substrate = np.load(filepath)
    print(f"Loaded substrate matrix: {substrate.shape}")
    print(f"Value range: [{substrate.min()}, {substrate.max()}]")
    print(f"Occupied pixels: {np.sum(substrate == 1)}")
    print(f"Blank pixels: {np.sum(substrate == 0)}")
    
    return substrate

def create_substrate_image(substrate_matrix, output_path="data/output/substrate_visualization.png"):
    """
    Create substrate occupancy visualization image
    
    Parameters:
    substrate_matrix: 2D numpy array, 0=blank, 1=occupied
    output_path: output image path
    """
    # Get dimensions
    height, width = substrate_matrix.shape
    
    # Create figure with exact aspect ratio matching substrate dimensions
    # To preserve original proportions without compression
    aspect_ratio = width / height
    
    # Simple approach: set a target area and calculate dimensions to match aspect ratio
    # Target figure area in square inches (reasonable size for visualization)
    target_area = 80  # square inches
    
    # Calculate dimensions that match aspect ratio and target area
    fig_height = np.sqrt(target_area / aspect_ratio)
    fig_width = fig_height * aspect_ratio
    
    # Limit maximum dimension to 24 inches (reasonable for display/saving)
    max_dimension = 24
    if fig_width > max_dimension or fig_height > max_dimension:
        scale_factor = max_dimension / max(fig_width, fig_height)
        fig_width *= scale_factor
        fig_height *= scale_factor
    
    # Ensure minimum dimensions for readability
    min_dimension = 4
    if fig_width < min_dimension or fig_height < min_dimension:
        scale_factor = min_dimension / min(fig_width, fig_height)
        fig_width *= scale_factor
        fig_height *= scale_factor
    
    print(f"  Creating figure with exact aspect ratio: {fig_width:.1f}×{fig_height:.1f} inches")
    print(f"  Aspect ratio: {fig_width/fig_height:.3f} (substrate: {aspect_ratio:.3f})")
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Create grayscale image
    # 0=blank -> white, 1=occupied -> black
    cmap = plt.cm.gray  # grayscale colormap
    # Invert: 0->white(1.0), 1->black(0.0)
    display_matrix = 1.0 - substrate_matrix.astype(float)
    
    # Display image with equal aspect ratio to preserve pixel shape
    im = ax.imshow(display_matrix, cmap=cmap, vmin=0, vmax=1,
                   extent=[0, width, height, 0], aspect='equal')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Occupancy Status', rotation=270, labelpad=15)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Occupied', 'Blank'])
    
    # Set axes labels
    ax.set_xlabel('X Coordinate (pixels)', fontsize=12)
    ax.set_ylabel('Y Coordinate (pixels)', fontsize=12)
    ax.set_title('Substrate Occupancy Visualization', fontsize=16, fontweight='bold')
    
    # Add grid (optional, can make image clearer)
    ax.grid(True, which='both', color='lightgray', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add statistics text
    occupied_pixels = np.sum(substrate_matrix == 1)
    blank_pixels = np.sum(substrate_matrix == 0)
    total_pixels = occupied_pixels + blank_pixels
    occupancy_rate = occupied_pixels / total_pixels * 100 if total_pixels > 0 else 0
    
    stats_text = f"Size: {width}×{height} pixels\n"
    stats_text += f"Occupied: {occupied_pixels} pixels ({occupancy_rate:.1f}%)\n"
    stats_text += f"Blank: {blank_pixels} pixels"
    
    # Place text in upper right corner
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout without compression
    # Use constrained_layout instead of tight_layout to preserve aspect ratio
    plt.tight_layout(pad=2.0, h_pad=2.0, w_pad=2.0)
    
    # Save image without compression and without tight bounding box
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches=None, pad_inches=0.5)
    print(f"Image saved to: {output_path}")
    
    # Display image (commented out for headless environments)
    # plt.show()
    print("  Note: Image saved. Uncomment plt.show() to display.")
    
    return fig, ax

def create_high_contrast_image(substrate_matrix, output_path="data/output/substrate_binary.png"):
    """
    Create high-contrast black-and-white image (true binary bitmap)
    
    Parameters:
    substrate_matrix: 2D numpy array, 0=blank, 1=occupied
    output_path: output image path
    """
    # Get dimensions
    height, width = substrate_matrix.shape
    
    # Create figure with exact aspect ratio matching substrate dimensions
    # To preserve original proportions without compression
    aspect_ratio = width / height
    
    # Simple approach: set a target area and calculate dimensions to match aspect ratio
    # Target figure area in square inches (reasonable size for visualization)
    target_area = 80  # square inches
    
    # Calculate dimensions that match aspect ratio and target area
    # height * width = target_area, and width/height = aspect_ratio
    # So: height * (height * aspect_ratio) = target_area
    # => height² * aspect_ratio = target_area
    # => height = sqrt(target_area / aspect_ratio)
    fig_height = np.sqrt(target_area / aspect_ratio)
    fig_width = fig_height * aspect_ratio
    
    # Limit maximum dimension to 24 inches (reasonable for display/saving)
    max_dimension = 24
    if fig_width > max_dimension or fig_height > max_dimension:
        # Scale down proportionally
        scale_factor = max_dimension / max(fig_width, fig_height)
        fig_width *= scale_factor
        fig_height *= scale_factor
    
    # Ensure minimum dimensions for readability
    min_dimension = 4
    if fig_width < min_dimension or fig_height < min_dimension:
        # Scale up proportionally
        scale_factor = min_dimension / min(fig_width, fig_height)
        fig_width *= scale_factor
        fig_height *= scale_factor
    
    print(f"  Creating figure with exact aspect ratio: {fig_width:.1f}×{fig_height:.1f} inches")
    print(f"  Aspect ratio: {fig_width/fig_height:.3f} (substrate: {aspect_ratio:.3f})")
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Create pure black-and-white image
    # 0=blank -> white(255), 1=occupied -> black(0)
    # Use binary colormap
    cmap = plt.cm.binary  # black-and-white binary colormap
    
    # Display image with equal aspect ratio
    im = ax.imshow(substrate_matrix, cmap=cmap, vmin=0, vmax=1,
                   extent=[0, width, height, 0], aspect='equal')
    
    # Set axes labels
    ax.set_xlabel('X Coordinate (pixels)', fontsize=12)
    ax.set_ylabel('Y Coordinate (pixels)', fontsize=12)
    ax.set_title('Substrate Occupancy - Black & White Bitmap', fontsize=16, fontweight='bold')
    
    # Add grid
    ax.grid(True, which='both', color='lightgray', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', edgecolor='black', label='Blank Area'),
        Patch(facecolor='black', edgecolor='black', label='Occupied Area')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Adjust layout without compression
    plt.tight_layout(pad=2.0, h_pad=2.0, w_pad=2.0)
    
    # Save as high-DPI image without compression
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches=None, pad_inches=0.5)
    print(f"Black & white bitmap saved to: {output_path}")
    
    # Display image (commented out for headless environments)
    # plt.show()
    print("  Note: Image saved. Uncomment plt.show() to display.")
    
    return fig, ax

def create_physical_coordinate_image(substrate_matrix, pixel_resolution=0.5,
                                     output_path="data/output/substrate_physical.png"):
    """
    Create visualization with physical coordinates
    
    Parameters:
    substrate_matrix: 2D numpy array
    pixel_resolution: pixel resolution (mm/pixel)
    output_path: output image path
    """
    height_px, width_px = substrate_matrix.shape
    width_mm = width_px * pixel_resolution
    height_mm = height_px * pixel_resolution
    
    # Create figure with exact aspect ratio matching physical dimensions
    # To preserve original proportions without compression
    aspect_ratio = width_mm / height_mm
    
    # Simple approach: set a target area and calculate dimensions to match aspect ratio
    # Target figure area in square inches (reasonable size for visualization)
    target_area = 80  # square inches
    
    # Calculate dimensions that match aspect ratio and target area
    fig_height = np.sqrt(target_area / aspect_ratio)
    fig_width = fig_height * aspect_ratio
    
    # Limit maximum dimension to 24 inches (reasonable for display/saving)
    max_dimension = 24
    if fig_width > max_dimension or fig_height > max_dimension:
        scale_factor = max_dimension / max(fig_width, fig_height)
        fig_width *= scale_factor
        fig_height *= scale_factor
    
    # Ensure minimum dimensions for readability
    min_dimension = 4
    if fig_width < min_dimension or fig_height < min_dimension:
        scale_factor = min_dimension / min(fig_width, fig_height)
        fig_width *= scale_factor
        fig_height *= scale_factor
    
    print(f"  Creating figure with exact aspect ratio: {fig_width:.1f}×{fig_height:.1f} inches")
    print(f"  Aspect ratio: {fig_width/fig_height:.3f} (physical: {aspect_ratio:.3f})")
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Use physical coordinates with equal aspect ratio
    cmap = plt.cm.binary
    im = ax.imshow(substrate_matrix, cmap=cmap, vmin=0, vmax=1,
                   extent=[0, width_mm, height_mm, 0], aspect='equal')
    
    # Set axes labels
    ax.set_xlabel('X Coordinate (mm)', fontsize=12)
    ax.set_ylabel('Y Coordinate (mm)', fontsize=12)
    ax.set_title(f'Substrate Occupancy - Physical Dimensions ({width_mm:.1f}mm × {height_mm:.1f}mm)',
                 fontsize=16, fontweight='bold')
    
    # Add grid (mm grid)
    ax.grid(True, which='both', color='lightgray', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Adjust layout without compression
    plt.tight_layout(pad=2.0, h_pad=2.0, w_pad=2.0)
    
    # Save image without compression
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches=None, pad_inches=0.5)
    print(f"Physical coordinate image saved to: {output_path}")
    
    # Display image (commented out for headless environments)
    # plt.show()
    print("  Note: Image saved. Uncomment plt.show() to display.")
    
    return fig, ax

def main():
    """Main function"""
    print("=== Substrate Occupancy Visualization ===\n")
    
    # 1. Load substrate matrix
    substrate = load_substrate_matrix()
    
    # 2. Create standard visualization image
    print("\n1. Creating standard visualization image...")
    create_substrate_image(substrate, "data/output/substrate_visualization.png")
    
    # 3. Create black & white bitmap
    print("\n2. Creating black & white bitmap...")
    create_high_contrast_image(substrate, "data/output/substrate_binary.png")
    
    # 4. Create physical coordinate image
    print("\n3. Creating physical coordinate image...")
    create_physical_coordinate_image(substrate, pixel_resolution=0.5,
                                     output_path="data/output/substrate_physical.png")
    
    print("\n=== Visualization Complete ===")
    print("Generated images saved in data/output/ directory:")
    print("  - substrate_visualization.png: Standard visualization with colorbar")
    print("  - substrate_binary.png: Black & white bitmap")
    print("  - substrate_physical.png: Physical coordinate scale image")

if __name__ == "__main__":
    main()