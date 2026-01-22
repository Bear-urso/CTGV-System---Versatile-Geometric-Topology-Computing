"""
CTGV System - Utility Functions
"""
import matplotlib.pyplot as plt
import numpy as np

def visualize_ctgv_processing(original, processed):
    """
    Generates a heatmap to compare input with topological output.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Original Pattern Plot
    im1 = axes[0].imshow(original, cmap='viridis', interpolation='nearest')
    axes[0].set_title("Original Pattern (Input)")
    plt.colorbar(im1, ax=axes[0])

    # Network Processed Pattern Plot
    im2 = axes[1].imshow(processed, cmap='plasma', interpolation='nearest')
    axes[1].set_title("Final Topological Field (Output)")
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.show()