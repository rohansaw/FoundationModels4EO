"""
Visualization helpers for the foundation models notebook.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import leafmap
from ipyleaflet import Rectangle
from geo_helpers import compute_clusters
from geo_helpers import create_embedding_rgb
from matplotlib.patches import Patch



def plot_rgb_image(rgb_array, title="RGB Image"):
    """Display an RGB image."""
    plt.figure(figsize=(10, 8))
    plt.imshow(rgb_array)
    plt.axis("off")
    plt.title(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_rgb_timeseries(rgb_dict, region_name="Region", year=2024):
    """Display RGB images for multiple months in a time series."""
    month_names = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }
    
    n_months = len(rgb_dict)
    ncols = min(3, n_months)
    nrows = (n_months + ncols - 1) // ncols
    
    # Make figures larger to use full row width
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6*nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (month, rgb) in enumerate(sorted(rgb_dict.items())):
        axes[idx].imshow(rgb)
        axes[idx].set_title(f'{month_names[month]} {year}', fontsize=13, fontweight='bold')
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(len(rgb_dict), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'{region_name} - Sentinel-2 RGB Time Series', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()


def plot_crop_labels(cdl_data, simplified_labels=None, year=2021):
    """Visualize crop labels - both detailed and simplified views."""
    if simplified_labels is None:
        # Just show the raw data
        plt.figure(figsize=(10, 8))
        plt.imshow(cdl_data, cmap='tab20', interpolation='nearest')
        plt.title(f'USDA Crop Labels ({year})', fontsize=12, fontweight='bold')
        plt.axis('off')
        plt.colorbar(label='CDL Code')
        plt.tight_layout()
        plt.show()
        return
    
    # Show both views
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: All crop classes
    im1 = ax1.imshow(cdl_data, cmap='tab20', interpolation='nearest')
    ax1.set_title(f'All Crop Classes ({year})', fontsize=12, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='CDL Code')
    
    # Right: Simplified view - Corn, Soy, Other
    colors = ['black', 'yellow', 'green', 'lightgray']
    cmap_simple = ListedColormap(colors)
    
    im2 = ax2.imshow(simplified_labels, cmap=cmap_simple, vmin=0, vmax=3, interpolation='nearest')
    ax2.set_title(f'Corn & Soy Focus ({year})', fontsize=12, fontweight='bold')
    ax2.axis('off')
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, ticks=[0, 1, 2, 3])
    cbar2.ax.set_yticklabels(['No Data', 'Corn', 'Soybeans', 'Other'])
    
    plt.tight_layout()
    plt.show()


def plot_embeddings_rgb(embeddings_rgb, bands=[0, 15, 8]):
    """Display embedding visualization as RGB composite."""
    plt.figure(figsize=(10, 8))
    plt.imshow(embeddings_rgb, interpolation='nearest')
    plt.axis("off")
    plt.title(f"Foundation Model Embeddings\n(Dimensions {bands[0]+1}, {bands[1]+1}, {bands[2]+1} → R, G, B)", 
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_clustering_results(embeddings=None, k_values=[3, 5, 10], cluster_results=None):
    """Visualize unsupervised clustering of embeddings."""
    # If precomputed results provided, use them
    if cluster_results is not None:
        k_values = sorted(cluster_results.keys())
    else:
        # Compute clusters on the fly
        cluster_results = compute_clusters(embeddings, k_values)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, k in enumerate(k_values):
        if idx >= 4:
            break
        labels_image = cluster_results[k]
        
        axes[idx].imshow(labels_image, cmap='tab20', interpolation='nearest')
        axes[idx].axis("off")
        axes[idx].set_title(f"KMeans Clustering (k={k})", fontsize=12, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(len(k_values), 4):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_classification_results(y_test, y_pred, accuracy, class_names=['Other', 'Corn', 'Soy']):
    """Display classification metrics and confusion matrix."""
    print(f"\n{'='*60}")
    print(f"CLASSIFICATION RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {accuracy:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2%})', fontsize=12, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def plot_prediction_map(labels_true, labels_pred, accuracy, embeddings=None, valid_mask=None):
    """Visualize ground truth vs predictions side-by-side."""
    H, W = labels_true.shape
    
    # Reshape predictions if needed
    if labels_pred.ndim == 1 and valid_mask is not None:
        pred_map = np.full(H * W, -1, dtype=np.int32)
        pred_map[valid_mask.flatten()] = labels_pred
        labels_pred = pred_map.reshape(H, W)
    
    colors = ['black', 'yellow', 'green', 'lightgray']
    cmap_classes = ListedColormap(colors)
    
    if embeddings is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
        
        # Show embedding RGB
        emb_rgb = create_embedding_rgb(embeddings)
        ax1.imshow(emb_rgb, interpolation='nearest')
        ax1.set_title('Foundation Model Embeddings', fontsize=11, fontweight='bold')
        ax1.axis('off')
    else:
        fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Ground truth
    im2 = ax2.imshow(labels_true, cmap=cmap_classes, vmin=-1, vmax=2, interpolation='nearest')
    ax2.set_title('Ground Truth (USDA CDL)', fontsize=11, fontweight='bold')
    ax2.axis('off')
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, ticks=[-1, 0, 1, 2])
    cbar2.ax.set_yticklabels(['No Data', 'Other', 'Corn', 'Soy'])
    
    # Predictions
    im3 = ax3.imshow(labels_pred, cmap=cmap_classes, vmin=-1, vmax=2, interpolation='nearest')
    ax3.set_title(f'Predictions (Acc: {accuracy:.2%})', fontsize=11, fontweight='bold')
    ax3.axis('off')
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, ticks=[-1, 0, 1, 2])
    cbar3.ax.set_yticklabels(['No Data', 'Other', 'Corn', 'Soy'])
    
    plt.tight_layout()
    plt.show()


def plot_generalization_comparison(region_name, rgb, labels_true, pred_iowa_only, pred_combined,
                                   acc_iowa, acc_combined, embeddings=None):
    """Compare model performance: Iowa-only vs Iowa+Region."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    colors = ['black', 'yellow', 'green', 'lightgray']
    cmap_classes = ListedColormap(colors)
    
    # Row 1: RGB, Ground Truth, Iowa-only predictions
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f'{region_name} - Sentinel-2 RGB', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(labels_true, cmap=cmap_classes, vmin=-1, vmax=2, interpolation='nearest')
    axes[0, 1].set_title(f'{region_name} - Ground Truth', fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(pred_iowa_only, cmap=cmap_classes, vmin=-1, vmax=2, interpolation='nearest')
    axes[0, 2].set_title(f'Iowa Model (Acc: {acc_iowa:.2%})', fontsize=11, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 2: Embeddings, Combined predictions, Stats
    if embeddings is not None:
        emb_rgb = create_embedding_rgb(embeddings)
        axes[1, 0].imshow(emb_rgb, interpolation='nearest')
        axes[1, 0].set_title(f'{region_name} - Embeddings', fontsize=11, fontweight='bold')
    else:
        axes[1, 0].text(0.5, 0.5, 'Embeddings', ha='center', va='center', fontsize=14)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(pred_combined, cmap=cmap_classes, vmin=-1, vmax=2, interpolation='nearest')
    axes[1, 1].set_title(f'Iowa+{region_name} Model (Acc: {acc_combined:.2%})', fontsize=11, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Performance comparison text
    improvement = acc_combined - acc_iowa
    axes[1, 2].text(0.5, 0.7, f'Model Performance on {region_name}:', 
                    ha='center', fontsize=12, fontweight='bold', transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.5, 0.5, f'Iowa-only: {acc_iowa:.2%}', 
                    ha='center', fontsize=11, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.5, 0.4, f'Iowa+{region_name}: {acc_combined:.2%}', 
                    ha='center', fontsize=11, transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.5, 0.3, f'Improvement: +{100*improvement/acc_iowa:.1f}%', 
                    ha='center', fontsize=11, color='green', fontweight='bold', 
                    transform=axes[1, 2].transAxes)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()


def show_study_area_map(lon_min, lon_max, lat_min, lat_max):
    """Display an interactive map with the study area highlighted."""
    m = leafmap.Map(center=[(lat_min + lat_max) / 2, (lon_min + lon_max) / 2], zoom=10)
    
    # Add rectangle showing study area
    rectangle = Rectangle(
        bounds=((lat_min, lon_min), (lat_max, lon_max)),
        color='red',
        fill_color='red',
        fill_opacity=0,
        weight=2
    )
    m.add_layer(rectangle)
    
    return m


def plot_classification_vs_clustering(embeddings, y_true, y_pred, cluster_results, class_names):
    """Show classification results next to clustering for comparison."""
    
    n_clusters = len(cluster_results)
    n_cols = n_clusters + 1
    
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]
    
    # Classification predictions with legend
    # Colors: other=brown, corn=green, soy=blue
    colors = ['saddlebrown', 'green', 'blue']
    cmap_pred = ListedColormap(colors[:len(class_names)])
    axes[0].imshow(y_pred, cmap=cmap_pred, vmin=0, vmax=len(class_names)-1, interpolation='nearest')
    
    # Calculate accuracy if ground truth available
    if y_true is not None and y_true.size > 0:
        # Flatten and filter valid pixels
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        valid_mask = (y_true_flat >= 0) & (y_true_flat < len(class_names))
        if valid_mask.sum() > 0:
            acc = accuracy_score(y_true_flat[valid_mask], y_pred_flat[valid_mask])
            axes[0].set_title(f'Classification Predictions\n(Accuracy: {acc:.2%})', 
                            fontsize=13, fontweight='bold')
        else:
            axes[0].set_title('Classification Predictions', fontsize=13, fontweight='bold')
    else:
        axes[0].set_title('Supervised Classification', fontsize=13, fontweight='bold')
    axes[0].axis('off')
    
    # Add legend for classification
    legend_elements = [Patch(facecolor=colors[i], label=class_names[i].capitalize()) 
                      for i in range(len(class_names))]
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=10, 
                  framealpha=0.9, edgecolor='black')
    
    
    # Remaining plots: Clustering results
    for idx, k in enumerate(sorted(cluster_results.keys())):
        cluster_labels = cluster_results[k]
        axes[idx + 1].imshow(cluster_labels, cmap='tab10', interpolation='nearest')
        axes[idx + 1].set_title(f'Unsupervised Clustering\n(k={k} clusters)', 
                               fontsize=13, fontweight='bold')
        axes[idx + 1].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_sample_map_by_class(df, center_lat=33.5, center_lon=-90.8, zoom=7):
    """Create interactive map showing sample distribution by crop class."""
    import pandas as pd
    
    m_all = leafmap.Map(center=[center_lat, center_lon], zoom=zoom)
    
    # Define colors for different classes
    class_colors = {
        0: '#FFD700',    # corn - gold
        1: '#32CD32',    # soy - lime green
        2: '#808080'     # other - gray
    }
    
    # Create GeoJSON features for each class
    for label, color in class_colors.items():
        class_data = df[df['label'] == label].copy()
        if len(class_data) == 0:
            continue
            
        class_name = class_data['class_name'].iloc[0]
        
        # Create list of coordinates for this class
        features = []
        for idx, row in class_data.iterrows():
            if pd.notna(row['latitude']) and pd.notna(row['longitude']):
                features.append({
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Point',
                        'coordinates': [row['longitude'], row['latitude']]
                    },
                    'properties': {
                        'id': int(idx),
                        'class': class_name,
                        'split': row['split']
                    }
                })
        
        geojson = {
            'type': 'FeatureCollection',
            'features': features
        }
        
        # Add to map with proper point styling
        m_all.add_geojson(
            geojson,
            layer_name=f'{class_name} ({len(features)} samples)',
            style={
                'color': color,
                'fillColor': color,
                'fillOpacity': 0.7,
                'weight': 1
            },
            point_style={
                'radius': 3,
                'fillColor': color,
                'color': color,
                'weight': 1,
                'fillOpacity': 0.7
            }
        )
    
    print("Map 1: Distribution of ALL samples by crop type")
    print(f"  - Total samples: {len(df)}")
    print(f"  - Corn (gold): {len(df[df['label']==0])} samples")
    print(f"  - Soy (green): {len(df[df['label']==1])} samples")
    print(f"  - Other (gray): {len(df[df['label']==2])} samples")
    
    return m_all


def plot_sample_map_by_split(df, center_lat=33.5, center_lon=-90.8, zoom=7):
    """Create interactive map showing train vs test sample distribution."""
    import pandas as pd
    
    m_split = leafmap.Map(center=[center_lat, center_lon], zoom=zoom)
    
    # Define colors for train/test split
    split_colors = {
        'train': '#FF4444',  # red
        'test': '#4169E1'    # blue
    }
    
    split_sizes = {
        'train': 4,
        'test': 3
    }
    
    # Create GeoJSON features for each split
    for split_name, color in split_colors.items():
        split_data = df[df['split'] == split_name].copy()
        
        # Create list of coordinates for this split
        features = []
        for idx, row in split_data.iterrows():
            if pd.notna(row['latitude']) and pd.notna(row['longitude']):
                features.append({
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Point',
                        'coordinates': [row['longitude'], row['latitude']]
                    },
                    'properties': {
                        'id': int(idx),
                        'class': row['class_name'],
                        'split': split_name
                    }
                })
        
        geojson = {
            'type': 'FeatureCollection',
            'features': features
        }
        
        # Add to map with proper point styling
        m_split.add_geojson(
            geojson,
            layer_name=f'{split_name.upper()} ({len(features)} samples)',
            style={
                'color': color,
                'fillColor': color,
                'fillOpacity': 0.8,
                'weight': 1
            },
            point_style={
                'radius': split_sizes[split_name],
                'fillColor': color,
                'color': color,
                'weight': 1,
                'fillOpacity': 0.8
            }
        )
    
    print("Map 2: Spatial distribution of TRAIN vs TEST samples")
    print(f"  - Training samples (red, larger): {len(df[df['split']=='train'])}")
    print(f"  - Test samples (blue, smaller): {len(df[df['split']=='test'])}")
    print("\nThis helps identify:")
    print("  • Spatial clustering of samples")
    print("  • Geographic separation between train/test sets")
    print("  • Whether train and test samples are in close proximity")
    
    return m_split