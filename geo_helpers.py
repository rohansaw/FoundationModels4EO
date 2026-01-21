"""
Helper functions for geospatial data and foundation model embeddings.

These utils should probably be refactored, right now I there is quite a lot of duplication
and this code serves simply for supporting the demonstration notebooks.
"""

import numpy as np
import pandas as pd
import rioxarray
import planetary_computer as pc
from pystac_client import Client
import rasterio
from rasterio.session import AWSSession
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.windows import from_bounds
from dask.diagnostics import ProgressBar
from scipy.ndimage import zoom
import os
from sklearn.cluster import MiniBatchKMeans
import calendar
import json





# Configure AWS for public S3 access
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'
os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'
os.environ['CPL_VSIL_CURL_ALLOWED_EXTENSIONS'] = 'tif,tiff'
os.environ['VSI_CACHE'] = 'TRUE'
os.environ['VSI_CACHE_SIZE'] = '50000000'

# Constants
STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1"
AEF_INDEX_URL = "https://data.source.coop/tge-labs/aef/v1/annual/aef_index.parquet"
CORN_CODE = 1
SOY_CODE = 5


def load_sentinel2_rgb(lon_min, lon_max, lat_min, lat_max, year, month=None):
    """Load Sentinel-2 RGB image for a region.
    
    Args:
        lon_min, lon_max, lat_min, lat_max: Bounding box coordinates in WGS84
        year: Year to load imagery from
        month: Optional month to load imagery from
        target_shape: Optional (height, width) tuple to ensure consistent output size.
                     If None, uses native resolution of clipped data.
    
    Returns:
        RGB numpy array of shape (height, width, 3)
    """
    if month is not None:
        print(f"Loading Sentinel-2 RGB imagery for {year}-{month:02d}...")
        start_date = f"{year}-{month:02d}-01"
        
        # Get the last day of the month using calendar module
        last_day = calendar.monthrange(year, month)[1]
        end_date = f"{year}-{month:02d}-{last_day:02d}"
    
    catalog = Client.open(STAC_API)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=[lon_min, lat_min, lon_max, lat_max],
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": 20}},
        max_items=5,
    )
    items = list(search.get_items())
    
    if not items:
        raise RuntimeError(f"No imagery found for the specified date range")
    
    items_sorted = sorted(items, key=lambda x: x.properties.get("eo:cloud_cover", 100))
    item = pc.sign(items_sorted[0])
    
    rgb_url = item.assets["visual"].href
    rgb_da = rioxarray.open_rasterio(rgb_url)
    rgb_da = rgb_da.rio.write_crs(rgb_da.rio.crs)
    
    # Clip to the exact bounding box
    rgb_clipped = rgb_da.rio.clip_box(
        minx=lon_min, miny=lat_min, maxx=lon_max, maxy=lat_max, crs="EPSG:4326"
    )
    
    # CHW to HWC and ensure north-up orientation
    rgb = np.transpose(rgb_clipped.values, (1, 2, 0))
    
    date_str = item.properties.get('datetime', '')[:10]
    print(f" Loaded Sentinel-2 RGB: {rgb.shape} from {date_str}")
    return rgb


def load_sentinel2_rgb_timeseries(lon_min, lon_max, lat_min, lat_max, year, months, target_shape=None):
    """Load Sentinel-2 RGB images for multiple months with consistent spatial grid.
    
    Args:
        lon_min, lon_max, lat_min, lat_max: Bounding box coordinates in WGS84
        year: Year to load imagery from
        months: List of months to load imagery from
        target_shape: Optional (height, width) tuple to ensure all images have consistent size.
                     If None, will calculate based on bounding box and 10m resolution.
    
    Returns:
        Dictionary mapping month -> RGB array (all with same shape and spatial alignment)
    """
    print(f"\nLoading Sentinel-2 time series for {len(months)} months...")
    rgb_timeseries_raw = {}

    # First pass: load all images as xarray DataArrays
    for month in months:
        try:
            rgb_da = _load_sentinel2_rgb_dataarray(lon_min, lon_max, lat_min, lat_max, year, month)
            if rgb_da is not None:
                rgb_timeseries_raw[month] = rgb_da
        except Exception as e:
            print(f"  Warning: Could not load imagery for month {month}: {e}")
    
    if not rgb_timeseries_raw:
        return {}
    
    # Determine target shape based on bounding box and 10m resolution
    if target_shape is None:
        # Use first image's transform to calculate expected shape
        first_da = list(rgb_timeseries_raw.values())[0]
        target_shape = (first_da.shape[1], first_da.shape[2])  # (height, width)
        print(f"\n Using reference shape: {target_shape}")
    
    # Use first image as reference grid
    reference_da = list(rgb_timeseries_raw.values())[0]
    
    # Reproject all images to match the reference grid
    rgb_timeseries = {}
    print(f" Aligning all images to consistent grid...")
    
    for month, rgb_da in rgb_timeseries_raw.items():
        if rgb_da.shape == reference_da.shape and np.allclose(rgb_da.rio.transform(), reference_da.rio.transform()):
            # Already aligned
            rgb = np.transpose(rgb_da.values, (1, 2, 0))
        else:
            # Reproject to match reference
            rgb_da_aligned = rgb_da.rio.reproject_match(reference_da)
            rgb = np.transpose(rgb_da_aligned.values, (1, 2, 0))
            print(f"   Month {month}: {rgb_da.shape[1:]} -> {rgb.shape[:2]}")
        
        rgb_timeseries[month] = rgb
    
    print(f" All images aligned to shape {target_shape}")
    return rgb_timeseries


def _load_sentinel2_rgb_dataarray(lon_min, lon_max, lat_min, lat_max, year, month):
    """Load Sentinel-2 RGB image as xarray DataArray (internal helper)."""
    print(f"Loading Sentinel-2 RGB imagery for {year}-{month:02d}...")
    start_date = f"{year}-{month:02d}-01"
    
    # Get the last day of the month using calendar module
    last_day = calendar.monthrange(year, month)[1]
    end_date = f"{year}-{month:02d}-{last_day:02d}"
    
    catalog = Client.open(STAC_API)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=[lon_min, lat_min, lon_max, lat_max],
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": 20}},
        max_items=5,
    )
    items = list(search.get_items())
    
    if not items:
        raise RuntimeError(f"No imagery found for the specified date range")
    
    items_sorted = sorted(items, key=lambda x: x.properties.get("eo:cloud_cover", 100))
    item = pc.sign(items_sorted[0])
    
    rgb_url = item.assets["visual"].href
    rgb_da = rioxarray.open_rasterio(rgb_url)
    rgb_da = rgb_da.rio.write_crs(rgb_da.rio.crs)
    
    # Clip to the exact bounding box
    rgb_clipped = rgb_da.rio.clip_box(
        minx=lon_min, miny=lat_min, maxx=lon_max, maxy=lat_max, crs="EPSG:4326"
    )
    
    date_str = item.properties.get('datetime', '')[:10]
    print(f" Loaded Sentinel-2 RGB: {rgb_clipped.shape} from {date_str}")
    return rgb_clipped


def load_crop_labels(lon_min, lon_max, lat_min, lat_max, year):
    """Load USDA Cropland Data Layer labels."""
    print(f"Loading crop labels (USDA CDL) for {year}...")
    
    catalog = Client.open(STAC_API)
    search = catalog.search(
        collections=["usda-cdl"],
        bbox=[lon_min, lat_min, lon_max, lat_max],
        datetime=f"{year}-01-01/{year}-12-31",
        max_items=10,
    )
    items = list(search.get_items())
    
    item_with_cropland = None
    for item in items:
        if "cropland" in item.assets:
            item_with_cropland = item
            break
    
    if item_with_cropland is None:
        raise ValueError(f"No cropland data found for the specified region and year {year}")
    
    signed_item = pc.sign(item_with_cropland)
    cdl_url = signed_item.assets["cropland"].href
    cdl_da = rioxarray.open_rasterio(cdl_url)
    cdl_da = cdl_da.rio.write_crs(cdl_da.rio.crs)
    cdl_clipped = cdl_da.rio.clip_box(
        minx=lon_min, miny=lat_min, maxx=lon_max, maxy=lat_max, crs="EPSG:4326"
    )
    
    cdl_data = cdl_clipped.values[0]
    print(f" Loaded crop labels: {cdl_data.shape}")
    return cdl_data

def load_foundation_model_embeddings(lon_min, lon_max, lat_min, lat_max, year):
    """Load AEF embeddings with fallback to alternative tiles if primary fails."""
    print(f"Loading foundation model embeddings for {year}...")
    
    filters = [
        ('year', '=', year),
        ('wgs84_west', '<=', lon_max),
        ('wgs84_east', '>=', lon_min),
        ('wgs84_south', '<=', lat_max),
        ('wgs84_north', '>=', lat_min)
    ]
    
    # Load all necessary columns to calculate overlap
    df = pd.read_parquet(AEF_INDEX_URL, engine='pyarrow', filters=filters, 
                         columns=['path', 'wgs84_west', 'wgs84_east', 'wgs84_south', 'wgs84_north'])
    
    if df.empty:
        raise RuntimeError(f"No embeddings found for year {year} at the specified coordinates")
    
    # Calculate overlap area for all tiles and sort by overlap
    def calculate_overlap_area(row):
        # Calculate intersection bounds
        overlap_west = max(lon_min, row['wgs84_west'])
        overlap_east = min(lon_max, row['wgs84_east'])
        overlap_south = max(lat_min, row['wgs84_south'])
        overlap_north = min(lat_max, row['wgs84_north'])
        
        # Calculate overlap area
        overlap_width = max(0, overlap_east - overlap_west)
        overlap_height = max(0, overlap_north - overlap_south)
        return overlap_width * overlap_height
    
    df['overlap_area'] = df.apply(calculate_overlap_area, axis=1)
    
    # Sort by overlap area (descending) to try tiles in order of best overlap
    df_sorted = df.sort_values('overlap_area', ascending=False).reset_index(drop=True)
    
    if len(df_sorted) > 1:
        print(f" Found {len(df_sorted)} tiles, will try in order of overlap area...")
    
    # Try loading tiles in order until one succeeds
    last_error = None
    for idx, row in df_sorted.iterrows():
        tile_url = str(row['path'])
        overlap = row['overlap_area']
        
        try:
            print(f" Attempting tile {idx + 1}/{len(df_sorted)} (overlap: {overlap:.6f} sq degrees)...")
            
            # Use rioxarray with optimized settings
            with rasterio.Env(
                session=AWSSession(requester_pays=False),
                GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
                GDAL_NUM_THREADS='ALL_CPUS'
            ):
                aef_da = rioxarray.open_rasterio(
                    tile_url, 
                    masked=True, 
                    lock=False,
                )
                
                # Clip to bounds (WGS84/EPSG:4326)
                aef_clipped = aef_da.rio.clip_box(
                    minx=lon_min, miny=lat_min, maxx=lon_max, maxy=lat_max,
                    crs="EPSG:4326"
                )
                
                # Compute with progress
                with ProgressBar():
                    aef_clipped = aef_clipped.compute(scheduler='threads', num_workers=4)
            
            aef_raw = aef_clipped.values
            aef = _dequantize_aef(aef_raw)

            # Flip vertically to match Sentinel-2 orientation
            # aef shape is (64, height, width), so flip axis 1 (height)
            aef = np.flip(aef, axis=1)
            
            print(f" Successfully loaded embeddings: {aef.shape} (64 dimensions)")
            return aef
            
        except Exception as e:
            last_error = e
            print(f" Failed to load tile {idx + 1}: {type(e).__name__}: {str(e)}")
            if idx < len(df_sorted) - 1:
                print(f" Trying next tile...")
            continue
    
    # If all tiles failed, raise the last error
    raise RuntimeError(f"Failed to load embeddings from all {len(df_sorted)} available tiles. Last error: {last_error}")


def _dequantize_aef(raw_int8):
    """Convert quantized int8 embeddings to float32."""
    raw = raw_int8.astype(np.float32)
    m = raw == -128  # nodata
    v = ((raw / 127.5) ** 2) * np.sign(raw)
    v[m] = np.nan
    return v


def create_embedding_rgb(embeddings, bands=[0, 15, 8], vmin=-0.3, vmax=0.3):
    """Create RGB visualization from embedding dimensions."""
    rgb = np.zeros((embeddings.shape[1], embeddings.shape[2], 3), dtype=np.float32)
    
    for i, dim in enumerate(bands):
        band_data = embeddings[dim, :, :]
        band_clipped = np.clip(band_data, vmin, vmax)
        band_normalized = (band_clipped - vmin) / (vmax - vmin)
        rgb[:, :, i] = band_normalized
    
    return rgb


def simplify_crop_labels(cdl_data):
    """Simplify CDL labels to: 0=Other, 1=Corn, 2=Soy."""
    labels = np.zeros_like(cdl_data, dtype=np.int32)
    labels[cdl_data == CORN_CODE] = 1
    labels[cdl_data == SOY_CODE] = 2
    return labels


def prepare_training_data(embeddings, labels, n_samples_per_class=150, random_seed=42):
    """Prepare training and test data from embeddings and labels."""
    print("Preparing training data...")
    
    # Get valid pixels (have both embeddings and labels)
    valid_mask = ~np.isnan(embeddings[0, :, :]) & (labels > 0)
    
    # Extract features and labels
    X_all = embeddings[:, valid_mask].T  # (n_pixels, 64)
    y_all = labels[valid_mask]
    
    print(f"Total valid pixels: {len(y_all)}")
    print(f"  Other: {np.sum(y_all == 0)}")
    print(f"  Corn: {np.sum(y_all == 1)}")
    print(f"  Soy: {np.sum(y_all == 2)}")
    
    # Sample training data - stratified by class
    np.random.seed(random_seed)
    train_indices = []
    
    for class_label in [0, 1, 2]:
        class_indices = np.where(y_all == class_label)[0]
        if len(class_indices) >= n_samples_per_class:
            sampled = np.random.choice(class_indices, n_samples_per_class, replace=False)
        else:
            print(f"Warning: only {len(class_indices)} samples for class {class_label}")
            sampled = class_indices
        train_indices.extend(sampled)
    
    train_indices = np.array(train_indices)
    all_indices = np.arange(len(y_all))
    test_indices = np.setdiff1d(all_indices, train_indices)
    
    X_train = X_all[train_indices]
    y_train = y_all[train_indices]
    X_test = X_all[test_indices]
    y_test = y_all[test_indices]
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'X_all': X_all,
        'y_all': y_all,
        'valid_mask': valid_mask
    }


def align_labels_to_embeddings(cdl_data, embeddings):
    """Resample CDL labels to match embedding resolution."""
    target_h, target_w = embeddings.shape[1], embeddings.shape[2]
    source_h, source_w = cdl_data.shape
    
    zoom_h = target_h / source_h
    zoom_w = target_w / source_w
    
    # Nearest neighbor for categorical data
    resampled = zoom(cdl_data, (zoom_h, zoom_w), order=0)
    
    print(f"Aligned labels from {cdl_data.shape} to {resampled.shape}")
    return resampled


def get_class_names():
    """Return the class names for simplified labels."""
    return ['Other', 'Corn', 'Soy']


def print_crop_statistics(cdl_data):
    """Print statistics about crop coverage."""
    corn_pixels = np.sum(cdl_data == CORN_CODE)
    soy_pixels = np.sum(cdl_data == SOY_CODE)
    other_pixels = np.sum((cdl_data > 0) & (cdl_data != CORN_CODE) & (cdl_data != SOY_CODE))
    total_pixels = corn_pixels + soy_pixels + other_pixels
    
    print(f"\nCrop Statistics:")
    print(f"  Corn: {corn_pixels:,} pixels ({100*corn_pixels/total_pixels:.1f}%)")
    print(f"  Soybeans: {soy_pixels:,} pixels ({100*soy_pixels/total_pixels:.1f}%)")
    print(f"  Other: {other_pixels:,} pixels ({100*other_pixels/total_pixels:.1f}%)")


def prepare_csv_samples(df, n_samples_per_class, random_seed=42):
    """Prepare train and test sets from CSV with embeddings."""
    np.random.seed(random_seed)
    
    train_indices = []
    test_indices = []
    
    for class_name, class_config in n_samples_per_class.items():
        class_label = class_config['label']
        n_train = class_config['n_train']
        n_test = class_config['n_test']
        
        # Get all indices for this class
        class_indices = np.where(df['label'] == class_label)[0]
        
        # Handle cases where there aren't enough samples
        if len(class_indices) < n_train + n_test:
            print(f"Warning: Class {class_name} has only {len(class_indices)} samples, adjusting...")
            n_train = len(class_indices) // 2
            n_test = len(class_indices) - n_train
        
        # Sample training and test indices
        train_idx = np.random.choice(class_indices, n_train, replace=False)
        remaining = list(set(class_indices) - set(train_idx))
        test_idx = np.random.choice(remaining, n_test, replace=False)
        
        train_indices.extend(train_idx)
        test_indices.extend(test_idx)
    
    # Convert to arrays and shuffle
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    # Extract embeddings (columns starting with 'A')
    embedding_columns = [col for col in df.columns if col.startswith('A')]
    X = df[embedding_columns].values
    y = df['label'].values
    
    # Create splits
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    return X_train, y_train, X_test, y_test, train_indices, test_indices


def show_split_statistics(df, train_indices, test_indices, title="Train-Test Split"):
    """Display statistics about the train-test split."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    y_all = df['label'].values
    y_train = y_all[train_indices]
    y_test = y_all[test_indices]
    
    print(f"Training set: {len(train_indices)} samples")
    for label in sorted(np.unique(y_all)):
        count = np.sum(y_train == label)
        class_name = df[df['label'] == label]['class_name'].iloc[0]
        print(f"  {class_name}: {count} samples")
    
    print(f"\nTest set: {len(test_indices)} samples")
    for label in sorted(np.unique(y_all)):
        count = np.sum(y_test == label)
        class_name = df[df['label'] == label]['class_name'].iloc[0]
        print(f"  {class_name}: {count} samples")
    print(f"{'='*60}")


def compute_clusters(embeddings, k_values=[3, 5, 10], random_state=42):
    n_bands, height, width = embeddings.shape
    X = embeddings.reshape(n_bands, -1).T

    results = {}

    for k in k_values:
        print(f"k={k}")
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            batch_size=4096,
            random_state=random_state,
            n_init=3,
            max_iter=100
        )
        labels = kmeans.fit_predict(X)
        results[k] = labels.reshape(height, width)

    return results


def predict_on_embeddings(clf, embeddings):
    """Apply a trained classifier to spatial embeddings."""
    # Reshape embeddings to (n_pixels, 64)
    n_bands, height, width = embeddings.shape
    X_spatial = embeddings.reshape(n_bands, -1).T  # (H*W, 64)
    
    # Predict
    y_pred = clf.predict(X_spatial)
    
    # Reshape back to spatial
    y_pred_spatial = y_pred.reshape(height, width)
    
    return y_pred_spatial


def prepare_sample_coordinates(df, train_idx, test_idx):
    """Extract coordinates from .geo column and add split indicators."""
    
    def extract_coordinates(geo_string):
        """Extract lon and lat from geo JSON string"""
        try:
            geo_data = json.loads(geo_string)
            coords = geo_data['coordinates']
            # GeoJSON is [lon, lat]
            return coords[0], coords[1]
        except:
            return None, None
    
    # Check if there are already latitude/longitude columns
    if 'latitude' in df.columns and 'longitude' in df.columns:
        print(" Using existing latitude/longitude columns")
    else:
        print(" Extracting coordinates from .geo column")
        # Add coordinates to dataframe
        lons, lats = [], []
        for geo_str in df['.geo']:
            lon, lat = extract_coordinates(geo_str)
            lons.append(lon)
            lats.append(lat)
        
        df['longitude'] = lons
        df['latitude'] = lats
    
    # Create split indicator
    df['split'] = 'unused'
    df.loc[train_idx, 'split'] = 'train'
    df.loc[test_idx, 'split'] = 'test'
    
    print(f"\nCoordinate extraction complete:")
    print(f"  - Total samples: {len(df)}")
    print(f"  - Valid coordinates: {df['longitude'].notna().sum()}/{len(df)}")
    print(f"\nCoordinate ranges:")
    print(f"  - Longitude: [{df['longitude'].min():.2f}, {df['longitude'].max():.2f}]")
    print(f"  - Latitude: [{df['latitude'].min():.2f}, {df['latitude'].max():.2f}]")
    print(f"\nSample distribution:")
    print(df['split'].value_counts())
    
    return df
