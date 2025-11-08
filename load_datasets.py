"""
Dataset loaders for benchmark datasets used in the RURR-SL paper.
Handles various formats: .mat files, image directories, etc.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import scipy.io as sio
from PIL import Image
import pandas as pd


def load_glioma(dataset_path: str = "datasets/GLIOMA.mat") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load GLIOMA dataset from .mat file.
    
    Parameters
    ----------
    dataset_path : str
        Path to GLIOMA.mat file
        
    Returns
    -------
    X : ndarray of shape (d, n)
        Feature matrix where d=4434 genes, n=50 samples
    y : ndarray of shape (n,)
        True labels (4 glioma subtypes)
    """
    mat_data = sio.loadmat(dataset_path)
    
    # Common keys in .mat files: 'X', 'Y', 'data', 'label', 'fea', 'gnd'
    # Try different possible variable names
    if 'X' in mat_data:
        X = mat_data['X']
    elif 'data' in mat_data:
        X = mat_data['data']
    elif 'fea' in mat_data:
        X = mat_data['fea']
    else:
        # Find the first non-metadata key
        keys = [k for k in mat_data.keys() if not k.startswith('__')]
        X = mat_data[keys[0]]
    
    if 'Y' in mat_data:
        y = mat_data['Y'].ravel()
    elif 'label' in mat_data:
        y = mat_data['label'].ravel()
    elif 'gnd' in mat_data:
        y = mat_data['gnd'].ravel()
    else:
        keys = [k for k in mat_data.keys() if not k.startswith('__')]
        if len(keys) > 1:
            y = mat_data[keys[1]].ravel()
        else:
            raise ValueError("Could not find labels in .mat file")
    
    # Ensure X is (d, n) format
    if X.shape[0] < X.shape[1]:
        # Already in (d, n) format
        pass
    else:
        # In (n, d) format, transpose
        X = X.T
    
    # Convert labels to 0-indexed if needed
    if y.min() == 1:
        y = y - 1
    
    return X, y


def load_colon(dataset_path: str = "datasets/COLON.mat") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load COLON dataset from .mat file.
    
    Parameters
    ----------
    dataset_path : str
        Path to COLON.mat file
        
    Returns
    -------
    X : ndarray of shape (d, n)
        Feature matrix where d=2000 genes, n=62 samples
    y : ndarray of shape (n,)
        True labels (2 classes: tumor vs normal)
    """
    mat_data = sio.loadmat(dataset_path)
    
    # Try common variable names
    if 'X' in mat_data:
        X = mat_data['X']
    elif 'data' in mat_data:
        X = mat_data['data']
    elif 'fea' in mat_data:
        X = mat_data['fea']
    else:
        keys = [k for k in mat_data.keys() if not k.startswith('__')]
        X = mat_data[keys[0]]
    
    if 'Y' in mat_data:
        y = mat_data['Y'].ravel()
    elif 'label' in mat_data:
        y = mat_data['label'].ravel()
    elif 'gnd' in mat_data:
        y = mat_data['gnd'].ravel()
    else:
        keys = [k for k in mat_data.keys() if not k.startswith('__')]
        if len(keys) > 1:
            y = mat_data[keys[1]].ravel()
        else:
            raise ValueError("Could not find labels in .mat file")
    
    # Ensure X is (d, n) format
    if X.shape[0] < X.shape[1]:
        pass
    else:
        X = X.T
    
    # Convert labels to 0-indexed
    if y.min() == 1:
        y = y - 1
    
    return X, y


def load_colon_csv(dataset_path: str = "datasets/colon - labled.csv") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load COLON dataset from labeled CSV file.

    Parameters
    ----------
    dataset_path : str
        Path to colon - labled.csv file (62 samples × 2000 genes + class label)

    Returns
    -------
    X : ndarray of shape (d, n)
        Feature matrix where d=2000 genes, n=62 samples
    y : ndarray of shape (n,)
        True labels (0 = Normal, 1 = Abnormal)
    """
    df = pd.read_csv(dataset_path)

    # Drop any unnamed index columns
    unnamed_cols = [col for col in df.columns if col.lower().startswith("unnamed")]
    df = df.drop(columns=unnamed_cols)

    if "Class" not in df.columns:
        raise ValueError("Expected 'Class' column with labels in colon dataset.")

    y_raw = df["Class"].astype(str).str.lower()
    y = (y_raw != "normal").astype(int).to_numpy()  # Abnormal -> 1, Normal -> 0

    feature_df = df.drop(columns=["Class"])
    X = feature_df.to_numpy(dtype=float).T  # (d, n)

    return X, y


def load_att_faces(dataset_path: str = "datasets/att_faces") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load AT&T (ORL) face database.
    
    Parameters
    ----------
    dataset_path : str
        Path to att_faces directory containing s1, s2, ..., s40 folders
        
    Returns
    -------
    X : ndarray of shape (d, n)
        Feature matrix where d=10304 (92×112 pixels), n=400 images
    y : ndarray of shape (n,)
        True labels (40 subjects)
    """
    dataset_path = Path(dataset_path)
    
    images = []
    labels = []
    
    # Each subject has a folder s1, s2, ..., s40
    for subject_idx in range(1, 41):
        subject_dir = dataset_path / f"s{subject_idx}"
        if not subject_dir.exists():
            continue
        
        # Each subject has 10 images
        for img_file in sorted(subject_dir.glob("*.pgm")):
            img = Image.open(img_file).convert('L')  # Grayscale
            img_array = np.array(img).flatten()
            images.append(img_array)
            labels.append(subject_idx - 1)  # 0-indexed
    
    X = np.array(images).T  # Shape: (d, n)
    y = np.array(labels)
    
    return X, y


def load_gt_faces(dataset_path: str = "datasets/gt_db") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Georgia Tech face database.
    
    Parameters
    ----------
    dataset_path : str
        Path to gt_db directory
        
    Returns
    -------
    X : ndarray of shape (d, n)
        Feature matrix, n=750 images (50 subjects × 15 images)
    y : ndarray of shape (n,)
        True labels (50 subjects)
    """
    dataset_path = Path(dataset_path)
    
    # Handle nested structure (gt_db/gt_db/...)
    if (dataset_path / "gt_db").exists():
        dataset_path = dataset_path / "gt_db"
    
    images = []
    labels = []
    
    # Each subject has a folder s01, s02, ..., s50
    for subject_idx in range(1, 51):
        subject_dir = dataset_path / f"s{subject_idx:02d}"
        if not subject_dir.exists():
            continue
        
        # Load all images for this subject
        for img_file in sorted(subject_dir.glob("*.jpg")):
            img = Image.open(img_file).convert('L')
            # Resize to consistent size (e.g., 64x64)
            img = img.resize((64, 64))
            img_array = np.array(img).flatten()
            images.append(img_array)
            labels.append(subject_idx - 1)
    
    X = np.array(images).T
    y = np.array(labels)
    
    return X, y


def load_flower17(dataset_path: str = "datasets/flower17") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Oxford 17 Flowers dataset.
    
    Parameters
    ----------
    dataset_path : str
        Path to flower17 directory containing jpg/ folder
        
    Returns
    -------
    X : ndarray of shape (d, n)
        Feature matrix, n=1360 images (17 categories × 80 images)
    y : ndarray of shape (n,)
        True labels (17 flower categories)
    """
    dataset_path = Path(dataset_path)
    
    # Images are in jpg/ subfolder
    jpg_dir = dataset_path / "jpg"
    if not jpg_dir.exists():
        jpg_dir = dataset_path
    
    images = []
    labels = []
    
    # Images are named image_0001.jpg to image_1360.jpg
    # First 80 are class 0, next 80 are class 1, etc.
    image_files = sorted(jpg_dir.glob("image_*.jpg"))
    
    for idx, img_file in enumerate(image_files):
        img = Image.open(img_file).convert('RGB')
        # Resize to consistent size
        img = img.resize((128, 128))
        img_array = np.array(img).flatten()
        images.append(img_array)
        
        # Label is determined by position (80 images per class)
        label = idx // 80
        labels.append(label)
    
    X = np.array(images).T
    y = np.array(labels)
    
    return X, y


def load_imm_faces(dataset_path: str = "datasets/imm") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load IMM face database.
    
    Parameters
    ----------
    dataset_path : str
        Path to IMM face database directory
        
    Returns
    -------
    X : ndarray of shape (d, n)
        Feature matrix, n=240 images (40 subjects × 6 images)
    y : ndarray of shape (n,)
        True labels (40 subjects)
    """
    dataset_path = Path(dataset_path)
    
    images = []
    labels = []
    
    # IMM database structure varies, try to find image files
    image_files = list(dataset_path.rglob("*.jpg")) + list(dataset_path.rglob("*.asf"))
    
    # Group by subject ID (usually in filename like 01-1m.jpg, 01-2m.jpg, etc.)
    for img_file in sorted(image_files):
        if img_file.suffix.lower() not in ['.jpg', '.png', '.bmp']:
            continue
        
        img = Image.open(img_file).convert('L')
        img = img.resize((64, 64))
        img_array = np.array(img).flatten()
        images.append(img_array)
        
        # Extract subject ID from filename (e.g., "01-1m.jpg" -> subject 0)
        subject_id = int(img_file.stem.split('-')[0]) - 1
        labels.append(subject_id)
    
    X = np.array(images).T
    y = np.array(labels)
    
    return X, y


# Example usage
if __name__ == "__main__":
    print("Loading GLIOMA dataset...")
    X, y = load_glioma()
    print(f"  X shape: {X.shape} (features × samples)")
    print(f"  y shape: {y.shape}")
    print(f"  Number of classes: {len(np.unique(y))}")
    print(f"  Class distribution: {np.bincount(y.astype(int))}")
    print()
    
    # Test other datasets if available
    try:
        print("Loading AT&T faces...")
        X, y = load_att_faces()
        print(f"  X shape: {X.shape}")
        print(f"  Number of subjects: {len(np.unique(y))}")
    except Exception as e:
        print(f"  Could not load: {e}")

