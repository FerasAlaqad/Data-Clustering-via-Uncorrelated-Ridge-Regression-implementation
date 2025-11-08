"""
Dataset loaders following the paper's specifications.
All datasets return X as (d, n) format where d=features, n=samples.
Paper Table II shows dataset details.
"""

import numpy as np
from pathlib import Path
from typing import Tuple
import scipy.io as sio
from PIL import Image
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_glioma(dataset_path: str = "datasets/GLIOMA.mat") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load GLIOMA dataset
    Paper Table II: 50 samples, 4434 genes, 4 classes
    
    Returns
    -------
    X : ndarray of shape (4434, 50)
        Feature matrix (genes × samples)
    y : ndarray of shape (50,)
        True labels (0-3)
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
            raise ValueError("Could not find labels")
    
    # Ensure (d, n) format
    if X.shape[0] > X.shape[1]:
        X = X.T
    
    # Convert to 0-indexed
    if y.min() == 1:
        y = y - 1
    
    # Paper doesn't specify normalization explicitly, but it's standard for gene expression
    # Standardize features (each gene)
    X = StandardScaler().fit_transform(X.T).T
    
    return X, y


def load_colon(dataset_path: str = "datasets/COLON.mat") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load COLON dataset from .mat file
    Paper Table II: 62 samples, 2000 genes, 2 classes
    
    Returns
    -------
    X : ndarray of shape (2000, 62)
        Feature matrix (genes × samples)
    y : ndarray of shape (62,)
        True labels (0-1)
    """
    mat_data = sio.loadmat(dataset_path)
    
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
            raise ValueError("Could not find labels")
    
    # Ensure (d, n) format
    if X.shape[0] > X.shape[1]:
        X = X.T
    
    # Convert to 0-indexed
    if y.min() == 1:
        y = y - 1
    
    # Standardize features
    X = StandardScaler().fit_transform(X.T).T
    
    return X, y


def load_colon_csv(dataset_path: str = "datasets/colon - labled.csv") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load COLON dataset from CSV file
    Paper Table II: 62 samples, 2000 genes, 2 classes
    
    Returns
    -------
    X : ndarray of shape (2000, 62)
        Feature matrix (genes × samples)
    y : ndarray of shape (62,)
        True labels (0-1)
    """
    df = pd.read_csv(dataset_path)
    
    # Drop unnamed columns
    unnamed_cols = [col for col in df.columns if col.lower().startswith("unnamed")]
    df = df.drop(columns=unnamed_cols)
    
    if "Class" not in df.columns:
        raise ValueError("Expected 'Class' column")
    
    # Extract labels
    y_raw = df["Class"].astype(str).str.lower()
    y = (y_raw != "normal").astype(int).to_numpy()
    
    # Extract features
    feature_df = df.drop(columns=["Class"])
    X = feature_df.to_numpy(dtype=float).T  # (d, n)
    
    # Standardize features
    X = StandardScaler().fit_transform(X.T).T
    
    return X, y


def load_att_faces(dataset_path: str = "datasets/att_faces") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load AT&T (ORL) face database
    Paper Table II: 400 images, 10304 features, 40 subjects
    
    Returns
    -------
    X : ndarray of shape (10304, 400)
        Feature matrix (pixels × images)
    y : ndarray of shape (400,)
        True labels (0-39)
    """
    dataset_path = Path(dataset_path)
    
    images = []
    labels = []
    
    # 40 subjects, each has folder s1, s2, ..., s40
    for subject_idx in range(1, 41):
        subject_dir = dataset_path / f"s{subject_idx}"
        if not subject_dir.exists():
            continue
        
        # Each subject has 10 images
        for img_file in sorted(subject_dir.glob("*.pgm")):
            img = Image.open(img_file).convert('L')
            img_array = np.array(img).flatten().astype(float)
            images.append(img_array)
            labels.append(subject_idx - 1)
    
    X = np.array(images).T  # (d, n)
    y = np.array(labels)
    
    # Normalize pixel values to [0, 1]
    X = X / 255.0
    
    return X, y


def load_gt_faces(dataset_path: str = "datasets/gt_db") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Georgia Tech face database
    Paper Table II: 750 images, 50 subjects
    
    Returns
    -------
    X : ndarray of shape (d, 750)
        Feature matrix (pixels × images)
    y : ndarray of shape (750,)
        True labels (0-49)
    """
    dataset_path = Path(dataset_path)
    
    # Handle nested directory structure
    if (dataset_path / "gt_db").exists():
        dataset_path = dataset_path / "gt_db"
    
    images = []
    labels = []
    
    # 50 subjects, folders s01, s02, ..., s50
    for subject_idx in range(1, 51):
        subject_dir = dataset_path / f"s{subject_idx:02d}"
        if not subject_dir.exists():
            continue
        
        # Load all images for this subject
        for img_file in sorted(subject_dir.glob("*.jpg")):
            img = Image.open(img_file).convert('L')
            # Resize to consistent size (64x64 as commonly used)
            img = img.resize((64, 64))
            img_array = np.array(img).flatten().astype(float)
            images.append(img_array)
            labels.append(subject_idx - 1)
    
    X = np.array(images).T
    y = np.array(labels)
    
    # Normalize to [0, 1]
    X = X / 255.0
    
    return X, y


def load_flower17(dataset_path: str = "datasets/flower17") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Oxford 17 Flowers dataset
    Paper Table II: 1360 images, 17 categories
    
    Returns
    -------
    X : ndarray of shape (d, 1360)
        Feature matrix (pixels × images)
    y : ndarray of shape (1360,)
        True labels (0-16)
    """
    dataset_path = Path(dataset_path)
    
    # Images in jpg/ subfolder
    jpg_dir = dataset_path / "jpg"
    if not jpg_dir.exists():
        jpg_dir = dataset_path
    
    images = []
    labels = []
    
    # Images named image_0001.jpg to image_1360.jpg
    # First 80 images = class 0, next 80 = class 1, etc.
    image_files = sorted(jpg_dir.glob("image_*.jpg"))
    
    for idx, img_file in enumerate(image_files):
        img = Image.open(img_file).convert('RGB')
        # Resize to consistent size
        img = img.resize((128, 128))
        img_array = np.array(img).flatten().astype(float)
        images.append(img_array)
        
        # 80 images per class
        label = idx // 80
        labels.append(label)
    
    X = np.array(images).T
    y = np.array(labels)
    
    # Normalize to [0, 1]
    X = X / 255.0
    
    return X, y


def load_imm_faces(dataset_path: str = "datasets/imm") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load IMM face database
    Paper Table II and Fig. 5: 240 images, 40 subjects
    
    Returns
    -------
    X : ndarray of shape (d, 240)
        Feature matrix (pixels × images)
    y : ndarray of shape (240,)
        True labels (0-39)
    """
    dataset_path = Path(dataset_path)
    
    images = []
    labels = []
    
    # IMM database: images named like 01-1m.jpg, 01-2m.jpg, etc.
    # Subject IDs range from 01 to 40
    image_files = sorted(dataset_path.rglob("*.jpg"))
    
    for img_file in image_files:
        if img_file.suffix.lower() not in ['.jpg', '.png', '.bmp']:
            continue
        
        img = Image.open(img_file).convert('L')
        img = img.resize((64, 64))
        img_array = np.array(img).flatten().astype(float)
        images.append(img_array)
        
        # Extract subject ID from filename (e.g., "01-1m.jpg" -> subject 0)
        try:
            subject_id = int(img_file.stem.split('-')[0]) - 1
            labels.append(subject_id)
        except:
            continue
    
    X = np.array(images).T
    y = np.array(labels)
    
    # Normalize to [0, 1]
    X = X / 255.0
    
    return X, y


# ---------------------------------------------------------------------------
# Synthetic toy datasets (Paper Figures 3-4 and additional tests)
# ---------------------------------------------------------------------------


def load_toy_dataset(
    dataset_type: str,
    random_state: int = 42,
    use_saved: bool = True,
    data_dir: str = "datasets",
    save_generated: bool = True,
    allow_generate: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load synthetic toy dataset.

    Parameters
    ----------
    dataset_type : str
        Either '3cluster', 'multicluster', 'nested_circles', or 'moons'.
    random_state : int
        Random seed for reproducibility (only used if generation is required).
    use_saved : bool
        If True, load from saved files; if False, generate fresh data.
    data_dir : str
        Directory containing saved datasets.
    save_generated : bool
        If True, persist a newly generated dataset to disk.
    allow_generate : bool
        If False, never generate a new dataset; instead raise FileNotFoundError when
        a saved dataset is unavailable.

    Returns
    -------
    X : ndarray of shape (d, n)
        Feature matrix (features × samples).
    y : ndarray of shape (n,)
        True labels.
    """
    npz_path = Path(data_dir) / f"{dataset_type}.npz"

    # Try to load saved dataset first
    if use_saved and npz_path.exists():
        try:
            data = np.load(npz_path)
            return data['X'], data['y']
        except Exception:
            if not allow_generate:
                raise
            # Fall back to generation on read failure

    if not allow_generate and not npz_path.exists():
        raise FileNotFoundError(
            f"Saved dataset '{dataset_type}' not found at {npz_path}. "
            "Generate it via generate_toy_datasets.py before running experiments."
        )

    # Generate dataset if not saved or use_saved=False
    if dataset_type == '3cluster':
        X, y = generate_3cluster_gaussian(random_state)
    elif dataset_type == 'multicluster':
        X, y = generate_multicluster(random_state)
    elif dataset_type == 'nested_circles':
        X, y = generate_nested_circles(random_state)
    elif dataset_type == 'moons':
        X, y = generate_moons(random_state)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Persist generated dataset for future runs if requested
    if save_generated:
        try:
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(npz_path, X=X, y=y)
        except Exception:
            pass

    return X, y


def generate_3cluster_gaussian(random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 3-cluster Gaussian distributed data as in Paper Figure 3.

    Returns
    -------
    X : ndarray of shape (2, n)
        2D feature matrix.
    y : ndarray of shape (n,)
        True cluster labels (0, 1, 2).
    """
    rng = np.random.RandomState(random_state)

    n_per_cluster = 100

    mean1 = np.array([-3, -3])
    cov1 = np.array([[0.5, 0], [0, 0.5]])
    X1 = rng.multivariate_normal(mean1, cov1, n_per_cluster).T
    y1 = np.zeros(n_per_cluster, dtype=int)

    mean2 = np.array([3, -3])
    cov2 = np.array([[0.5, 0], [0, 0.5]])
    X2 = rng.multivariate_normal(mean2, cov2, n_per_cluster).T
    y2 = np.ones(n_per_cluster, dtype=int)

    mean3 = np.array([0, 3])
    cov3 = np.array([[0.5, 0], [0, 0.5]])
    X3 = rng.multivariate_normal(mean3, cov3, n_per_cluster).T
    y3 = np.full(n_per_cluster, 2, dtype=int)

    X = np.hstack([X1, X2, X3])
    y = np.concatenate([y1, y2, y3])

    indices = rng.permutation(len(y))
    X = X[:, indices]
    y = y[indices]

    return X, y


def generate_multicluster(random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate multicluster data as in Paper Figure 4.

    Returns
    -------
    X : ndarray of shape (2, n)
        2D feature matrix.
    y : ndarray of shape (n,)
        True cluster labels (0-5).
    """
    rng = np.random.RandomState(random_state)

    n_per_cluster = 100
    cluster_centers = [
        np.array([-5, -5]),
        np.array([0, -5]),
        np.array([5, -5]),
        np.array([-5, 5]),
        np.array([0, 5]),
        np.array([5, 5]),
    ]

    X_list = []
    y_list = []

    for cluster_id, center in enumerate(cluster_centers):
        if cluster_id % 2 == 0:
            cov = np.array([[0.4, 0], [0, 0.4]])
        else:
            cov = np.array([[0.6, 0.1], [0.1, 0.6]])

        X_cluster = rng.multivariate_normal(center, cov, n_per_cluster).T
        y_cluster = np.full(n_per_cluster, cluster_id, dtype=int)

        X_list.append(X_cluster)
        y_list.append(y_cluster)

    X = np.hstack(X_list)
    y = np.concatenate(y_list)

    indices = rng.permutation(len(y))
    X = X[:, indices]
    y = y[indices]

    return X, y


def generate_nested_circles(random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate nested circles clustering problem (additional test case).

    Returns
    -------
    X : ndarray of shape (2, n)
        2D feature matrix.
    y : ndarray of shape (n,)
        True cluster labels (0, 1).
    """
    rng = np.random.RandomState(random_state)

    n_per_cluster = 150

    angles1 = rng.uniform(0, 2 * np.pi, n_per_cluster)
    radius1 = 1.0 + rng.normal(0, 0.1, n_per_cluster)
    X1 = np.vstack([radius1 * np.cos(angles1), radius1 * np.sin(angles1)])
    y1 = np.zeros(n_per_cluster, dtype=int)

    angles2 = rng.uniform(0, 2 * np.pi, n_per_cluster)
    radius2 = 3.0 + rng.normal(0, 0.1, n_per_cluster)
    X2 = np.vstack([radius2 * np.cos(angles2), radius2 * np.sin(angles2)])
    y2 = np.ones(n_per_cluster, dtype=int)

    X = np.hstack([X1, X2])
    y = np.concatenate([y1, y2])

    indices = rng.permutation(len(y))
    X = X[:, indices]
    y = y[indices]

    return X, y


def generate_moons(random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate two moons clustering problem (additional test case).

    Returns
    -------
    X : ndarray of shape (2, n)
        2D feature matrix.
    y : ndarray of shape (n,)
        True cluster labels (0, 1).
    """
    rng = np.random.RandomState(random_state)

    n_per_cluster = 150

    angles1 = np.linspace(0, np.pi, n_per_cluster)
    X1 = np.vstack([np.cos(angles1), np.sin(angles1)])
    X1 += rng.normal(0, 0.1, X1.shape)
    y1 = np.zeros(n_per_cluster, dtype=int)

    angles2 = np.linspace(0, np.pi, n_per_cluster)
    X2 = np.vstack([1 - np.cos(angles2), 0.5 - np.sin(angles2)])
    X2 += rng.normal(0, 0.1, X2.shape)
    y2 = np.ones(n_per_cluster, dtype=int)

    X = np.hstack([X1, X2])
    y = np.concatenate([y1, y2])

    indices = rng.permutation(len(y))
    X = X[:, indices]
    y = y[indices]

    return X, y


def load_ehtz53(dataset_path: str = "datasets/EHTZ_53") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load EHTZ_53 dataset
    Paper Table II: Dataset mentioned but format not specified
    
    This is a placeholder - adjust based on actual dataset format
    """
    # Try .mat format first
    if Path(dataset_path + ".mat").exists():
        mat_data = sio.loadmat(dataset_path + ".mat")
        
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
                raise ValueError("Could not find labels")
        
        # Ensure (d, n) format
        if X.shape[0] > X.shape[1]:
            X = X.T
        
        # Convert to 0-indexed
        if y.min() == 1:
            y = y - 1
        
        return X, y
    else:
        raise FileNotFoundError(f"EHTZ_53 dataset not found at {dataset_path}")


def load_fei_faces(dataset_path: str = "datasets/FEI") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load FEI face database
    Paper Table II: Dataset mentioned but format not specified
    
    This is a placeholder - adjust based on actual dataset format
    """
    dataset_path = Path(dataset_path)
    
    images = []
    labels = []
    
    # FEI database structure varies - try common patterns
    image_files = list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.png"))
    
    for img_file in sorted(image_files):
        img = Image.open(img_file).convert('L')
        img = img.resize((64, 64))
        img_array = np.array(img).flatten().astype(float)
        images.append(img_array)
        
        # Extract subject ID from filename
        # Adjust based on actual naming convention
        try:
            subject_id = int(img_file.stem.split('-')[0]) - 1
            labels.append(subject_id)
        except:
            labels.append(0)
    
    X = np.array(images).T
    y = np.array(labels)
    
    # Normalize to [0, 1]
    X = X / 255.0
    
    return X, y


if __name__ == "__main__":
    """Test dataset loaders"""
    print("Testing dataset loaders...")
    print("="*70)
    
    # Test GLIOMA
    try:
        print("\n1. GLIOMA:")
        X, y = load_glioma()
        print(f"   X shape: {X.shape} (features × samples)")
        print(f"   Classes: {len(np.unique(y))}")
        print(f"   Samples per class: {np.bincount(y.astype(int))}")
        print(f"   Feature range: [{X.min():.3f}, {X.max():.3f}]")
    except Exception as e:
        print(f"   Failed: {e}")
    
    # Test COLON
    try:
        print("\n2. COLON:")
        X, y = load_colon_csv()
        print(f"   X shape: {X.shape} (features × samples)")
        print(f"   Classes: {len(np.unique(y))}")
        print(f"   Samples per class: {np.bincount(y.astype(int))}")
        print(f"   Feature range: [{X.min():.3f}, {X.max():.3f}]")
    except Exception as e:
        print(f"   Failed: {e}")
    
    # Test AT&T
    try:
        print("\n3. AT&T Faces:")
        X, y = load_att_faces()
        print(f"   X shape: {X.shape} (features × samples)")
        print(f"   Classes: {len(np.unique(y))}")
        print(f"   Feature range: [{X.min():.3f}, {X.max():.3f}]")
    except Exception as e:
        print(f"   Failed: {e}")
    
    print("\n" + "="*70)