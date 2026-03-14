import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
import alphashape
from matplotlib.path import Path
import os

# =============================================================================
# CUSTOM METRICS AND K-MEANS
# =============================================================================

def custom_silhouette_score(X, labels):
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1 or len(unique_labels) == len(X):
        return -1.0
        
    n_samples = len(X)
    s_scores = np.zeros(n_samples)
    
    for cluster_label in unique_labels:
        cluster_mask = (labels == cluster_label)
        cluster_indices = np.where(cluster_mask)[0]
        pts_in_cluster = X[cluster_mask]
        
        if len(pts_in_cluster) <= 1:
            s_scores[cluster_indices] = 0.0
            continue
            
        for i, idx in enumerate(cluster_indices):
            pt = pts_in_cluster[i]
            dists_in_cluster = np.linalg.norm(pts_in_cluster - pt, axis=1)
            a_i = np.sum(dists_in_cluster) / (len(pts_in_cluster) - 1)
            
            b_i = np.inf
            for other_cluster in unique_labels:
                if other_cluster == cluster_label: continue
                other_pts = X[labels == other_cluster]
                if len(other_pts) == 0: continue
                avg_dist = np.mean(np.linalg.norm(other_pts - pt, axis=1))
                if avg_dist < b_i:
                    b_i = avg_dist
            
            s_scores[idx] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0.0
            
    return np.mean(s_scores)

def custom_davies_bouldin_score(X, labels):
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    if n_clusters <= 1:
        return np.inf
    
    cluster_k = [X[labels == k] for k in unique_labels]
    centroids = [np.mean(k, axis=0) if len(k) > 0 else np.zeros((X.shape[1],)) for k in cluster_k]
    
    S = []
    for i in range(n_clusters):
        if len(cluster_k[i]) == 0:
            S.append(0)
        else:
            S.append(np.mean(np.linalg.norm(cluster_k[i] - centroids[i], axis=1)))
            
    R = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i != j:
                distance = np.linalg.norm(centroids[i] - centroids[j])
                R[i,j] = np.inf if distance == 0 else (S[i] + S[j]) / distance
                    
    D = np.max(R, axis=1)
    return np.mean(D)

class CustomKMeans:
    def __init__(self, n_clusters=8, max_iter=100, random_state=42, n_init=5):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_init = n_init

    def fit(self, X):
        np.random.seed(self.random_state)
        best_inertia = np.inf
        n_samples = X.shape[0]
        
        for _ in range(self.n_init):
            indices = np.random.choice(n_samples, min(self.n_clusters, n_samples), replace=False)
            centers = X[indices]
            
            for _ in range(self.max_iter):
                dists = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
                labels = np.argmin(dists, axis=1)
                
                new_centers = []
                for i in range(len(centers)):
                    cluster_points = X[labels == i]
                    if len(cluster_points) > 0:
                        new_centers.append(cluster_points.mean(axis=0))
                    else:
                        new_centers.append(X[np.random.randint(n_samples)])
                new_centers = np.array(new_centers)
                
                if np.allclose(centers, new_centers):
                    break
                centers = new_centers
                
            dists = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
            inertia = np.sum(np.min(dists, axis=1)**2)
            
            if inertia < best_inertia:
                best_inertia = inertia
                self.cluster_centers_ = centers
                self.labels_ = labels
                
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def find_optimal_k(X, min_k=2, max_k_factor=0.1):
    n = X.shape[0]
    max_k = min(int(np.log10(max(n,10))) * 8 + 2, int(max_k_factor * n), 100)
    max_k = max(min_k, max_k)
    best_k = min_k
    best_score = -1

    for k in range(min_k, max_k + 1):
        try:
            kmeans = CustomKMeans(n_clusters=k, random_state=42, n_init=3)
            labels = kmeans.fit_predict(X)
            score = custom_silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k
        except:
            continue
    return best_k

def get_bezier_points(p0, p1, p2, num_points=20):
    t = np.linspace(0, 1, num_points)
    term1 = ((1-t)**2)[:, None] * p0
    term2 = (2*(1-t)*t)[:, None] * p1
    term3 = (t**2)[:, None] * p2
    return term1 + term2 + term3

def ray_casting_inside(point, boundary_points):
    if boundary_points.shape[1] != 2:
        mins = np.min(boundary_points, axis=0)
        maxs = np.max(boundary_points, axis=0)
        return np.all((point >= mins - 0.01) & (point <= maxs + 0.01))
    poly_path = Path(boundary_points)
    return poly_path.contains_point(point)

def get_boundary(Xc, hull_type='convex', alpha=0.1):
    if Xc.shape[0] <= Xc.shape[1]:
        return Xc
        
    if hull_type == 'alpha':
        try:
            alpha_shape = alphashape.alphashape(Xc, alpha)
            if alpha_shape.geom_type == 'Polygon':
                boundary = np.array(alpha_shape.exterior.coords[:-1])
            else:
                boundary = Xc
        except:
            boundary = Xc
    else:
        try:
            hull = ConvexHull(Xc)
            boundary = Xc[hull.vertices]
        except:
            boundary = Xc
    return boundary

def get_centroid(Xc):
    k_opt = find_optimal_k(Xc)
    kmeans = CustomKMeans(n_clusters=k_opt, random_state=42, n_init=10)
    kmeans.fit(Xc)
    return kmeans.cluster_centers_.mean(axis=0)

# =============================================================================
# AUGMENTATION METHODS
# =============================================================================

def generate_bezier_convex(Xc, class_size, centroid, boundary):
    target = int(class_size * 0.5)

    synthetic = []
    attempts = 0
    max_attempts = target * 20

    while len(synthetic) < target and attempts < max_attempts:
        i = np.random.randint(len(boundary))
        p0 = boundary[i]
        p2 = boundary[(i+1) % len(boundary)]
        p1 = 0.5 * (p0 + p2)

        bezier_pts = get_bezier_points(p0, p1, p2)
        bt = bezier_pts[np.random.randint(len(bezier_pts))]
        st = centroid + 0.7 * (bt - centroid)

        if ray_casting_inside(st, boundary):
            synthetic.append(st)
        attempts += 1

    validation_score = len(synthetic) / attempts if attempts > 0 else 0.0
    return np.array(synthetic), validation_score

def generate_bezier_alpha(Xc, class_size, centroid, boundary):
    target = int(class_size * 0.5)

    synthetic = []
    attempts = 0
    max_attempts = target * 20

    while len(synthetic) < target and attempts < max_attempts:
        i = np.random.randint(len(boundary))
        p0 = boundary[i]
        p2 = boundary[(i+1) % len(boundary)]
        p1 = 0.5 * (p0 + p2)

        bezier_pts = get_bezier_points(p0, p1, p2)
        bt = bezier_pts[np.random.randint(len(bezier_pts))]
        st = centroid + 0.7 * (bt - centroid)

        if ray_casting_inside(st, boundary):
            synthetic.append(st)
        attempts += 1

    validation_score = len(synthetic) / attempts if attempts > 0 else 0.0
    return np.array(synthetic), validation_score

def generate_gaussian_shell(Xc, class_size, centroid, boundary, rhull):
    target = int(class_size * 0.5)

    synthetic = []
    attempts = 0
    max_attempts = target * 20

    while len(synthetic) < target and attempts < max_attempts:
        direction = np.random.normal(size=Xc.shape[1])
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        else:
            attempts += 1
            continue

        r = np.random.uniform(0, 0.7 * rhull)
        st = centroid + r * direction

        if ray_casting_inside(st, boundary):
            synthetic.append(st)
        attempts += 1

    validation_score = len(synthetic) / attempts if attempts > 0 else 0.0
    return np.array(synthetic), validation_score

def generate_bezier_chord(Xc, class_size, centroid, boundary):
    target = int(class_size * 0.5)

    n = len(boundary)
    synthetic = []
    attempts = 0
    max_attempts = target * 20

    if n < 3:
        return np.array([]), 0.0

    while len(synthetic) < target and attempts < max_attempts:
        i = np.random.randint(n)
        j = np.random.randint(n)

        if abs(i-j) <= 1 or abs(i-j) >= n-1:
            attempts += 1
            continue

        p0 = boundary[i]
        p2 = boundary[j]
        p1 = 0.5 * (p0 + p2)

        bezier_pts = get_bezier_points(p0, p1, p2)
        bt = bezier_pts[np.random.randint(len(bezier_pts))]
        st = centroid + 0.7 * (bt - centroid)

        if ray_casting_inside(st, boundary):
            synthetic.append(st)
        attempts += 1

    validation_score = len(synthetic) / attempts if attempts > 0 else 0.0
    return np.array(synthetic), validation_score

def generate_linear_contraction(Xc, class_size, centroid, boundary):
    target = int(class_size * 0.5)

    t_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    candidates = []

    for b in boundary:
        for t in t_values:
            s = (1 - t) * b + t * centroid
            candidates.append(s)

    candidates = np.array(candidates)

    validation_score = 1.0 if len(candidates) > 0 else 0.0
    if len(candidates) >= target and target > 0:
        idx = np.random.choice(len(candidates), target, replace=False)
        return candidates[idx], validation_score

    return candidates, validation_score

# =============================================================================
# PIPELINE
# =============================================================================

def augment_dataset(df, class_col, feature_cols, method):
    X = df[feature_cols].values
    y = df[class_col].values

    augmented = []
    labels = []

    print(f"\nProcessing {len(np.unique(y))} classes for {method}...")

    for c in np.unique(y):
        print(f"\n--- Class {c} ({method}) ---")
        Xc = X[y == c]
        size = len(Xc)

        if size < 3:
            print(f"Skipping augmentation for class {c} (Size: {size} < 3)")
            augmented.append(Xc)
            labels.extend([c]*size)
            continue

        print(f"Original Class {c} Size: {size}")
        
        k_opt = find_optimal_k(Xc)
        print(f"Optimal K for Class {c}: {k_opt}")
        
        kmeans_orig = CustomKMeans(n_clusters=k_opt, random_state=42, n_init=3)
        labels_orig = kmeans_orig.fit_predict(Xc)
        
        sil_orig = custom_silhouette_score(Xc, labels_orig)
        db_orig = custom_davies_bouldin_score(Xc, labels_orig)
        print(f"Original Silhouette Score: {sil_orig:.4f}")
        print(f"Original Davies-Bouldin Score: {db_orig:.4f}")
        
        centroid = kmeans_orig.cluster_centers_.mean(axis=0)

        if method == 'bezier_convex':
            boundary = get_boundary(Xc, 'convex')
            S, validation_score = generate_bezier_convex(Xc, size, centroid, boundary)
        elif method == 'bezier_alpha':
            boundary = get_boundary(Xc, 'alpha', 0.1)
            S, validation_score = generate_bezier_alpha(Xc, size, centroid, boundary)
        elif method == 'gaussian_shell':
            try:
                hull = ConvexHull(Xc)
                rhull = np.max(np.linalg.norm(Xc[hull.vertices] - centroid, axis=1))
                boundary = Xc[hull.vertices]
            except:
                rhull = np.max(np.linalg.norm(Xc - centroid, axis=1))
                boundary = Xc
            S, validation_score = generate_gaussian_shell(Xc, size, centroid, boundary, rhull)
        elif method == 'bezier_chord':
            boundary = get_boundary(Xc, 'convex')
            S, validation_score = generate_bezier_chord(Xc, size, centroid, boundary)
        elif method == 'linear_contraction':
            boundary = get_boundary(Xc, 'convex')
            S, validation_score = generate_linear_contraction(Xc, size, centroid, boundary)
        else:
            S, validation_score = np.array([]), 0.0

        print(f"Method Validation Score (Ray Casting Inside): {validation_score:.4f}")

        augmented.extend([Xc, S])
        labels.extend([c]*size + [c]*len(S))
        print(f"Generated {len(S)} new synthetic samples for Class {c}")

        if len(S) > 0:
            Xc_aug = np.vstack((Xc, S))
            k_opt_aug = find_optimal_k(Xc_aug)
            kmeans_aug = CustomKMeans(n_clusters=k_opt_aug, random_state=42, n_init=3)
            labels_aug = kmeans_aug.fit_predict(Xc_aug)
            
            sil_aug = custom_silhouette_score(Xc_aug, labels_aug)
            db_aug = custom_davies_bouldin_score(Xc_aug, labels_aug)
            print(f"Post-Augmentation Silhouette Score: {sil_aug:.4f}")
            print(f"Post-Augmentation Davies-Bouldin Score: {db_aug:.4f}")

    X_aug = np.vstack([a for a in augmented if len(a) > 0])
    df_aug = pd.DataFrame(X_aug, columns=feature_cols)
    df_aug[class_col] = labels
    return df_aug

def dataset_info(df, class_col, feature_cols, optimal_k):
    print("\n===== DATASET INFORMATION =====")
    print("Number of features:", len(feature_cols))
    print("Number of classes:", len(df[class_col].unique()))
    print("Total samples:", len(df))
    print("Optimal K value:", optimal_k)
    print("\nSamples per class:")
    class_counts = df[class_col].value_counts().sort_index()
    for cls, count in class_counts.items():
        print(f"Class {cls} : {count}")

def main():
    import sys
    DATASET_FILE = r"E:\fyr\datasets\shuttle.trn"
    if len(sys.argv) > 1:
        DATASET_FILE = sys.argv[1]
        
    try:
        df = pd.read_csv(DATASET_FILE, header=None, sep=r'[,\s;]+', engine="python")
    except Exception as e:
        print("Failed to load dataset:", e)
        return
        
    print("Dataset shape:", df.shape)
    cols = list(df.columns)
    cols[-1] = "class"
    df.columns = cols

    feature_cols = [c for c in df.columns if c != "class"]
    methods = [
        'bezier_convex',
        'bezier_alpha',
        'gaussian_shell',
        'bezier_chord',
        'linear_contraction'
    ]

    os.makedirs("augmented_datasets", exist_ok=True)
    results = []

    X = df[feature_cols].values
    k = find_optimal_k(X)
    dataset_info(df, "class", feature_cols, k)

    kmeans = CustomKMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    sil0 = custom_silhouette_score(X, labels)
    db0 = custom_davies_bouldin_score(X, labels)

    results.append(["Original", len(df), 0, len(df), sil0, db0])

    for m in methods:
        print("\nRunning method:", m)
        try:
            aug_df = augment_dataset(df, "class", feature_cols, m)
            save_path = f"augmented_datasets/{m}_augmented.csv"
            aug_df.to_csv(save_path, index=False)
            print("Saved:", save_path)

            X_aug = aug_df[feature_cols].values
            labels = kmeans.fit_predict(X_aug)

            sil = custom_silhouette_score(X_aug, labels)
            db = custom_davies_bouldin_score(X_aug, labels)

            results.append([m, len(df), len(aug_df)-len(df), len(aug_df), sil, db])
        except Exception as e:
            print("Method failed:", m, "Error:", e)

    table = pd.DataFrame(results, columns=[
        "Method", "Original Samples", "Synthetic Samples", 
        "Total Samples", "Silhouette Score", "Davies Bouldin Index"
    ])

    print("\nRESULT TABLE\n")
    print(table.to_string(index=False))

if __name__ == "__main__":
    main()
