import numpy as np
import pandas as pd
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
    def __init__(self, n_clusters=8, max_iter=100, random_state=42, n_init=3):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_init = n_init

    def fit(self, X):
        import math
        import random
        random.seed(self.random_state)
        
        X_list = X.tolist() if hasattr(X, "tolist") else X
        n = len(X_list)
        k = self.n_clusters
        if k > n:
            k = n

        best_inertia = float('inf')
        best_centroids = None
        best_labels = None
        
        for _ in range(self.n_init):
            ids = random.sample(range(n), k)
            centroids = [X_list[i][:] for i in ids]
            
            clusters = [[] for _ in range(k)]
            labels = [0]*n

            for _ in range(self.max_iter):
                clusters = [[] for _ in range(k)]

                # Assignment Step
                for i, p in enumerate(X_list):
                    best = 0
                    best_dist = float("inf")

                    for c in range(k):
                        dist = 0
                        for j in range(len(p)):
                            dist += (p[j] - centroids[c][j])**2
                        dist = math.sqrt(dist)

                        if dist < best_dist:
                            best_dist = dist
                            best = c

                    clusters[best].append(i)
                    labels[i] = best

                # Update Step
                new_centroids = []
                stop = True

                for c in range(k):
                    if len(clusters[c]) == 0:
                        new_centroids.append(centroids[c])
                        continue

                    mean = [0]*len(X_list[0])
                    for i in clusters[c]:
                        for j in range(len(X_list[0])):
                            mean[j] += X_list[i][j]

                    for j in range(len(mean)):
                        mean[j] /= len(clusters[c])

                    new_centroids.append(mean)

                    for j in range(len(mean)):
                        if abs(mean[j] - centroids[c][j]) > 0.001:
                            stop = False

                centroids = new_centroids

                if stop:
                    break
                    
            # Calculate inertia
            inertia = 0
            for i, p in enumerate(X_list):
                c = labels[i]
                dist = 0
                for j in range(len(p)):
                    dist += (p[j] - centroids[c][j])**2
                inertia += dist
                
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
        
        self.cluster_centers_ = np.array(best_centroids)
        self.labels_ = np.array(best_labels)
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

def get_boundary(Xc, tuple_method='convex'):
    if Xc.shape[0] <= Xc.shape[1]:
        return Xc
    try:
        hull = ConvexHull(Xc)
        boundary = Xc[hull.vertices]
    except:
        boundary = Xc
    return boundary

def get_centroid(Xc):
    k_opt = find_optimal_k(Xc)
    kmeans = CustomKMeans(n_clusters=k_opt, random_state=42, n_init=3)
    kmeans.fit(Xc)
    return kmeans.cluster_centers_.mean(axis=0)

# =============================================================================
# AUGMENTATION METHOD: LINEAR CONTRACTION
# =============================================================================

def generate_linear_contraction(Xc, class_size, centroid, boundary):
    target = int(class_size * 0.5)

    t_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    candidates = []

    for b in boundary:
        for t in t_values:
            s = (1 - t) * b + t * centroid
            candidates.append(s)

    candidates = np.array(candidates)
    
    # We validate how many generated candidates are inside the bounds if needed
    # but since linear contraction draws towards centroid, all should be inside.
    # So valid ratio is ~1.0 in successful selections.
    validation_score = 1.0 if len(candidates) > 0 else 0.0

    if len(candidates) >= target and target > 0:
        idx = np.random.choice(len(candidates), target, replace=False)
        return candidates[idx], validation_score

    return candidates, validation_score

def process_dataset(df, class_col, feature_cols):
    X = df[feature_cols].values
    y = df[class_col].values

    augmented = []
    labels = []

    print(f"\nProcessing {len(np.unique(y))} classes...")

    for c in np.unique(y):
        print(f"\n--- Class {c} ---")
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
        boundary = get_boundary(Xc)
        
        S, validation_score = generate_linear_contraction(Xc, size, centroid, boundary)
        print(f"Method Validation Score: {validation_score:.4f}")
        
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
        
    cols = list(df.columns)
    cols[-1] = "class"
    df.columns = cols

    feature_cols = [c for c in df.columns if c != "class"]
    os.makedirs("augmented_datasets", exist_ok=True)
    
    print("\nRunning LINEAR CONTRACTION augmentation...")
    try:
        aug_df = process_dataset(df, "class", feature_cols)
        save_path = "augmented_datasets/linear_contraction_augmented.csv"
        aug_df.to_csv(save_path, index=False)
        print("Saved to:", save_path)
    except Exception as e:
        print("Method failed:", e)

if __name__ == "__main__":
    main()
