import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Successfully loaded {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

# Preprocess Data
def preprocess_data(df):
    df.ffill(inplace=True)

    # One-hot encode categorical features
    categorical_features = ['SchoolDepartment', 'CourseTitle', 'RequiredSkill', 'GainedSkill']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(df[categorical_features])

    # Scale numerical features
    if 'Hour' in df.columns:
        df['Hour'] = pd.to_datetime(df['Hour'], errors='coerce').dt.hour.fillna(0).astype(float)
    else:
        logging.warning("Column 'Hour' does not exist.")
        df['Hour'] = 0
    scaler = StandardScaler()
    numeric_features = scaler.fit_transform(df[['Hour']])

    # Combine features
    features = np.hstack((encoded_features, numeric_features))
    return features

# Apply PCA
def apply_pca(features, n_components=2):
    pca = PCA(n_components=n_components, random_state=42)
    reduced_features = pca.fit_transform(features)
    logging.info(f"PCA reduced features to {n_components} dimensions.")
    return reduced_features

# Evaluate Clustering
def evaluate_clustering(features, clusters):
    if len(set(clusters)) > 1:
        sil_score = silhouette_score(features, clusters)
        calinski_harabasz = calinski_harabasz_score(features, clusters)
        davies_bouldin = davies_bouldin_score(features, clusters)
        return sil_score, calinski_harabasz, davies_bouldin
    else:
        return None, None, None

# Visualize Clusters
def visualize_clusters(features, clusters, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(features[:, 0], features[:, 1], c=clusters, cmap='viridis', s=30, alpha=0.7)
    plt.title(f"Cluster Visualization - {title}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(label='Cluster')
    plt.show()

# Train and Evaluate Models
def train_and_evaluate_models(features):
    results = []

    # KMeans
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans_clusters = kmeans.fit_predict(features)
    kmeans_scores = evaluate_clustering(features, kmeans_clusters)
    if kmeans_scores[0] is not None:
        results.append(['KMeans'] + list(kmeans_scores))
        visualize_clusters(features, kmeans_clusters, "KMeans")

    # Agglomerative Clustering
    agg = AgglomerativeClustering(n_clusters=4)
    agg_clusters = agg.fit_predict(features)
    agg_scores = evaluate_clustering(features, agg_clusters)
    if agg_scores[0] is not None:
        results.append(['AgglomerativeClustering'] + list(agg_scores))
        visualize_clusters(features, agg_clusters, "Agglomerative Clustering")

    # DBSCAN
    dbscan = DBSCAN(eps=0.4, min_samples=5)
    dbscan_clusters = dbscan.fit_predict(features)
    dbscan_scores = evaluate_clustering(features, dbscan_clusters)
    if dbscan_scores[0] is not None:
        results.append(['DBSCAN'] + list(dbscan_scores))
        visualize_clusters(features, dbscan_clusters, "DBSCAN")

    # Gaussian Mixture
    gmm = GaussianMixture(n_components=4, random_state=42)
    gmm_clusters = gmm.fit_predict(features)
    gmm_scores = evaluate_clustering(features, gmm_clusters)
    if gmm_scores[0] is not None:
        results.append(['GaussianMixture'] + list(gmm_scores))
        visualize_clusters(features, gmm_clusters, "Gaussian Mixture")

    return results

# Main Function
def main():
    # Load and preprocess data
    academic_df = load_data('data/academic.csv')
    features = preprocess_data(academic_df)

    # Apply PCA for dimensionality reduction
    reduced_features = apply_pca(features, n_components=2)

    # Train and evaluate clustering algorithms
    results = train_and_evaluate_models(reduced_features)

    # Display results
    columns = ['Algorithm', 'Silhouette Score', 'Calinski-Harabasz Index', 'Davies-Bouldin Index']
    results_df = pd.DataFrame(results, columns=columns)
    print(results_df)

if __name__ == '__main__':
    main()
