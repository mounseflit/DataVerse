import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import os
import logging
import pickle

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
def preprocess_data(academic_df):
    # Fill missing values with forward fill
    academic_df.ffill(inplace=True)

    # One-hot encode categorical features
    categorical_features = ['SchoolDepartment', 'CourseTitle', 'RequiredSkill', 'GainedSkill']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(academic_df[categorical_features])

    # Handle 'Hour' column if it exists
    if 'Hour' in academic_df.columns:
        try:
            academic_df['Hour'] = pd.to_datetime(academic_df['Hour'], format='%H:%M:%S', errors='coerce').dt.hour
            academic_df['Hour'] = academic_df['Hour'].fillna(0).astype(float)
        except Exception as e:
            logging.warning(f"Failed to parse 'Hour' column: {e}")
            academic_df['Hour'] = 0
    else:
        logging.warning("Column 'Hour' does not exist.")
        academic_df['Hour'] = 0

    # Scale numerical features
    scaler = StandardScaler()
    numeric_features = scaler.fit_transform(academic_df[['Hour']])

    # Combine encoded and numerical features
    features = np.hstack((encoded_features, numeric_features))
    return features, encoder, scaler


# PCA for Dimensionality Reduction
def preprocess_with_pca(features, n_components=2):
    pca = PCA(n_components=n_components, random_state=42)
    reduced_features = pca.fit_transform(features)
    logging.info(f"PCA reduced features to {n_components} dimensions.")
    return reduced_features


# Optimize KMeans
def optimize_kmeans(features):
    best_k = 0
    best_score = -1
    for k in range(2, 11):  # Try cluster sizes from 2 to 10
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(features)
        score = silhouette_score(features, clusters)
        if score > best_score:
            best_k = k
            best_score = score
    logging.info(f"Optimal number of clusters: {best_k} with Silhouette Score: {best_score}")
    return best_k


# Train DBSCAN with Parameter Tuning
def train_dbscan(features):
    best_eps = 0.1
    best_score = -1
    best_model = None
    for eps in np.arange(0.1, 1.0, 0.1):  # Test a range of `eps` values
        dbscan = DBSCAN(eps=eps, min_samples=5)
        clusters = dbscan.fit_predict(features)
        if len(set(clusters)) > 1:  # Ensure at least two clusters
            score = silhouette_score(features, clusters)
            if score > best_score:
                best_eps = eps
                best_score = score
                best_model = dbscan
    if best_model is None:
        logging.warning("DBSCAN did not produce valid clusters.")
        return None, None
    logging.info(f"DBSCAN optimal eps: {best_eps} with Silhouette Score: {best_score}")
    return best_model, best_model.fit_predict(features)


# Evaluate Models
def evaluate_models(features, models):
    scores = {}
    for name, (model, clusters) in models.items():
        if clusters is None or len(set(clusters)) <= 1:
            logging.warning(f"{name} did not create valid clusters.")
            continue
        sil_score = silhouette_score(features, clusters)
        calinski_harabasz = calinski_harabasz_score(features, clusters)
        davies_bouldin = davies_bouldin_score(features, clusters)
        scores[name] = {
            'Silhouette Score': sil_score,
            'Calinski-Harabasz Index': calinski_harabasz,
            'Davies-Bouldin Index': davies_bouldin
        }
        logging.info(f"{name} - Silhouette Score: {sil_score}")
        logging.info(f"{name} - Calinski-Harabasz Index: {calinski_harabasz}")
        logging.info(f"{name} - Davies-Bouldin Index: {davies_bouldin}")
    return scores


# Save Models
def save_models(models, encoder, scaler):
    try:
        os.makedirs('models', exist_ok=True)
        for name, (model, _) in models.items():
            if model is not None:
                with open(f"models/{name}_model.pkl", 'wb') as file:
                    pickle.dump(model, file)
        with open("models/encoder.pkl", 'wb') as file:
            pickle.dump(encoder, file)
        with open("models/scaler.pkl", 'wb') as file:
            pickle.dump(scaler, file)
        logging.info("Models and encoders saved successfully.")
    except Exception as e:
        logging.error(f"Error saving models and encoders: {e}")


# Main Function
def main():
    # Load datasets
    academic_df = load_data(os.path.join('data', 'academic.csv'))
    department_df = load_data(os.path.join('data', 'department.csv'))

    if academic_df.empty or department_df.empty:
        logging.error("One or more datasets could not be loaded. Exiting.")
        return

    # Preprocessing
    features, encoder, scaler = preprocess_data(academic_df)

    # Apply PCA
    reduced_features = preprocess_with_pca(features, n_components=2)

    # Optimize KMeans
    optimal_k = optimize_kmeans(reduced_features)
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans_clusters = kmeans.fit_predict(reduced_features)

    # Train DBSCAN with optimal parameters
    dbscan_model, dbscan_clusters = train_dbscan(reduced_features)

    # Evaluate models
    models = {
        "KMeans": (kmeans, kmeans_clusters),
        "DBSCAN": (dbscan_model, dbscan_clusters)
    }
    scores = evaluate_models(reduced_features, models)

    # Save models and encoders
    save_models(models, encoder, scaler)


if __name__ == '__main__':
    main()
