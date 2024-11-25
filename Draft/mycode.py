import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
import h5py
import os
import logging
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logging.info(f'Successfully loaded {file_path}')
        return df
    except Exception as e:
        logging.error(f'Error loading {file_path}: {e}')
        return pd.DataFrame()

def preprocess_data(academic_df, department_df):
    # Fill missing values
    academic_df.fillna('', inplace=True)
    department_df.fillna('', inplace=True)

    # Encode categorical variables
    label_encoders = {}
    for column in ['SchoolDepartment', 'CourseTitle', 'Email', 'RequiredSkill', 'Professors', 'GainedSkill']:
        le = LabelEncoder()
        academic_df[column] = le.fit_transform(academic_df[column])
        label_encoders[column] = le

    # Normalize data
    if 'Hour' in academic_df.columns:
        academic_df['Hour'] = pd.to_datetime(academic_df['Hour'], format='%H:%M:%S', errors='coerce').dt.hour
        scaler = StandardScaler()
        academic_df['Hour'] = scaler.fit_transform(academic_df[['Hour']].values.reshape(-1, 1))
    else:
        logging.warning("Column 'Hour' does not exist in the academic_df dataframe.")
        scaler = None

    return academic_df, label_encoders, scaler

def train_model(features):
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(features)
    return kmeans, clusters

def evaluate_model(features, clusters):
    sil_score = silhouette_score(features, clusters)
    calinski_harabasz = calinski_harabasz_score(features, clusters)
    davies_bouldin = davies_bouldin_score(features, clusters)

    logging.info(f'Silhouette Score: {sil_score}')
    logging.info(f'Calinski-Harabasz Index: {calinski_harabasz}')
    logging.info(f'Davies-Bouldin Index: {davies_bouldin}')

def save_model(model, label_encoders, scaler):
    try:
        os.makedirs('models', exist_ok=True)
        with h5py.File('models/kmeans_model.h5', 'w') as file:
            file.create_dataset('cluster_centers', data=model.cluster_centers_)
            file.create_dataset('labels', data=model.labels_)
        with open('models/label_encoders.pkl', 'wb') as file:
            pickle.dump(label_encoders, file)
        if scaler:
            with open('models/scaler.pkl', 'wb') as file:
                pickle.dump(scaler, file)
        logging.info('Models and encoders saved successfully.')
    except Exception as e:
        logging.error(f'Error saving models and encoders: {e}')

def main():
    # Load datasets
    academic_df = load_data(os.path.join('data', 'academic.csv'))
    department_df = load_data(os.path.join('data', 'department.csv'))

    if academic_df.empty or department_df.empty:
        logging.error('One or more datasets could not be loaded. Exiting.')
        return

    # Data Cleaning and Preprocessing
    academic_df, label_encoders, scaler = preprocess_data(academic_df, department_df)

    # Feature Engineering
    features = academic_df[['SchoolDepartment', 'CourseTitle', 'RequiredSkill', 'GainedSkill', 'Hour']]
    features = features.fillna(0)

    # Model Training
    kmeans, clusters = train_model(features)

    # Evaluate model performance
    evaluate_model(features, clusters)

    # Model Deployment
    save_model(kmeans, label_encoders, scaler)

if __name__ == '__main__':
    main()
    def train_multiple_models(features):
        models = {
            'KMeans': KMeans(n_clusters=5, random_state=42),
            'GaussianMixture': GaussianMixture(n_components=5, random_state=42),
            'AgglomerativeClustering': AgglomerativeClustering(n_clusters=5)
        }
        trained_models = {}
        for name, model in models.items():
            if name == 'GaussianMixture':
                clusters = model.fit_predict(features)
            else:
                clusters = model.fit(features).labels_
            trained_models[name] = (model, clusters)
        return trained_models

    def evaluate_multiple_models(features, trained_models):
        scores = {}
        for name, (model, clusters) in trained_models.items():
            sil_score = silhouette_score(features, clusters)
            calinski_harabasz = calinski_harabasz_score(features, clusters)
            davies_bouldin = davies_bouldin_score(features, clusters)
            scores[name] = {
                'Silhouette Score': sil_score,
                'Calinski-Harabasz Index': calinski_harabasz,
                'Davies-Bouldin Index': davies_bouldin
            }
            logging.info(f'{name} - Silhouette Score: {sil_score}')
            logging.info(f'{name} - Calinski-Harabasz Index: {calinski_harabasz}')
            logging.info(f'{name} - Davies-Bouldin Index: {davies_bouldin}')
        return scores
