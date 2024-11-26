import pandas as pd
import pickle

# Load cleaned dataset
cleaned_df = pd.read_csv('data/academic.csv')

# Load preprocessing objects
with open(r"Prepro\tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open(r"Prepro\pca.pkl", "rb") as f:
    pca = pickle.load(f)

# Load clustering model
with open(r"Notebook_models\KMeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

# Prepare data for clustering
# Combine required and gained skills
cleaned_df['skills_combined'] = cleaned_df['RequiredSkill'] + " " + cleaned_df['GainedSkill']

# Apply TF-IDF to encode skills, handling NaN values
tfidf_features = tfidf.transform(cleaned_df['skills_combined'].fillna('')).toarray()

# Apply PCA for dimensionality reduction
pca_features = pca.transform(tfidf_features)

# Apply PCA for dimensionality reduction, ensuring the number of features matches
if tfidf_features.shape[1] != pca.n_features_:
    raise ValueError(f"TF-IDF features have {tfidf_features.shape[1]} features, but PCA expects {pca.n_features_} features.")
pca_features = pca.transform(tfidf_features)

# Generate cluster metadata
cluster_metadata = {}

for cluster_id in cleaned_df['cluster'].unique():
    cluster_data = cleaned_df[cleaned_df['cluster'] == cluster_id]
    
    # Most common department
    department = cluster_data['school_department'].mode()[0]
    
    # Most common professor
    professor = cluster_data['professors'].mode()[0]
    
    # Top skills
    skills = (
        cluster_data['skills_combined']
        .str.split()
        .explode()
        .value_counts()
        .head(5)
        .index.tolist()
    )
    
    # Store in metadata dictionary
    cluster_metadata[cluster_id] = {
        'department': department,
        'professor': professor,
        'skills': ', '.join(skills)
    }

# Save the metadata
with open("cluster_metadata.pkl", "wb") as f:
    pickle.dump(cluster_metadata, f)

print("Cluster metadata generated and saved successfully.")
