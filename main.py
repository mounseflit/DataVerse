import pandas as pd
import numpy as np
from datetime import datetime
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Load datasets
academic_df = pd.read_csv("./data/academic.csv")
academic_df.head()

department_df = pd.read_csv("./data/department.csv")
department_df.head()

# Clean and Standardize column names
academic_df.columns = academic_df.columns.str.lower().str.replace(' ', '_')
department_df.columns = department_df.columns.str.lower().str.replace(' ', '_')

academic_df.head()

# Since we got a non understandable column name we rename some specific columns
academic_df.rename(columns={'schooldepartment': 'school_department'}, inplace=True)
academic_df.rename(columns={'coursetitle': 'course_title'}, inplace=True)
academic_df.rename(columns={'requiredskill': 'required_skill'}, inplace=True)
academic_df.rename(columns={'gainedskill': 'gained_skill'}, inplace=True)
academic_df.rename(columns={'strartingorasgnmtdate': 'starting_or_assignment_date'}, inplace=True)

academic_df.head()

department_df.head()

# Handle Missing Values
# Let's check first how much missed values are in our dataframes
print("\nMissing values in Academic Dataset:")
print(academic_df.isnull().sum())
print("\nMissing values in Department Dataset:")
print(department_df.isnull().sum())


# Impute missing 'hour' column in academic_df
academic_df['hour'] = academic_df['hour'].fillna('00:00:00')

# Fill missing values in other columns by 'unknown'
academic_df.fillna('unknown', inplace=True)
academic_df.fillna('unknown', inplace=True)

print("\nMissing values in Academic Dataset:")
print(academic_df.isnull().sum())
print("\nMissing values in Department Dataset:")
print(department_df.isnull().sum())
# As we can see we removed successfully all NaN values

# Standardize department names by making everything in lower case
academic_df['school_department'] = academic_df['school_department'].str.lower().str.strip()
department_df['name'] = department_df['name'].str.lower().str.strip()

# Handle special characters in department name like that & become "and"
academic_df['school_department'] = academic_df['school_department'].str.replace('&', 'and')
academic_df[:1]

academic_df

# Merge datasets on a common column "school_department"
merged_df = pd.merge(academic_df, department_df, left_on='school_department', right_on='name', how='inner')
merged_df.head()

# Combine required and gained skills
merged_df['skills_combined'] = merged_df['required_skill'] + " " + merged_df['gained_skill']

# Encode text features using TF-IDF
tfidf = TfidfVectorizer(max_features=100)
skills_tfidf = tfidf.fit_transform(merged_df['skills_combined']).toarray()

# Add TF-IDF features back to the dataframe
skills_df = pd.DataFrame(skills_tfidf, columns=[f'skill_tfidf_{i}' for i in range(skills_tfidf.shape[1])])
merged_df = pd.concat([merged_df.reset_index(drop=True), skills_df], axis=1)

# Encode categorical columns
le = LabelEncoder()
merged_df['school_department_encoded'] = le.fit_transform(merged_df['school_department'])
merged_df['professors_encoded'] = le.fit_transform(merged_df['professors'])

# Select numerical features for clustering
numerical_features = merged_df.select_dtypes(include=['float64', 'int64']).dropna(axis=1)

# Reduce dimensionality for visualization
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(numerical_features)

# List of clustering algorithms
clustering_algorithms = {
    "KMeans": KMeans(n_clusters=5, random_state=42),
    "AgglomerativeClustering": AgglomerativeClustering(n_clusters=5),
    "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
    "GaussianMixture": GaussianMixture(n_components=5, random_state=42)
}

# Dictionary to store evaluation metrics
results = {}

# Apply clustering algorithms, evaluate, and save models
for name, model in clustering_algorithms.items():
    # Fit or predict clusters
    if name == "GaussianMixture":
        model.fit(numerical_features)
        clusters = model.predict(numerical_features)
    else:
        clusters = model.fit_predict(numerical_features)

    # Add cluster labels to DataFrame
    merged_df[f"{name}_cluster"] = clusters

    # Compute evaluation metrics
    silhouette_avg = silhouette_score(numerical_features, clusters) if len(set(clusters)) > 1 else np.nan
    calinski_harabasz = calinski_harabasz_score(numerical_features, clusters) if len(set(clusters)) > 1 else np.nan
    davies_bouldin = davies_bouldin_score(numerical_features, clusters) if len(set(clusters)) > 1 else np.nan

    # Store metrics in results
    results[name] = {
        "Silhouette Score": silhouette_avg,
        "Calinski-Harabasz Index": calinski_harabasz,
        "Davies-Bouldin Index": davies_bouldin
    }

    # Save model
    with open(f"notebook_models/{name}_model.pkl", "wb") as file:
        pickle.dump(model, file)

    # Plot clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(
        reduced_features[:, 0],
        reduced_features[:, 1],
        c=clusters,
        cmap="viridis",
        s=10
    )
    plt.title(f"{name} Clustering")
    plt.xlabel("PCA Feature 1")
    plt.ylabel("PCA Feature 2")
    plt.colorbar(label="Cluster")
    plt.show()

# Print evaluation results
print("Clustering Evaluation Metrics:\n")
for name, metrics in results.items():
    print(f"{name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    print()
