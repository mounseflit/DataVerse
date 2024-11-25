import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import pickle

# Load datasets
academic_df = pd.read_csv('data/academic.csv')
department_df = pd.read_csv('data/department.csv')

# Data Cleaning and Preprocessing
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
    # Convert 'Hour' column to datetime and then to numerical format
    academic_df['Hour'] = pd.to_datetime(academic_df['Hour'], format='%H:%M:%S', errors='coerce').dt.hour
    scaler = StandardScaler()
    academic_df['Hour'] = scaler.fit_transform(academic_df[['Hour']].values.reshape(-1, 1))
else:
    print("Column 'Hour' does not exist in the academic_df dataframe.")

# Feature Engineering
# Selecting key variables for clustering
features = academic_df[['SchoolDepartment', 'CourseTitle', 'RequiredSkill', 'GainedSkill', 'Hour']]

# Ensure there are no NaN values in the features
features = features.fillna(0)

# Model Training
# Test multiple algorithms (using KMeans as an example)
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(features)

# Evaluate model performance
sil_score = silhouette_score(features, clusters)
calinski_harabasz = calinski_harabasz_score(features, clusters)
davies_bouldin = davies_bouldin_score(features, clusters)

print(f'Silhouette Score: {sil_score}')
print(f'Calinski-Harabasz Index: {calinski_harabasz}')
print(f'Davies-Bouldin Index: {davies_bouldin}')

# Model Deployment
# Save the trained model
with open('kmeans_model.pkl', 'wb') as file:
    pickle.dump(kmeans, file)

# Save the label encoders and scaler
with open('label_encoders.pkl', 'wb') as file:
    pickle.dump(label_encoders, file)

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)