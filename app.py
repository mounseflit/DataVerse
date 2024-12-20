from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# Initialize FastAPI
app = FastAPI(title="Course Recommendation API", version="1.0")

# Load Preprocessing Objects
with open(r"Prepro\tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open(r"Prepro\pca.pkl", "rb") as f:
    pca = pickle.load(f)

# Load KMeans Model
with open(r"Notebook_models\KMeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

# Integrated cleaned cluster metadata
cluster_metadata = {
    0: {
        "Department": "Social Impact Engineering",
        "Recommended Professor": "Samira Alaoui",
        "Required Skills": "Brochures, Communication Planning/Strategy, Data Collection Tools, Event Coordination, Graphic Design",
        "Gained Skills": "Data Collection Tools, Event Coordination, Graphic Design"
    },
    1: {
        "Department": "Academic Affairs and Staff Development",
        "Recommended Professor": "Samira Alaoui",
        "Required Skills": "Brochures, Communication Planning/Strategy, Data Collection Tools, Event Coordination, Graphic Design",
        "Gained Skills": "Data Collection Tools, Event Coordination, Graphic Design"
    },
    2: {
        "Department": "Social Impact Engineering",
        "Recommended Professor": "Mohammed Fassi",
        "Required Skills": "Brochures, Communication Planning/Strategy, Data Collection Tools, Event Coordination, Graphic Design",
        "Gained Skills": "Data Collection Tools, Event Coordination, Graphic Design"
    },
    3: {
        "Department": "Social Impact Engineering",
        "Recommended Professor": "Samira Alaoui",
        "Required Skills": "Brochures, Communication Planning/Strategy, Data Collection Tools, Event Coordination, Graphic Design",
        "Gained Skills": "Data Collection Tools, Event Coordination, Graphic Design"
    },
    4: {
        "Department": "Social Impact Engineering",
        "Recommended Professor": "Mohammed Fassi",
        "Required Skills": "Brochures, Communication Planning/Strategy, Data Collection Tools, Event Coordination, Graphic Design",
        "Gained Skills": "Data Collection Tools, Graphic Design, Internal and External Communication"
    }
}

@app.post("/predict/")
def predict_course_details(course_title: str):
    """
    Predict course details based on course title.
    """
    try:
        # Preprocess input using TF-IDF
        tfidf_features = tfidf.transform([course_title]).toarray()

        # Dimensionality reduction using PCA
        pca_features = pca.transform(tfidf_features)

        # Predict cluster using KMeans
        cluster = kmeans.predict(pca_features)[0]

        # Retrieve metadata for the cluster
        if cluster in cluster_metadata:
            recommendations = cluster_metadata[cluster]
        else:
            recommendations = {
                "Department": "Unknown",
                "Recommended Professor": "Unknown",
                "Required Skills": "Unknown",
                "Gained Skills": "Unknown"
            }

        return {
            "course_title": course_title,
            "predicted_cluster": int(cluster),
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_field/")
def predict_field(course_title: str, field: str):
    """
    Predict specific fields (e.g., department, professor) based on course title.
    """
    try:
        # Preprocess input using TF-IDF
        tfidf_features = tfidf.transform([course_title]).toarray()

        # Dimensionality reduction using PCA
        pca_features = pca.transform(tfidf_features)

        # Predict cluster using KMeans
        cluster = kmeans.predict(pca_features)[0]

        # Retrieve the requested field
        if cluster in cluster_metadata and field in cluster_metadata[cluster]:
            return {
                "course_title": course_title,
                "field_requested": field,
                "field_value": cluster_metadata[cluster][field]
            }
        else:
            return {
                "course_title": course_title,
                "field_requested": field,
                "field_value": "Unknown"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
