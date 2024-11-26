from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (you can restrict this to specific origins)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)


# Initialize FastAPI
app = FastAPI(title="Course Recommendation API", version="1.0")

# Load Preprocessing Objects
with open("./preprocessors/label_encoder_professors.pkl", "rb") as f:
    le_professors = pickle.load(f)

with open("./preprocessors/label_encoder_school_department.pkl", "rb") as f:
    le_school_department = pickle.load(f)

with open("./preprocessors/tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("./preprocessors/pca.pkl", "rb") as f:
    pca = pickle.load(f)

# Load Trained Models
with open("./models/KMeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

# Cluster Metadata
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
