# Academic Resource Recommendation System  
**Welcome to the "Open Source and Dataverse" Hackathon!**  

This document provides an overview of the Academic Resource Recommendation System project. It details the objectives, features, and implementation phases for the system, as well as how participants can contribute to its development.

---

## 📖 **Project Overview**  

In schools and universities, it is often challenging to recommend academic resources due to the lack of alignment between department courses, required skills, gained skills, and professors' expertise. This project aims to address this issue by developing an innovative web application powered by Machine Learning to provide personalized academic resource recommendations.  

**Key Objectives:**  
- Build a cluster-based recommendation system using Machine Learning.  
- Develop advanced resource management and statistical analysis features to enhance decision-making for administrators.  
- Ensure optimal usability and security within a web application framework.  

---

## 🎯 **Main Features**  

1. **Machine Learning Recommendation System**  
   - Develop a clustering model to recommend academic resources based on student needs.  
   - Evaluate models using performance metrics like Silhouette Score, Calinski-Harabasz Index, and Davies-Bouldin Index.  

2. **Backend Functionality**  
   - Implement RESTful APIs for CRUD operations on courses and professors.  
   - Enable Machine Learning predictions via API endpoints.  
   - Provide statistical insights through dynamic visualizations.  

3. **Frontend Interface**  
   - Admin dashboard for resource management and statistical visualization.  
   - Professor dashboard for course-related notifications and updates.  
   - Authentication and secure user management.  

---

## 🛠️ **Project Structure**  

### **1. Machine Learning (Cluster-Based Recommendation System)**  
- **Data Preparation**:  
  - Clean and preprocess data for consistency and usability.  
  - Encode categorical variables and engineer features for optimal performance.  

- **Modeling**:  
  - Train clustering algorithms and evaluate using Silhouette Score, Calinski-Harabasz Index, and Davies-Bouldin Index.  
  - Export the trained model in `.pkl` or `.h5` format for deployment.  

- **Deployment**:  
  - Integrate the model into the backend via RESTful APIs for real-time predictions.  

---

### **2. Backend Development**  
- **API Endpoints**:  
  - CRUD operations for managing courses and professors.  
  - Predictions for course details based on input data.  

- **Statistical Analysis**:  
  - Visualizations for metrics like course completion rates, required vs. gained skills, and professor workloads.  

- **Security**:  
  - Implement token-based authentication and password encryption using `bcrypt`.  
  - Notification service for real-time updates to professors.  

---

### **3. Frontend Development**  
- **User Interfaces**:  
  - **Home Page**: Introduction to the platform’s objectives.  
  - **Admin Dashboard**: Statistics visualization, resource management, and testing the recommendation system.  
  - **Professor Dashboard**: Notifications and assigned course details.  

- **Authentication**:  
  - Secure login page for admins and professors.  

---

## 📅 **Hackathon Schedule**  

| **Phase**                          | **Start Time**         | **Review Time**        |  
|------------------------------------|------------------------|------------------------|  
| Machine Learning & Database Setup | 11:00 AM, 25/11        | 10:00 PM, 25/11        |  
| Backend Development                | 11:00 PM, 25/11        | 9:00 AM, 26/11         |  
| Frontend Development               | 9:00 AM, 26/11         | 4:00 PM, 26/11         |  
| Documentation & Presentation       | 4:30 PM, 26/11         |                        |  

---

## 📌 **Requirements**  

### **Target Participants**  
- Students, professionals, and enthusiasts in web development and data science.  

### **Required Skills**  
- Full-stack web development.  
- Artificial Intelligence modeling.  
- Database management.  

---

## 🌟 **Project Added Value**  

- Personalized recommendations improve students' learning experiences.  
- Optimized resource allocation reduces waste.  
- Statistical insights enhance decision-making for administrators.  
- Improved alignment of required and gained skills for academic success.  

---

## 🚀 **Getting Started**  

### **1. Prerequisites**  
- Install Python 3.9 or higher.  
- Install required packages: `pandas`, `numpy`, `scikit-learn`, `flask`, `bcrypt`, etc.  


### **2. Setting Up the Environment**  

1. **Clone the Repository**  
   Clone the project repository from the provided URL:  
   ```bash
   git clone <repository_url>
   cd academic-resource-recommendation-system
   ```

2. **Install Dependencies**  
   Install the required dependencies using the `requirements.txt` file:  
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up the Database**  
   Restore the database using the provided dataset to ensure proper structure and relationships:  
   - Import the dataset into your database management system.  
   - Ensure all tables and relationships are correctly implemented.  
   - Run migrations if necessary.

4. **Environment Variables**  
   Create a `.env` file in the root directory with the following configurations:  
   ```env
   DATABASE_URL=<your_database_connection_string>
   SECRET_KEY=<your_secret_key>
   ```

---

### **3. Running the Application**  

1. **Start the Backend Server**  
   Run the backend Flask application:  
   ```bash
   python app.py
   ```

   The server should now be running at `http://localhost:5000`.  

2. **Start the Frontend**  
   If using a React or Vue.js framework, navigate to the frontend directory and install dependencies:  
   ```bash
   npm install
   npm start
   ```

   The frontend will typically run on `http://localhost:3000`.  

3. **Access the Application**  
   Open your browser and navigate to the appropriate URLs:  
   - Backend API: `http://localhost:5000/api`
   - Frontend Interface: `http://localhost:3000`  

---

### **4. Testing the Application**  

1. **Machine Learning Models**  
   - Use Jupyter Notebook or Python scripts to validate the Machine Learning model performance.  
   - Evaluate the clustering model using metrics like Silhouette Score and Calinski-Harabasz Index.  

2. **Backend Endpoints**  
   - Test RESTful APIs using tools like Postman or curl.  
   - Ensure all endpoints (CRUD operations, prediction, and statistics) are functioning as expected.  

3. **Frontend User Flow**  
   - Test user interfaces for both admin and professor roles.  
   - Verify the login functionality and secure token-based authentication.  

4. **Bug Reporting**  
   Document any bugs or issues encountered during testing and submit them to the development team.  

---

### **5. Deployment**  

1. **Backend Deployment**  
   - Use a hosting platform such as AWS, Heroku, or DigitalOcean.  
   - Ensure the database is connected and secure.  

2. **Frontend Deployment**  
   - Use a hosting platform like Vercel or Netlify.  
   - Connect the frontend to the backend API.

3. **Environment Variables**  
   - Configure environment variables for production in your hosting platform.  

4. **Testing in Production**  
   - Conduct end-to-end testing to ensure the application works seamlessly in production.  

---

## 🧪 **Evaluation Metrics**  

1. **Machine Learning Model**  
   - Accuracy and performance of the clustering model.  
   - Scores based on Silhouette, Calinski-Harabasz, and Davies-Bouldin metrics.  

2. **Backend APIs**  
   - Completeness and efficiency of API endpoints.  
   - Secure implementation of authentication and data management.  

3. **Frontend Usability**  
   - Intuitive user interface and responsiveness.  
   - Proper visualization of statistics and recommendations.  

4. **Documentation**  
   - Clarity and comprehensiveness of the README and project documentation.  

---

## 🎉 **Next Steps**  

- Collaborate with your team to finalize the implementation.  
- Review the judging criteria to ensure your project meets all requirements.  
- Prepare your presentation to demonstrate your solution to the judges.  

**Good luck, and let’s build something amazing together!** 🚀

