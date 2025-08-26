🌧️ Rainfall Prediction using Machine Learning

This project applies multiple Machine Learning algorithms to predict rainfall based on weather parameters like Temperature, Humidity, Wind Speed, and Pressure.
It compares the performance of different ML models and also provides data visualization to better understand feature correlations and distributions.

📌 Features

Loads a Rainfall dataset (CSV) or falls back to a sample dataset if not found.
->Preprocessing:
Handles missing values
Encodes categorical target variable (rainfall: Yes/No → 1/0)
Scales features using StandardScaler

->Exploratory Data Analysis (EDA):
Correlation heatmap
Feature distributions
Pairplots for feature relationships

->Machine Learning Models:
Logistic Regression
Random Forest Classifier
Support Vector Machine (SVM)
Gradient Boosting Classifier
K-Means Clustering (unsupervised)

->Model performance comparison with bar chart visualization.

📂 Project Structure
├── Rainfall .csv         # Dataset (if available)
├── rainfall_prediction.py # Main script
├── README.md             # Project documentation

⚙️ Installation

Clone the repository and install the required dependencies:

git clone https://github.com/your-username/rainfall-prediction.git
cd rainfall-prediction
pip install -r requirements.txt


requirements.txt should include:

pandas
numpy
matplotlib
seaborn
scikit-learn

▶️ Usage

->Run the script:
  python rainfall_prediction.py


->If Rainfall .csv is present, it will load the dataset.
->Otherwise, it will use a sample dataset.
->The script will:
  Show heatmap of correlations
  Train and evaluate multiple ML models
  Plot accuracy comparison bar chart
  Display feature distribution histograms and pairplots

📊 Results

->Example visualization outputs:
Feature Correlation Heatmap
Model Accuracy Comparison
Feature Distributions
Pairwise Feature Comparison
Among models tested, ensemble methods like Random Forest and Gradient Boosting generally perform better for classification tasks like rainfall prediction.

🚀 Future Improvements

Use a larger real-world rainfall dataset
Hyperparameter tuning with GridSearchCV/RandomizedSearchCV
Add more ML models (XGBoost, Neural Networks)
Deploy as a Flask/Django API or Streamlit dashboard
