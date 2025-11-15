# img_classification
Designed to assess understanding of image classification using machine learning techniques. Building a model that can accurately classify images into different categories using Random Forest. The project also involves data preprocessing, model training, result interpretation, and a conceptual discussion on deploying the model.

🧠 Image Classification Using Random Forest and SVM

📋 Overview

This project demonstrates image classification using two classical machine learning algorithms: Random Forest and Support Vector Machine (SVM).
The goal is to classify images into five categories — dalmatian, dollar_bill, pizza, soccer_ball, and sunflower.

All images were resized to 64×64 pixels, flattened into one-dimensional vectors, and normalized to improve model performance.
Both models were trained and evaluated using scikit-learn, with GridSearchCV applied for hyperparameter tuning.
🧰 Methodology

Dataset Preparation

Dataset stored in folder structure:

images/
  dalmatian/
  dollar_bill/
  pizza/
  soccer_ball/
  sunflower/


Each subfolder represents a class label.

Images are resized to 64×64 and pixel values are normalized to [0, 1].

Model Training

Random Forest: Tuned using parameters like n_estimators, max_depth, min_samples_split, and min_samples_leaf.

SVM: Implemented using a pipeline with StandardScaler, tested across kernels (linear, rbf) and regularization values (C).

Evaluation Metrics

Accuracy, Precision, Recall, and F1-score.

Confusion matrices and feature importance plots were generated for visualization.

Results Summary

Model	Accuracy	Precision	Recall	F1-score
Random Forest	0.8065	0.8419	0.8065	0.8039
SVM	0.8710	0.8748	0.8710	0.8698

The SVM model achieved the highest accuracy and balanced performance across classes.

🧪 Visual Results

Random Forest Confusion Matrix

Top 30 Feature Importances

SVM Confusion Matrix

(Plots are included in the accompanying Jupyter notebook and report.)

🚀 How to Run the Notebook

Open in Google Colab

Upload the Jupyter notebook (Image_Classification_RF_SVM.ipynb) to your Google Drive.

Open it with Google Colab.

Mount Google Drive

from google.colab import drive
drive.mount('/content/drive')


Set the Dataset Path
Update the path to match your image directory:

data_dir = "/content/drive/My Drive/images"


Run All Cells
Execute the notebook sequentially.
The notebook will:

Preprocess the dataset

Train both Random Forest and SVM

Display evaluation metrics and visualizations

Predict a New Image (Optional)

classify_new_image("/content/drive/My Drive/sample.jpg", best_rf)

🛠️ Dependencies

Ensure the following libraries are installed (Colab installs them automatically):

pip install scikit-learn pillow matplotlib seaborn

🧩 Files Included

Image_Classification_RF_SVM.ipynb — main notebook with full workflow

Image_Classification_Report.docx — 2-page analysis report

README.md — project overview and instructions

images/ — dataset organized by class labels

📦 Deployment Notes

Both models can be exported using joblib or pickle and loaded in production for inference.
For real-world applications, integrating with a simple FastAPI or Flask service will allow REST-based image classification.
