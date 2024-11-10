Hybrid Distance-based Methods for
Fraud Detection

This project aims to detect fraudulent credit card transactions using the k-Nearest Neighbors (k-NN) algorithm. Due to the imbalanced nature of fraud detection datasets, we applied various techniques for data preprocessing, feature reduction, and model evaluation.

## Project Overview

Credit card fraud detection is a critical application for protecting users and financial institutions from malicious activities. This project involves:

- Data preprocessing and normalization
- Visualization of data distribution for fraud and non-fraud classes
- Dimensionality reduction using PCA
- Training and optimizing a k-Nearest Neighbors classifier
- Model evaluation and performance metrics

## Dataset

The dataset used in this project is the **Credit Card Fraud Detection** dataset from Kaggle. This dataset contains transaction data where each entry is classified as either fraud (Class=1) or non-fraud (Class=0). 

Key columns:
- **V1-V28**: Anonymized principal components from PCA.
- **Amount**: Transaction amount.
- **Class**: Target variable (0 = non-fraud, 1 = fraud).

*Note: The original dataset was large, so we used a 5% sample for faster processing in this implementation.*

## Project Structure

### 1. Data Loading and Preprocessing
- Loaded the dataset and checked for missing values.
- Standardized the `Amount` column using StandardScaler.
- Dropped irrelevant columns to keep the feature set concise.

### 2. Exploratory Data Analysis (EDA)
- Visualized class distribution (fraud vs. non-fraud).
- Analyzed distributions of various features using histograms.
- Plotted correlation heatmaps to understand feature relationships.

### 3. Dimensionality Reduction
- Applied PCA to reduce the dataset to two principal components, allowing us to visualize data clusters.
- Performed 2D visualization to observe the separation of fraud and non-fraud cases.

### 4. Subsampling and Balancing
- Due to class imbalance, we sampled an equal number of fraud and non-fraud cases to create a balanced dataset.
- Shuffled the data before splitting into training and test sets.

### 5. Model Training and Optimization
- Used k-Nearest Neighbors (k-NN) as the classifier.
- Evaluated different values of k (neighbors) to optimize the classifier.
- Selected the optimal k based on highest test accuracy and trained the final model.

### 6. Model Evaluation
- Used metrics such as accuracy, precision, recall, F1 score, and Matthews Correlation Coefficient (MCC).
- Plotted the ROC curve and calculated the AUC score for model assessment.

### 7. Saving and Loading the Model
- Saved the trained model to disk for reuse.
- Demonstrated how to reload the model and use it for future predictions.

## Key Results

- **Optimal k Value**: The optimal value for k was selected based on test accuracy.
- **Evaluation Metrics**: Accuracy, precision, recall, F1 score, and MCC were computed to evaluate model performance.
- **ROC Curve and AUC**: The ROC curve showed the model’s ability to distinguish between classes, with the AUC score reflecting classification performance.

## Requirements

The project uses the following Python libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib
