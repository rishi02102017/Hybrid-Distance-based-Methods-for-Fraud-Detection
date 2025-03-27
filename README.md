# 🛡️ Hybrid Distance-Based Methods for Fraud Detection

A smart approach to credit card fraud detection using **k-Nearest Neighbors (k-NN)** and **dimensionality reduction techniques** like **PCA** to enhance model performance on **imbalanced datasets**.

---

## 📌 Overview

Credit card fraud is a serious threat in today’s digital economy. This project builds a **hybrid distance-based fraud detection system** that incorporates:

- 📊 Data preprocessing & normalization  
- 🔍 Exploratory data analysis (EDA)  
- 🔻 Dimensionality reduction using PCA  
- 🤖 Training an optimized k-NN classifier  
- 📈 Performance evaluation using robust metrics  

---

## 📂 Dataset

- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
- Features:  
  - `V1-V28`: Anonymized PCA-transformed features  
  - `Amount`: Transaction value  
  - `Class`: Target (0 = non-fraud, 1 = fraud)  
- A **5% sample** was used for experimentation to ensure faster processing.

---

## 🧭 Project Pipeline

### 1️⃣ Data Loading & Preprocessing
- Checked for null values  
- Scaled `Amount` using `StandardScaler`  
- Removed irrelevant columns

### 2️⃣ Exploratory Data Analysis (EDA)
- Visualized class imbalance  
- Analyzed feature distributions  
- Correlation heatmap analysis

### 3️⃣ Dimensionality Reduction
- Applied PCA to reduce dimensionality  
- Visualized fraud vs. non-fraud clusters in 2D space

### 4️⃣ Sampling & Balancing
- Created a balanced dataset using undersampling  
- Ensured even class distribution for training

### 5️⃣ Model Training & Optimization
- Trained k-NN with various `k` values  
- Selected best `k` based on test set accuracy

### 6️⃣ Evaluation Metrics
- **Accuracy**, **Precision**, **Recall**, **F1 Score**, **Matthews Correlation Coefficient (MCC)**  
- Plotted **ROC curve** and calculated **AUC**

### 7️⃣ Model Persistence
- Saved final model using `joblib`  
- Demonstrated model reloading and prediction on new data

---

## 🚀 Key Results

✅ Optimal `k` selected through evaluation on test set  
✅ High performance on key metrics: Precision, Recall, MCC  
✅ ROC-AUC curve validated strong model performance

---

## 🧰 Tech Stack

| Tool            | Purpose                         |
|-----------------|---------------------------------|
| `pandas`        | Data loading & manipulation     |
| `numpy`         | Numerical computations          |
| `matplotlib`, `seaborn` | Visualization          |
| `scikit-learn`  | ML modeling, PCA, evaluation    |
| `joblib`        | Model serialization             |

---

## 📁 Files Included

- `ML_Project.ipynb` – Full implementation and walkthrough  
- `model.pkl` – Serialized k-NN model  
- `README.md` – Project documentation  

---

## ✨ Author
- **Jyotishman Das**  
- M.Tech AI @ IIT Jodhpur | AI/ML Engineer  
- [LinkedIn](https://www.linkedin.com/in/jyotishmandas85p)
- [Website](https://my-portfolio-jyotishman-das-projects.vercel.app/)


---
