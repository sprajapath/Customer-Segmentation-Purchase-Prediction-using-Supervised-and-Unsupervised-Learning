# Customer-Segmentation-Purchase-Prediction-using-Supervised-and-Unsupervised-Learning

# Customer Segmentation + Purchase Prediction (Supervised + Unsupervised ML)

This capstone project demonstrates a complete machine learning workflow using both:

- **Supervised Learning** (classification using Logistic Regression)
- **Unsupervised Learning** (clustering using KMeans)
- **Interactive Deployment Demo** using **Gradio**

The project uses the **Breast Cancer Wisconsin Dataset** (from Scikit-learn) to predict tumor type and cluster similar tumor samples.

---

## 📌 Project Overview

This project covers:

### ✅ Supervised Learning
A classification model is trained to predict whether a tumor is:

- **Malignant (0)**
- **Benign (1)**

The supervised model is implemented using a pipeline:

- `StandardScaler`
- `LogisticRegression`

---

### ✅ Unsupervised Learning
KMeans clustering is applied to group similar tumor samples.

To select the best number of clusters, the project evaluates **silhouette scores** for different values of **K (2 to 7)** and chooses the best-performing K.

Clusters are visualized using **PCA (2D projection)**.

---

### ✅ Gradio App (Interactive UI)
A Gradio interface is built to allow users to:

- Enter tumor feature values
- Receive:
  - **Prediction output** (malignant or benign)
  - **Cluster assignment output**

---

## 🎯 Objectives

- Build a supervised learning model for tumor classification.
- Apply unsupervised learning to cluster tumor samples.
- Use silhouette score to select optimal K.
- Visualize clusters in 2D using PCA.
- Deploy a lightweight demo using Gradio.

---

## 📊 Dataset

This project uses:

### Breast Cancer Wisconsin Dataset (Scikit-learn)
- **569 samples**
- **30 numerical features**
- Target labels:
  - `0 = Malignant`
  - `1 = Benign`

---

## 🧠 Models Used

### Supervised Model: Logistic Regression
- Pipeline includes feature scaling.
- Train-test split:
  - **80% training**
  - **20% testing**
  - Stratified split used to preserve class balance.

Evaluation includes:
- Accuracy score
- Classification report (precision, recall, F1-score)
- Confusion matrix

---

### Unsupervised Model: KMeans Clustering
- Features scaled before clustering.
- Silhouette score used to select K.
- PCA used to reduce dimensions from 30 → 2 for visualization.

---

## 🛠️ Tools & Libraries

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib / Seaborn
- Gradio

---

## 🚀 How to Run the Project

### 1) Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

pip install -r requirements.txt


pip install numpy pandas scikit-learn matplotlib seaborn gradio


