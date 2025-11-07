# 🩺 Day 11 — Diabetes Prediction using Machine Learning  

### 📘 Project Overview  
This project focuses on predicting the likelihood of diabetes in patients using the **Pima Indians Diabetes Dataset**.  
The model applies core **Machine Learning techniques** — including Logistic Regression and Random Forest — to classify patients as diabetic or non-diabetic based on medical indicators.  

### 🧠 Key Highlights  
- Performed full **data preprocessing** (handling missing values, normalization).  
- Trained **Logistic Regression** and **Random Forest** models for comparison.  
- Evaluated model performance using Accuracy, Precision, Recall, F1-score, and ROC-AUC metrics.  
- Visualized insights and model performance using **Matplotlib** and **Seaborn**.  
- Saved trained model and scaler for future predictions.  

### ⚙️ Tech Stack  
**Python**, **Pandas**, **Scikit-learn**, **Matplotlib**, **Seaborn**, **Joblib**

### 📈 Workflow  
1️⃣ Load and clean dataset (`diabetes.csv`)  
2️⃣ Train-test split  
3️⃣ Model training (Logistic Regression + Random Forest)  
4️⃣ Model evaluation and visualization  
5️⃣ Save outputs (`outputs/` folder)

### 🔍 Results Summary  
| Metric | Logistic Regression | Random Forest |
|--------|----------------------|----------------|
| Accuracy | 0.71 | 0.74 |
| Precision | 0.60 | 0.65 |
| Recall | 0.50 | 0.55 |
| ROC-AUC | 0.81 | 0.82 |

### 📊 Sample Output  
```
📥 Loading dataset...
✅ Dataset loaded successfully.

🧠 Training models...
🔸 Logistic Regression Results: Accuracy = 0.7078, ROC-AUC = 0.8130
🔸 Random Forest Results: Accuracy = 0.7403, ROC-AUC = 0.8173

✅ Outputs saved to /outputs
```

### 🧩 How to Run  
1. Clone this repository  
   ```bash
   git clone https://github.com/AbhineetS/Day-11-Diabetes-Prediction.git
   cd Day-11-Diabetes-Prediction
   ```
2. Activate virtual environment and install dependencies  
   ```bash
   pip3 install -r requirements.txt
   ```
3. Run the project  
   ```bash
   python3 run_diabetes_model.py
   ```

### 🏁 About  
This project is part of my **64-Day AI/ML Challenge**, aimed at building practical, working machine learning projects that strengthen real-world understanding of data and model deployment.  
Each project adds to a growing collection of polished, professional repositories demonstrating consistency and capability.

---

### 📬 Connect with Me  
**GitHub:** [AbhineetS](https://github.com/AbhineetS)  
**LinkedIn:** [linkedin.com/in/abhineetsingh-ai](#)