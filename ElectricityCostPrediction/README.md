# ⚡ Electricity Cost Prediction using Regression Models

## 📌 Project Overview

This project is an end-to-end **Machine Learning regression application** that predicts **electricity cost** based on multiple real-world input factors such as site area, water consumption, utilisation rate, resident count, and structure type.

The project not only compares **multiple regression models**, but also deploys them using a **Streamlit web application** with an interactive UI.

---

## 🎯 Objective

* Apply and compare **multiple regression algorithms**
* Select the best-performing model
* Save trained models using **Joblib**
* Build an interactive **Streamlit application** for predictions
* Present the project in a **portfolio / GitHub-ready format**

---

## 🧠 Machine Learning Models Used

The following regression models were trained and evaluated:

* Linear Regression
* Ridge Regression
* Lasso Regression
* ElasticNet Regression
* Gradient Boosting Regressor ⭐ (Best Model)

Each model was trained using the same feature set and evaluated using **R² score**.

---

## 📊 Features Used for Prediction

The model uses the following input features:

* **Site Area**
* **Water Consumption**
* **Utilisation Rate**
* **Resident Count**
* **Structure Type** (One-Hot Encoded)

  * Industrial
  * Mixed-Use
  * Residential

Final feature vector:

```
[site_area,
 water_consumption,
 utilisation_rate,
 resident_count,
 structure_type_industrial,
 structure_type_mixed_use,
 structure_type_residential]
```

---

## 🗂️ Project Structure

```
Regression_Project/
│
├── data/
│   └── dataset.csv
│
├── notebooks/
│   └── electricity_cost_model.ipynb
│
├── models/
│   ├── linear_regression_model.pkl
│   ├── ridge_model.pkl
│   ├── lasso_model.pkl
│   ├── elasticnet_model.pkl
│   └── gradient_boosting_model.pkl
│
├── app.py
├── requirements.txt
└── README.md
```

---

## 💾 Model Saving

All trained models were saved using **Joblib**:

```python
import joblib
joblib.dump(model, "model_name.pkl")
```

These models are later loaded directly into the Streamlit application.

---

## 🖥️ Streamlit Web Application

The Streamlit app allows users to:

* Adjust feature values using **sliders**
* Select a regression model from a **dropdown**
* View predictions instantly
* Identify the **best-performing model** using a badge
* Use the app comfortably in **dark mode**

### Run the App Locally

```bash
streamlit run app.py
```

---

## 📈 Evaluation Metric

* **R² Score** was used to evaluate all models
* Gradient Boosting achieved the highest performance and was selected as the best model

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Joblib
* Streamlit

---

## 🚀 Future Improvements

* Display predictions from **all models at once**
* Add **feature importance visualization**
* Deploy the app using **Streamlit Cloud / Hugging Face Spaces**
* Add model comparison charts

---

## 👩‍💻 Author

**Laiba Khan**
Machine Learning Enthusiast | Data Science Learner

---

## ⭐ Acknowledgement

This project was built as a practical implementation of regression techniques in Machine Learning, focusing on both **model performance** and **real-world usability**.

If you like this project, feel free to ⭐ the repository!
