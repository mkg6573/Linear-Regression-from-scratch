# Linear Regression From Scratch 📈

This project implements **Linear Regression from scratch using Python**, without using machine learning libraries like `scikit-learn`.  
The goal is to understand the **mathematics and logic behind linear regression**, including cost function optimization using **Gradient Descent**.




## 📊 Mobile Price Prediction Dataset

This dataset is designed to analyze how various smartphone specifications influence mobile prices and overall device value. It provides a structured view of hardware features, software attributes, and connectivity options, making it ideal for **data analysis**, **feature engineering**, and **machine learning model building**.

Although this is not an official commercial dataset, it is well-organized and informative. It helps in developing analytical thinking, understanding feature impact, and gaining hands-on experience with real-world-like data for both academic and professional learning.



## 📌 Key Features / Columns

* **brand** – Mobile phone brand
* **price** – Device price
* **rating** – User rating score
* **is_5g** – 5G network support (Yes/No)
* **is_nfc** – NFC support
* **is_ir_blaster** – IR blaster availability
* **processor_brand** – Processor manufacturer
* **core** – Number of processor cores
* **processor_speed** – Processor speed
* **ram** – RAM capacity
* **internal_memory** – Internal storage size
* **battery_size** – Battery capacity
* **fast_charge** – Fast charging support
* **charging_speed** – Charging speed
* **rear_mp** – Rear camera megapixels
* **front_mp** – Front camera megapixels
* **os** – Operating system
* **display_size** – Screen size
* **refresh_rate** – Display refresh rate
* ......

## 📊 Models Implemented

1. Simple Linear Regression  
2. Multiple Linear Regression  
3. Polynomial Regression  

---

## 📈 Model Performance Comparison

### 🔹 1. Simple Linear Regression

- R² Score: **0.3869**
- MAE: **9324.92**
- RMSE: **17917.85**

📌 Interpretation:
- Explains only 38% of variance in price.
- High error values indicate underfitting.
- Not suitable for complex mobile pricing patterns.

---

### 🔹 2. Multiple Linear Regression

- R² Score: **0.8416**
- MAE: **4963.06**
- RMSE: **9108.06**

📌 Interpretation:
- Explains 84% of variance in price.
- Significant improvement over Simple Linear Regression.
- Much lower prediction error.
- Best performing model among all tested models.

---

### 🔹 3. Polynomial Regression

- MSE: **547017769.79**
- RMSE: **23388.41**
- R² Score: **0.4365**

📌 Interpretation:
- Performance dropped compared to Multiple Linear Regression.
- Likely overfitting or improper polynomial degree selection.
- Not suitable for this dataset in current configuration.

---

## 🏆 Final Conclusion

| Model                     | R² Score | RMSE      | Performance |
|----------------------------|----------|-----------|------------|
| Simple Linear Regression   | 0.38     | 17917     | ❌ Poor     |
| Multiple Linear Regression | 0.84     | 9108      | ✅ Best     |
| Polynomial Regression      | 0.43     | 23388     | ❌ Weak     |

✅ **Multiple Linear Regression gives the best results with 84% accuracy.**

---

## 🛠 Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib

---


---

## 📌 Project Objective

To understand how different regression algorithms perform on real-world pricing data and build strong intuition about model evaluation metrics like:

- R² Score
- MAE
- RMSE
- MSE

---

⭐ If you like this project, consider giving it a star on GitHub!
