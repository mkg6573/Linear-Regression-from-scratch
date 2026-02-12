# Linear Regression From Scratch 📈

This project implements **Linear Regression from scratch using Python**, without using machine learning libraries like `scikit-learn`.  
The goal is to understand the **mathematics and logic behind linear regression**, including cost function optimization using **Gradient Descent**.

🚀 Live Project Deployment

🔗 Streamlit Web App:
👉 https://mobile-priceprediction-004.streamlit.app/

This web application allows users to input mobile specifications and get real-time price predictions using the trained Multiple Linear Regression model.


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
📌 Objective

Notebook : https://github.com/mkg6573/Linear-Regression-from-scratch/blob/main/Simple_Linear_Regression.ipynb

* The objective of this experiment is to implement Simple Linear Regression from scratch using Gradient Descent to model the linear relationship between a single independent variable and the target variable.

📊 Dataset Description

Dataset Name: after_EDA_dataset.csv

Independent Variable (Feature): processor_speed

Dependent Variable (Target): price

The dataset was cleaned and preprocessed after Exploratory Data Analysis (EDA).

🧠 Methodology
--
1️⃣ Data Preprocessing
-
Fill value : https://github.com/mkg6573/Linear-Regression-from-scratch/blob/main/fill-NAN-value.ipynb

EDA : https://github.com/mkg6573/Linear-Regression-from-scratch/blob/main/EDA.ipynb

corelation : https://github.com/mkg6573/Linear-Regression-from-scratch/blob/main/Correlation.ipynb
Loaded dataset using Pandas

Extracted feature (X) and target (y)

Reshaped arrays appropriately

2️⃣ Model Initialization
-
Parameters initialized:

θ₀ (Intercept)

θ₁ (Slope)

Learning rate defined

Number of epochs specified

3️⃣ Gradient Descent Implementation
-
Predictions calculated using:
          y=θ0​+θ1​x

Cost function computed using Mean Squared Error (MSE)

Gradients derived manually

Parameters updated iteratively

4️⃣ Training Monitoring
-
Cost recorded per epoch

Verified convergence via decreasing loss

🧮 Model Equation

y=θ0​+θ1​x


Where:
x = processor speed
y = price

📈 Evaluation Metrics

- R² Score: **0.3869**
- MAE: **9324.92**
- RMSE: **17917.85**

📌 Interpretation:
- Explains only 38% of variance in price.
- High error values indicate underfitting.
- Not suitable for complex mobile pricing patterns.

📊 Results & Observations

The cost function decreases steadily.

The model captures the linear relationship between processor speed and price.

Works well if the relationship is approximately linear.

🏁 Conclusion

Simple Linear Regression was successfully implemented from scratch using Gradient Descent, demonstrating the fundamental working of supervised learning without external ML libraries.

---

### 🔹 2. Multiple Linear Regression
📌 Objective
Notebook : https://github.com/mkg6573/Linear-Regression-from-scratch/blob/main/Multiple_Linear_Regression.ipynb
To implement Multiple Linear Regression from scratch using Gradient Descent to model the relationship between multiple features and price.

📊 Dataset Description

* Dataset Name: after_EDA_dataset.csv
* Independent Variables:
* processor_speed
* RAM
* storage
* (other selected features)

Dependent Variable: price
🧠 Methodology
1️⃣ Data Preparation

* Selected multiple relevant features
* Constructed feature matrix (X)
* Added bias column (ones column)

2️⃣ Model Initialization

* Parameter vector θ initialized to zeros
* Learning rate defined
* Epochs defined

3️⃣ Vectorized Gradient Descent

Predictions computed using: y=Xθ
Cost function computed using vectorized MSE

Gradient computed using matrix operations

Parameters updated simultaneously

4️⃣ Training Monitoring

Cost tracked over epochs

Convergence confirmed through decreasing loss

🧮 Model Equation
         y=θ0​+θ1​x1​+θ2​x2​+...+θn​xn​

📈 Evaluation Metrics
- R² Score: **0.8416**
- MAE: **4963.06**
- RMSE: **9108.06**

📌 Interpretation:
- Explains 84% of variance in price.
- Significant improvement over Simple Linear Regression.
- Much lower prediction error.
- Best performing model among all tested models.

📊 Results & Observations

Model captures influence of multiple features on price.

Better predictive performance than Simple Linear Regression.

Handles multivariate relationships effectively.

🏁 Conclusion

Multiple Linear Regression was successfully implemented using fully vectorized Gradient Descent, improving predictive accuracy by leveraging multiple input features
---

### 🔹 3. Polynomial Regression
📌 Objective

Notebook : https://github.com/mkg6573/Linear-Regression-from-scratch/blob/main/Polynomial_Regression.ipynb

The objective of this experiment is to implement Polynomial Regression from scratch using Gradient Descent to model the non-linear relationship between processor speed and price without using any pre-built machine learning libraries.

📊 Dataset Description

Dataset Name: after_EDA_dataset.csv

Independent Variable (Feature): processor_speed

Dependent Variable (Target): price

The dataset was cleaned and preprocessed after performing Exploratory Data Analysis (EDA).

🧠 Methodology
1️⃣ Data Preparation

Dataset loaded using Pandas

Extracted feature (X) and target (y)

Reshaped arrays where necessary

2️⃣ Feature Engineering

*Created polynomial feature of degree 2 manually:
                    𝑥2
*Constructed the Design Matrix (X):
   *Bias term (1)
   *Linear term (x)
   *Polynomial term (x²)

3️⃣ Model Initialization

*Parameters (θ₀, θ₁, θ₂) initialized to zero
*Learning rate defined
*Number of epochs defined

4️⃣ Gradient Descent Implementation

Predicted output calculated using matrix multiplication

Cost function computed using Mean Squared Error

Gradients calculated manually

Parameters updated iteratively

5️⃣ Monitoring Training

Cost recorded for each epoch

Convergence verified through decreasing loss

🧮 Model Equation

The Polynomial Regression model used:
          y=θ0​+θ1​x+θ2​x2
          Where:

𝑥= processor speed
𝑦= price
𝜃0= intercept
θ1= linear coefficient
θ2= polynomial coefficient

📈 Evaluation Metrics
- MSE: **547017769.79**
- RMSE: **23388.41**
- R² Score: **0.4365**

📌 Interpretation:
- Performance dropped compared to Multiple Linear Regression.
- Likely overfitting or improper polynomial degree selection.
- Not suitable for this dataset in current configuration.

📊 Results & Observations

The cost function decreases steadily, indicating successful convergence.

Polynomial Regression captures non-linear patterns more effectively than Simple Linear Regression.

The regression curve better fits the curvature of the data.
🏁 Conclusion

Polynomial Regression was successfully implemented from scratch using Gradient Descent. By introducing higher-order terms, the model effectively captured non-linear relationships and improved prediction accuracy.

---

### 🔹 4. Regularization (Ridge & Lasso)
📌 Objective
Notebook : https://github.com/mkg6573/Linear-Regression-from-scratch/blob/main/Regularization.ipynb
-To reduce overfitting and improve model stability.
-Ridge Regression (L2) shrinks large coefficients.
-Lasso Regression (L1) performs feature selection.
-Regularization stabilized coefficients but did not outperform Multiple Linear Regression.

🧠 Methodology
🔹 Ridge Regression (L2 Regularization)

Modified cost function:
       J(θ)=1/2m​∑(hθ​(x)−y)^2+λ/2m​∑θ^2
Penalizes large coefficients.
Shrinks weights but does not eliminate them.

🔹 Lasso Regression (L1 Regularization)
      J(θ)=1/2m​∑(hθ​(x)−y)^2+λ/m​∑∣θ∣
-Can reduce some coefficients to zero.
-Performs feature selection. 

### 🔹 5.  Model Comparison & Final Analysis
📊 Final Performance Comparison

| Model                     | R² Score | RMSE      | Performance |
|----------------------------|----------|-----------|------------|
| Simple Linear Regression   | 0.38     | 17917     | ❌ Poor     |
| Multiple Linear Regression | 0.84     | 9108      | ✅ Best     |
| Polynomial Regression      | 0.43     | 23388     | ❌ Weak     |

✅ **Multiple Linear Regression gives the best results with 84% accuracy.**

---
🏆 Final Conclusion

✅ Multiple Linear Regression gives the best results with 84% accuracy.

It explains most of the variance in mobile prices.

It significantly reduces prediction error.

It captures complex pricing relationships effectively.

Simple Linear Regression underfits the data, while Polynomial Regression did not improve performance in this case.

Therefore, Multiple Linear Regression was selected for deployment in the Streamlit web application.

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

🌐 Deployment Details
🔹 Model File
phone_price_model.pkl

🔹 Streamlit App
app.py

🔹 Prediction Logic
*Model trained on log1p(price)
*Predictions converted back using:
*predicted_price = np.expm1(log_price)

🔹 User Inputs in Web App
Brand
Processor Type
RAM
Storage
Camera
Battery
Display
5G Support
NFC
Charging Speed
etc.

🛠 Tech Stack
-Python
-NumPy
-Pandas
-Matplotlib
-Scikit-learn
-Joblib
-Streamlit

🎓 Learning Outcomes
-
Through this project, we gained hands-on experience in:
End-to-end regression modeling
Feature engineering
Model evaluation (MSE, RMSE, R²)
Overfitting and regularization
Model comparison and selection
Deployment using Streamlit
Converting ML models into real-world applications

🏁 Conclusion
-
This project demonstrates a complete machine learning pipeline:
EDA → Model Building → Model Evaluation → Model Comparison → Model Selection → Deployment
Among all tested models, Multiple Linear Regression provided the best balance between accuracy and generalization, achieving an R² score of 0.84 and the lowest RMSE.
The model is successfully deployed as a live web application for real-time mobile price prediction.

⭐ If you like this project, consider giving it a star on GitHub!
