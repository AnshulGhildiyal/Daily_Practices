# TASK 1: AIRBNB PRICE PREDICTION


## **Project Overview**
This project aims to predict the prices of Airbnb listings based on various features, including location, amenities, and property characteristics. By leveraging Machine Learning algorithms, the project provides insights into the key factors influencing rental prices and builds a predictive model to assist hosts and users in understanding market trends.

---

## **Dataset**
The dataset used for this project contains information about Airbnb listings, including:
- **Location**: Neighborhood, latitude, and longitude.
- **Property Details**: Room type, number of bedrooms, bathrooms, etc.
- **Amenities**: Availability of WiFi, kitchen, parking, etc.
- **Pricing Information**: The target variable is the price of the listing.

The dataset was preprocessed to handle missing values, encode categorical features, and normalize numerical features.

---

## **Key Steps**

1. **Exploratory Data Analysis (EDA):**
   - Visualized distributions of prices and other key features.
   - Examined correlations and patterns between predictors and price.

2. **Data Preprocessing:**
   - Handled missing values using appropriate imputation techniques.
   - Encoded categorical variables (e.g., one-hot encoding).
   - Scaled numerical variables for consistent input to ML algorithms.

3. **Modeling:**
   - Built a regression model using [insert model name, e.g., Linear Regression, Random Forest, etc.].
   - Evaluated model performance using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared.

4. **Hyperparameter Tuning:**
   - Implemented hyperparameter tuning using GridSearchCV and RandomizedSearchCV.
   - Optimized parameters for better model performance.

5. **Model Comparison:**
   - Compared multiple models, including Linear Regression, Random Forest, and Gradient Boosting.
   - Highlighted the best-performing model based on evaluation metrics.

6. **Explainability with SHAP:**
   - Used SHAP (SHapley Additive exPlanations) values to interpret model predictions.
   - Visualized feature importance and individual predictions for better understanding.

7. **Evaluation and Insights:**
   - Assessed model accuracy and identified key factors influencing Airbnb prices.
   - Suggested improvements based on feature importance and error analysis.

---

## **Technologies Used**
- **Python**: Core programming language.
- **Libraries**:
  - **Pandas**: Data manipulation and analysis.
  - **Matplotlib/Seaborn**: Data visualization.
  - **Scikit-learn**: Machine Learning modeling and evaluation.
- **Jupyter Notebook**: Development environment.

---

## **Results and Insights**
- The model achieved:
  - RMSE: 3527.1742324476904
  - R²: 0.5654211777836048
  - Random Forest MAE: 35.27284843624934
  - Gradient Boosting MAE: 35.1816597110947
  - 

- Visualization highlights:
  - Price distributions.
  - Relationships between room types and prices.
  - Neighborhood-level analysis.

---

## **Future Work**
- **Feature Engineering**: Incorporate time-based features (e.g., seasonality), group similar neighborhoods, and handle outliers.
- **Advanced Modeling**: Explore more sophisticated models such as Random Forest, XGBoost, or Neural Networks.
- **Explainability**: Use SHAP values or feature importance plots to improve interpretability.
- **Deployment**: Develop a user-friendly app or dashboard for price predictions.

---

## **How to Use**
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/airbnb-price-prediction.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   ```bash
   jupyter notebook Airbnb_Price_Prediction.ipynb
   ```

---

## **Contributing**
Contributions are welcome! If you have suggestions for improvements, please create a pull request or open an issue.


---

## **Acknowledgments**
- Dataset source: [Dataset]([url](https://www.kaggle.com/datasets/airbnb/seattle)).
- Tutorials and resources that helped in completing this project.

---
