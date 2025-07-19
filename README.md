# Gold-Price-Prediction
Introduction
Predicting the price of gold is a valuable task for financial analysts, investors, and economists. Gold is considered a stable investment and is influenced by several market factors including crude oil prices, stock market indices, and the US dollar rate. In this project, I developed a machine learning model that can predict gold prices based on a given set of financial indicators.

This notebook walks through the complete process — from understanding the data and preparing it for modeling, to building and evaluating a regression model using the Random Forest algorithm.

Objective
The main objective of this project was to:

Analyze the relationship between gold prices and other global financial variables.

Explore and visualize correlations among the features.

Build a regression model that can predict gold prices with high accuracy.

Evaluate model performance using appropriate statistical metrics.

Dataset Description
The dataset used in this project is a .csv file named gld_price_data.csv. It contains 2290 records and the following columns:

Date: The time of recording

SPX: S&P 500 index value

GLD: Gold ETF closing price (target variable)

USO: Crude oil ETF price

SLV: Silver ETF price

EUR/USD: Euro to US Dollar exchange rate

This dataset captures various economic indicators which may influence gold prices directly or indirectly.

Tools and Libraries Used
The following Python libraries were used throughout the project:

Pandas – for reading, cleaning, and manipulating the dataset

NumPy – for numerical operations

Matplotlib & Seaborn – for data visualization and plotting correlations

scikit-learn (sklearn) – for building and evaluating the machine learning model

Step-by-Step Workflow
1. Importing Libraries and Loading Data
The first step involved importing the necessary Python libraries. The dataset was then read using Pandas, and basic exploration such as head(), info(), and describe() was done to get a clear understanding of its structure and data types.

2. Exploratory Data Analysis (EDA)
EDA was a critical part of the project. I analyzed:

Data shape, missing values, and column types

Statistical summaries using describe()

Pairwise correlations using corr() and a heatmap

Distribution of each variable using distplot and boxplot

This helped identify patterns and relationships — for instance, the strong positive correlation between GLD and SLV, and inverse trends with SPX.

3. Data Preprocessing
Key steps taken in preprocessing:

Checked for and confirmed the absence of null values

Removed the 'Date' column for modeling as it is not numerically predictive

Split the data into independent features (X) and the target variable (y = GLD)

4. Train-Test Split
Using train_test_split() from sklearn, the dataset was split into training and test sets in an 80:20 ratio to ensure robust performance validation.

5. Model Building – Random Forest Regressor
The Random Forest Regressor was selected due to its ability to handle non-linear data and reduce overfitting.

Trained the model on the training set

Predicted on the test set

Used r2_score, mean_absolute_error, and mean_squared_error to evaluate accuracy

6. Model Evaluation
The model achieved a high R² score, indicating that a large portion of the variance in gold prices was explained by the predictors. Both MAE and MSE were low, suggesting good predictive performance and minimal errors.

To visually assess the model’s performance, actual vs. predicted gold prices were plotted. The points closely followed the diagonal line, validating the accuracy.

Results
The Random Forest model was able to capture the underlying patterns and produced accurate predictions.

Important variables contributing to prediction included silver prices (SLV), oil prices (USO), and S&P index (SPX).

The model could be used as a foundational tool in a larger financial analytics dashboard or automated forecasting pipeline.

What I Learned
The significance of correlation analysis in financial data

How small variations in market indices can affect asset prices like gold

Practical experience in building and tuning machine learning models for regression tasks

The end-to-end flow of a data science project — from data cleaning to final model validation

How to Run
Clone the repository

Open the Jupyter Notebook: Gold_Price_Prediction.ipynb

Ensure required libraries are installed (pip install pandas seaborn matplotlib scikit-learn)

Run all cells sequentially

Future Work
Include time-series analysis for temporal trends

Explore other regression algorithms (XGBoost, Gradient Boosting)

Deploy the model using a web framework (e.g., Flask or Streamlit)

Conclusion
This project demonstrated the use of machine learning to make accurate predictions of gold prices using financial indicators. The results indicate that Random Forest is an effective model for this task and provides strong performance even without extensive hyperparameter tuning. With further development, this approach can be integrated into real-world investment and trading decision systems.
