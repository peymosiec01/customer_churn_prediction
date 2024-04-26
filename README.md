# Bank Customer Churn Prediction

This repository contains a machine learning solution to predict customer churn for a subscription-based service or business, specifically focusing on a bank customer dataset obtained from Kaggle. The goal is to develop a model that can accurately predict whether a customer will exit the bank or not based on historical customer data, including features such as usage behavior and customer demographics.

## Dataset Description

The Bank Customer dataset consists of approximately 10,000 observations and 13 features. Some of the key features include:

- **CreditScore**: The credit score of the customer.
- **Geography**: The country of the customer.
- **Gender**: Gender of the customer.
- **Age**: Age of the customer.
- **Tenure**: Number of years the customer has been with the bank.
- **Balance**: Account balance of the customer.
- **NumOfProducts**: Number of bank products the customer uses.
- **HasCrCard**: Whether the customer has a credit card.
- **IsActiveMember**: Whether the customer is an active member.
- **EstimatedSalary**: Estimated salary of the customer.
- **Exited**: Target variable indicating whether the customer exited the bank (1) or not (0).

## Approach

### 1. Data Preprocessing

- Conversion of categorical features to numeric values.
- Feature engineering to select relevant features.
- Handling of missing values (if any).

### 2. Exploratory Data Analysis (EDA)

- Understanding the distribution of features.
- Analysis of class distribution to identify potential biases.
- Visualization of key insights to inform model development.

### 3. Model Development

- Utilization of machine learning algorithms such as Logistic Regression, Random Forests, and Gradient Boosting.
- Evaluation of models using performance metrics like accuracy, precision, recall, F1 score, and area under the ROC curve (AUC).
- Hyperparameter tuning using Grid Search to optimize model performance.

### 4. Model Evaluation

- Comparison of performance metrics for different models.
- Selection of the best-performing model based on evaluation results.

## Results

The Random Forest classifier demonstrated moderate performance with an accuracy of 0.78, precision of 0.48, recall of 0.68, and an F1 score of 0.56 on the validation subset. The area under the ROC curve (AUC) was 0.83, indicating acceptable model performance. However, there is room for improvement in precision and recall.

## Files

- `customer_churn_prediction.ipynb`: Jupyter notebook containing the code for data preprocessing, exploratory data analysis, model development, and evaluation.
- `bank_customer_dataset.csv`: The dataset used for training and testing the models.

## Usage

To replicate the analysis and model development:

1. Clone this repository to your local machine.
2. Ensure you have the required dependencies installed (NumPy, Pandas, Scikit-learn, Seaborn, Matplotlib, etc.).
3. Run the Jupyter notebook `customer_churn_prediction.ipynb` and follow the instructions.

## Conclusion

This project demonstrates the application of machine learning techniques to predict customer churn in a bank setting. While the developed model shows promising results, further refinement and optimization may enhance its predictive performance. Additionally, the insights gained from this analysis can inform strategies for customer retention and business decision-making.

Feel free to contribute to this repository by providing feedback, suggesting improvements, or extending the analysis with additional features or models. Your contributions are highly appreciated!

**Author:** Moses Babalola

**Contact:** mosebabalola@gmail.com

**Date:** 26/04/2024
