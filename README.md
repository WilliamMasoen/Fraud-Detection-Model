# Fraud Detection Model

This project focuses on building a machine learning model to detect fraudulent credit card transactions. The dataset used for this project comes from the [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), which contains anonymized transaction details with a highly imbalanced class distribution (fraudulent transactions make up only 0.17% of the data).

## ETL Workflow
1. **Extract**: The dataset is loaded from the Kaggle source using `pandas.read_csv()`.
2. **Transform**:
   - Log transformation is applied to the `Amount` column to normalize values.
   - The `Time` column is converted into an `hour` feature to capture time-based transaction patterns.
   - The original `Amount` and `Time` columns are dropped to reduce redundancy.
3. **Load**: The transformed data is split into training and test sets for model training and evaluation.

## Model Selection Process
### Neural Network
Initially, I attempted to use a neural network for classification. Despite fine-tuning hyperparameters such as the number of layers, learning rate, and activation functions, the model failed to achieve good recall scores on the fraud class. Due to the severe class imbalance, the neural network struggled to generalize well for fraudulent transactions.

### Random Forest
Next, I implemented a Random Forest model, which showed a significant improvement over the neural network. However, while the recall and precision for fraudulent transactions improved, the model was still not ideal for minimizing false negatives.

### XGBoost (Final Model)
Finally, I used XGBoost, a powerful gradient boosting algorithm, and it outperformed both previous models. By setting the `scale_pos_weight` parameter to balance the fraud class, along with optimizing hyperparameters such as `n_estimators`, `max_depth`, and `learning_rate`, the model achieved the following results:

```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.91      0.85      0.88        98

    accuracy                           1.00     56962
   macro avg       0.96      0.92      0.94     56962
weighted avg       1.00      1.00      1.00     56962

Test Accuracy:  0.9995962220427653
```

## Key Takeaways
- **Accuracy alone is misleading**: Previous models had over 98% accuracy, but this was due to the dominance of legitimate transactions in the dataset. The key metric to improve was recall for fraud cases.
- **XGBoost's advantage**: By leveraging boosting and handling class imbalance with `scale_pos_weight`, XGBoost significantly improved recall without sacrificing precision.
- **Feature Engineering Matters**: Transforming and selecting the right features, such as normalizing `Amount` and extracting `hour` from `Time`, helped improve model performance.
