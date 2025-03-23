# Titanic Survival Prediction

## Project Overview
This project predicts passenger survival on the Titanic using various machine learning classifiers. The dataset undergoes preprocessing before model training and evaluation. The best model is selected based on performance metrics.

## Dataset
The dataset used is `tested.csv`, containing passenger details and survival labels.
- Dataset link: [Kaggle - Titanic Test File](https://www.kaggle.com/datasets/brendan45774/test-file)

## Preprocessing Steps
1. **Handling Missing Values:**
   - `Age` is imputed with the median value.
   - `Fare` missing values are filled with the median.
   - `Cabin`, `Ticket`, and `Name` columns are dropped due to high missing values or irrelevance.

2. **Feature Engineering:**
   - Categorical encoding is applied using one-hot encoding (`Sex`, `Embarked`).

3. **Splitting Data:**
   - The dataset is split into training (80%) and test (20%) sets.
   
4. **Feature Scaling:**
   - StandardScaler is used to normalize numerical features.

## Model Selection
Various classifiers are trained and evaluated:
- **Support Vector Classifier (SVC)**
- **XGBoost Classifier**
- **Random Forest Classifier**
- **AdaBoost Classifier**
- **Gradient Boosting Classifier**
- **Naive Bayes (GaussianNB)**
- **Logistic Regression**

### Hyperparameter Tuning
- GridSearchCV is used for tuning SVC hyperparameters (`C`, `gamma`, `kernel`).
- The best model is selected based on accuracy.

## Performance Analysis
The classifiers are evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**

## Model Storage
- The best model (`SVC`) and the imputer are saved using `joblib` in the `models/` directory.

## Visualization
- A bar plot compares classifier accuracy.
- A heatmap displays the confusion matrix for the best model.

## Execution
Run the script to preprocess data, train models, and evaluate performance.

```bash
python titanic_analysis.py
```

## Dependencies
Ensure the following libraries are installed:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- joblib

Install dependencies using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

## Output
- Printed performance metrics
- Saved best model (`models/titanic_model.pkl`)
- Saved imputer (`models/imputer.pkl`)

## Conclusion
At a predictive accuracy of **99.01%**, my model demonstrates its potential to forecast Titanic passenger survival effectively. This project not only illustrates the practical application of machine learning techniques on historical data but also provides insights into the influential factors behind survival rates during the Titanic disaster.



