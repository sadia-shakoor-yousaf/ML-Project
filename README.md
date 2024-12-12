# Cerebral Stroke Prediction Report

## 1. Introduction
Cerebral strokes are a leading cause of disability and mortality worldwide. Early prediction of stroke risk can enable timely interventions and reduce adverse outcomes. This report analyzes a dataset containing 7,000 records with demographic and health-related features to predict the likelihood of stroke occurrence. The class label 1 indicates a chance of stroke and is present in 132 records, highlighting a significant class imbalance. 

The objective is to enhance the recall for class 1 using three machine learning models: Random Forest, Logistic Regression, and Support Vector Machine (SVM).

## 2. Methodology

### Data Preprocessing
1. **Data Cleaning**:
   - Verified for missing or inconsistent values.
   - Handled categorical variables through one-hot encoding.
2. **Feature Scaling**:
   - Applied standard scaling to ensure uniformity in numerical features.
3. **Class Imbalance Handling**:
   - Used Synthetic Minority Oversampling Technique (SMOTE) to balance the dataset.

### Algorithms Applied
1. **Random Forest**:
   - An ensemble method combining multiple decision trees to improve prediction accuracy and control overfitting.
2. **Logistic Regression**:
   - A linear model optimized for binary classification tasks.
3. **Support Vector Machine (SVM)**:
   - A robust classification algorithm, effective for small datasets and capable of handling high-dimensional feature spaces.

### Optimization Techniques
- **Hyperparameter Tuning**:
  - Grid Search and Random Search was used to identify the optimal parameters for each model.

## 3. Results

### Metrics
Evaluation metrics include precision, recall, F1-score, and accuracy. Emphasis was placed on maximizing recall for class 1 due to its critical importance.

| Model               | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | Overall Accuracy |
|---------------------|---------------------|------------------|--------------------|------------------|
| Random Forest       | 0.06               | 0.27             | 0.09               | 0.95             |
| Logistic Regression | 0.06               | 0.86             | 0.10               | 0.69             |
| SVM                 | 0.06               | 0.27             | 0.09               | 0.89             |

### Comparison of Machine Learning Algorithms on Stroke Dataset

| Algorithm           | Accuracy | Precision | Recall | F1-Score | Best Hyperparameters                          | Execution Time (s) | Remarks                          |
|---------------------|----------|-----------|--------|----------|----------------------------------------------|---------------------|----------------------------------|
| Random Forest       | 0.95     | 0.10      | 0.18   | 0.13     | 'class_weight': None, 'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 382 | 118                 | Performed well with class 0 data but struggled with class 1 data. |
| Support Vector Machine (SVM) | 0.89     | 0.06      | 0.27   | 0.09     | 'kernel': 'rbf', 'gamma': 'auto', 'class_weight': 'balanced', 'C': 10 | 188                 | Achieved good recall but slow due to dataset size. |
| Logistic Regression | 0.69     | 0.06      | 0.86   | 0.10     | 'C': 0.01, 'class_weight': 'balanced', 'penalty': 'l1', 'solver': 'liblinear' | 0.3                 | Excellent recall but poor overall performance.    |

### Visualizations
1. **Confusion Matrix**:
   - Displays the classification performance for each model.

## 4. Analysis

### Insights
- **Random Forest** exhibited a balanced overall performance but struggled with class 1 precision.
- **SVM** achieved reasonable recall but was computationally intensive, with limited improvement in class 1 prediction.
- **Logistic Regression** demonstrated the highest recall for class 1 but suffered in precision and overall accuracy.

### Algorithm Comparison
- The trade-off between recall and precision is evident across models. Logistic Regression favors recall, Random Forest maintains a balance, and SVM focuses on precision and overall accuracy.

### Challenges
1. **Class Imbalance**:
   - The significant disparity between classes required careful handling to prevent biased models.
2. **Hyperparameter Optimization**:
   - Computationally intensive, especially for Random Forest and SVM.
3. **Overfitting**:
   - Prevented using cross-validation and parameter tuning.

## Conclusion
The study demonstrates the utility of machine learning algorithms in stroke prediction, emphasizing the importance of recall for high-risk cases. Future work could explore advanced techniques like ensemble learning or deep learning models to further enhance performance.
