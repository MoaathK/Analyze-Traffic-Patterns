
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Traffic.csv')

# Check for any missing values and fill them if necessary
df.ffill(inplace=True)

# Define the features and the target
print("Test")
X = df.drop(['TrafficSituation'], axis=1)
y = df['TrafficSituation']

# Preprocessing steps for numerical and categorical data
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Apply the preprocessing to the data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Oversampling the minority class using SMOTE
smote = SMOTE(random_state=0)
X_train_res, y_train_res = smote.fit_resample(X_train_preprocessed, y_train)

# Model training with Logistic Regression and Random Forest
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForestClassifier': RandomForestClassifier(n_estimators=100)
}

# Train the models and store the results
results = {}
for model_name in models:
    model = models[model_name]
    model.fit(X_train_res, y_train_res)
    predictions = model.predict(X_test_preprocessed)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='macro')
    recall = recall_score(y_test, predictions, average='macro')
    f1 = f1_score(y_test, predictions, average='macro')
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_preprocessed), multi_class='ovr', average='macro')
    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    }

# Evaluate the models using cross-validation
cv_scores = {}
for model_name in models:
    model = models[model_name]
    scores = cross_val_score(model, X_train_res, y_train_res, cv=5, scoring='accuracy')
    cv_scores[model_name] = scores

# Print the evaluation results
for model_name in results:
    print(f"{model_name} performance:")
    for metric in results[model_name]:
        print(f"{metric}: {results[model_name][metric]:.4f}")
    print(f"Cross-validation scores: {cv_scores[model_name]}")
    print('---')

# Confusion matrices
for model_name in models:
    model = models[model_name]
    predictions = model.predict(X_test_preprocessed)
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# Tools used
tools_used = ['Pandas', 'NumPy', 'Scikit-learn', 'Matplotlib', 'Seaborn', 'imbalanced-learn']
