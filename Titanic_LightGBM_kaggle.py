import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# For different classification models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
    print("LightGBM not installed. To use it, run: pip install lightgbm")

# For model evaluation
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# --- 1. Load the Datasets ---
print("--- Loading Kaggle Titanic Datasets ---")
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    # gender_submission_df = pd.read_csv('gender_submission.csv') # Not strictly needed for the script to run
    print("Train and Test datasets loaded successfully!")
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
except FileNotFoundError:
    print("Error: train.csv, test.csv not found.")
    print("Please make sure these files are in the same directory as your script.")
    raise FileNotFoundError("Kaggle data files not found. Please download them.")

# Store PassengerId from test_df for submission later, BEFORE any modification
test_passenger_ids = test_df['PassengerId']

# --- 2. Define Preprocessing Functions ---

def handle_missing_values(df_input):
    """Handles missing values for Age, Embarked, Fare, and drops Cabin."""
    df = df_input.copy()

    median_age = df['Age'].median()
    df['Age'].fillna(median_age, inplace=True)

    if 'Embarked' in df.columns and df['Embarked'].isnull().sum() > 0:
        mode_embarked = df['Embarked'].mode()[0]
        df['Embarked'].fillna(mode_embarked, inplace=True)

    if 'Fare' in df.columns and df['Fare'].isnull().sum() > 0:
        median_fare = df['Fare'].median()
        df['Fare'].fillna(median_fare, inplace=True)

    if 'Cabin' in df.columns:
        df.drop('Cabin', axis=1, inplace=True)

    return df

def feature_engineer(df_input):
    """Performs feature engineering: FamilySize, IsAlone, Title, and drops redundant features."""
    df = df_input.copy()

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    if 'Name' in df.columns:
        df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                            'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                            'Jonkheer', 'Dona', 'Mlle', 'Ms', 'Mme'], 'Rare')
        df['Title'] = df['Title'].replace('Mlle', 'Miss')
        df['Title'] = df['Title'].replace('Ms', 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')
    else:
        df['Title'] = 'Unknown'

    columns_to_drop = ['Name', 'SibSp', 'Parch', 'Ticket']
    if 'PassengerId' in df.columns:
        columns_to_drop.append('PassengerId')

    df.drop(columns_to_drop, axis=1, inplace=True, errors='ignore')

    return df

# --- 3. Apply Preprocessing Functions to both Train and Test Data ---
print("\n--- Handling Missing Values for Train and Test Data ---")
train_df = handle_missing_values(train_df)
test_df = handle_missing_values(test_df)

print("Missing values in train_df after handling:", train_df.isnull().sum().sum())
print("Missing values in test_df after handling:", test_df.isnull().sum().sum())

print("\n--- Applying Feature Engineering to Train and Test Data ---")
train_df = feature_engineer(train_df)
test_df = feature_engineer(test_df)

# --- 4. Set up Preprocessing Pipeline for Categorical and Numerical Features ---
print("\n--- Setting up Preprocessing Pipeline for Categorical and Numerical Features ---")

numerical_features = ['Age', 'Fare', 'FamilySize']
categorical_features = ['Pclass', 'Sex', 'Embarked', 'IsAlone', 'Title']

# Ensure identified categorical features are treated as such
for col in categorical_features:
    if col in train_df.columns:
        train_df[col] = train_df[col].astype('category')
    if col in test_df.columns:
        test_df[col] = test_df[col].astype('category')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
    ],
    remainder='passthrough'
)

X_train_full = train_df.drop('Survived', axis=1)
y_train_full = train_df['Survived']

X_train_processed = preprocessor.fit_transform(X_train_full)
X_test_processed = preprocessor.transform(test_df)

# Get feature names after one-hot encoding for transformed DataFrames
onehot_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()
processed_feature_names = numerical_features + onehot_feature_names

X_train_processed_df = pd.DataFrame(X_train_processed, columns=processed_feature_names)
X_test_processed_df = pd.DataFrame(X_test_processed, columns=processed_feature_names)

print(f"Shape of X_train_processed_df: {X_train_processed_df.shape}")
print(f"Shape of X_test_processed_df: {X_test_processed_df.shape}")

# --- 5. Prepare Data for Model Training ---
print("\n--- Preparing Data for Model Training ---")
X = X_train_processed_df
y = y_train_full
X_test_final = X_test_processed_df

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")

# --- 6. Model Training, Evaluation, and Prediction ---
print("\n--- Training and Evaluating Models ---")

models = {
    'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear'),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Support Vector Machine': SVC(random_state=42, probability=True, kernel='rbf'),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=7)
}

if lgb is not None:
    models['LightGBM'] = lgb.LGBMClassifier(random_state=42, n_estimators=200, learning_rate=0.05, num_leaves=31)

model_performance = {}
kaggle_predictions = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    y_pred_val = model.predict(X_val)
    y_prob_val = model.predict_proba(X_val)[:, 1]

    accuracy = accuracy_score(y_val, y_pred_val)
    precision = precision_score(y_val, y_pred_val)
    recall = recall_score(y_val, y_pred_val)
    f1 = f1_score(y_val, y_pred_val)
    roc_auc = roc_auc_score(y_val, y_prob_val)

    model_performance[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC AUC': roc_auc
    }

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")

    # Make predictions on the actual Kaggle test set
    kaggle_predictions[name] = model.predict(X_test_final)

# Convert performance results to a DataFrame for easy comparison
performance_df = pd.DataFrame(model_performance).T
print("\n--- Model Performance Comparison (Validation Set) ---")
print(performance_df)

# --- 7. Hyperparameter Tuning (Example: Random Forest using GridSearchCV) ---
print("\n--- Hyperparameter Tuning: Random Forest using GridSearchCV ---")
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [5, 7, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                              param_grid=param_grid_rf,
                              cv=3, n_jobs=-1, verbose=0, scoring='accuracy') # Reduced cv and verbose for faster run
grid_search_rf.fit(X_train, y_train)

print(f"\nBest parameters for Random Forest: {grid_search_rf.best_params_}")
print(f"Best cross-validation accuracy for Random Forest (GridSearchCV): {grid_search_rf.best_score_:.4f}")

best_rf_model = grid_search_rf.best_estimator_
y_pred_best_rf_val = best_rf_model.predict(X_val)
y_prob_best_rf_val = best_rf_model.predict_proba(X_val)[:, 1]

# Evaluate the best RF model
accuracy_best_rf = accuracy_score(y_val, y_pred_best_rf_val)
precision_best_rf = precision_score(y_val, y_pred_best_rf_val)
recall_best_rf = recall_score(y_val, y_pred_best_rf_val)
f1_best_rf = f1_score(y_val, y_pred_best_rf_val)
roc_auc_best_rf = roc_auc_score(y_val, y_prob_best_rf_val)

model_performance['Best Random Forest (tuned)'] = {
    'Accuracy': accuracy_best_rf,
    'Precision': precision_best_rf,
    'Recall': recall_best_rf,
    'F1-Score': f1_best_rf,
    'ROC AUC': roc_auc_best_rf
}
kaggle_predictions['Best Random Forest (tuned)'] = best_rf_model.predict(X_test_final)

print("\n--- Updated Model Performance Comparison (Validation Set) ---")
print(pd.DataFrame(model_performance).T)


# --- 8. Cross-Validation (Example: LightGBM, if available) ---
if lgb is not None:
    print("\n--- Cross-Validation: LightGBM ---")
    model_lgbm_cv = lgb.LGBMClassifier(random_state=42, n_estimators=200, learning_rate=0.05, num_leaves=31)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model_lgbm_cv, X, y, cv=kf, scoring='accuracy', n_jobs=-1)

    print(f"LightGBM Cross-Validation Scores (Accuracy): {cv_scores}")
    print(f"LightGBM Mean CV Accuracy: {cv_scores.mean():.4f}")
    print(f"LightGBM Standard Deviation of CV Accuracy: {cv_scores.std():.4f}")
else:
    print("\n--- Skipping LightGBM Cross-Validation: LightGBM library not found. ---")


# --- 9. Select the Best Model for Submission ---
print("\n--- Selecting the Best Model for Submission ---")

# You can choose the metric to optimize for. Common choices: 'Accuracy', 'F1-Score', 'ROC AUC'
optimization_metric = 'Accuracy'
best_model_name = None
best_score = -1

for name, metrics in model_performance.items():
    if metrics[optimization_metric] > best_score:
        best_score = metrics[optimization_metric]
        best_model_name = name

print(f"The best model based on '{optimization_metric}' is: {best_model_name} with a score of {best_score:.4f}")

final_predictions = kaggle_predictions[best_model_name]

# --- 10. Generate Submission File ---
print("\n--- Generating Submission File ---")

submission_df = pd.DataFrame({
    'PassengerId': test_passenger_ids,
    'Survived': final_predictions
})

submission_file_path = 'submission.csv'
submission_df.to_csv(submission_file_path, index=False)

print(f"Submission file created successfully at: {submission_file_path}")
print("\nFirst 5 rows of the submission file:")
print(submission_df.head())