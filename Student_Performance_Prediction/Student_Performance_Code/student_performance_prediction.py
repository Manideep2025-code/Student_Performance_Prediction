"""
========================================================================================
PROJECT TITLE: STUDENT PERFORMANCE PREDICTION USING LINEAR REGRESSION
DEPARTMENT: Computer Science and Engineering (Artificial Intelligence & Machine Learning)
INSTITUTION: MNR University, School of Engineering & Technology and AI
ACADEMIC YEAR: 2026-2027
ALGORITHM: Multiple Linear Regression (Ordinary Least Squares)
========================================================================================
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------
# STEP 1: LOAD AND INSPECT THE DATASET
# ---------------------------------------------------------
def load_and_inspect_data(filepath):
    print("=" * 70)
    print("STEP 1: LOADING AND INSPECTING DATASET")
    print("=" * 70)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nDataset Information:")
    df.info()
    
    print("\nFirst 5 Records:")
    print(df.head())
    
    print("\nSummary Statistics:")
    print(df.describe().round(2))
    
    print("\nMissing Values Count:")
    print(df.isnull().sum())
    
    print("\nDuplicate Records Count:", df.duplicated().sum())
    return df

# ---------------------------------------------------------
# STEP 2: FEATURE PREPARATION AND CORRELATION ANALYSIS
# ---------------------------------------------------------
def prepare_features(df):
    print("\n" + "=" * 70)
    print("STEP 2: FEATURE PREPARATION & CORRELATION ANALYSIS")
    print("=" * 70)
    
    # Independent variables (Features)
    feature_columns = ['Study_Hours', 'Attendance', 'Previous_Marks', 'Assignment_Score', 'Sleep_Hours']
    target_column = 'Final_Marks'
    
    X = df[feature_columns]
    y = df[target_column]
    
    print("Selected Independent Features (X):", feature_columns)
    print("Selected Target Variable (y):", target_column)
    
    # Correlation analysis
    corr = df.corr()
    print("\nCorrelation with Final Marks:")
    print(corr[target_column].sort_values(ascending=False).round(4))
    
    return X, y

# ---------------------------------------------------------
# STEP 3: TRAIN-TEST SPLITTING
# ---------------------------------------------------------
def split_dataset(X, y, test_size=0.2, random_state=42):
    print("\n" + "=" * 70)
    print("STEP 3: TRAIN-TEST SPLITTING")
    print("=" * 70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training Set Records: {X_train.shape[0]} (80%)")
    print(f"Testing Set Records:  {X_test.shape[0]} (20%)")
    return X_train, X_test, y_train, y_test

# ---------------------------------------------------------
# STEP 4: MODEL TRAINING (MULTIPLE LINEAR REGRESSION)
# ---------------------------------------------------------
def train_linear_regression(X_train, y_train):
    print("\n" + "=" * 70)
    print("STEP 4: MODEL TRAINING VIA ORDINARY LEAST SQUARES")
    print("=" * 70)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print("Model Training Successfully Completed.")
    print(f"Calculated Intercept (b0): {model.intercept_:.4f}")
    print("\nCalculated Regression Coefficients (Weights):")
    for feature_name, coef_val in zip(X_train.columns, model.coef_):
        print(f"  - {feature_name:18s}: {coef_val:.4f}")
    
    eq = f"Final_Marks = {model.intercept_:.4f}"
    for feature_name, coef_val in zip(X_train.columns, model.coef_):
        eq += f" + ({coef_val:.4f} * {feature_name})"
    print("\nSynthesized Multiple Linear Regression Equation:")
    print(eq)
    
    return model

# ---------------------------------------------------------
# STEP 5: MODEL EVALUATION
# ---------------------------------------------------------
def evaluate_model(model, X_train, y_train, X_test, y_test):
    print("\n" + "=" * 70)
    print("STEP 5: MODEL EVALUATION ON UNSEEN TEST DATA")
    print("=" * 70)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    mae_train = mean_absolute_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)
    
    mae_test = mean_absolute_error(y_test, y_test_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, y_test_pred)
    
    print("Training Set Metrics:")
    print(f"  - Training Mean Absolute Error (MAE): {mae_train:.4f}")
    print(f"  - Training R-squared Score (R2):     {r2_train:.4f} ({r2_train*100:.2f}%)")
    
    print("\nTesting Set Metrics:")
    print(f"  - Testing Mean Absolute Error (MAE):  {mae_test:.4f}")
    print(f"  - Testing Mean Squared Error (MSE):   {mse_test:.4f}")
    print(f"  - Testing Root Mean Squared Error:    {rmse_test:.4f}")
    print(f"  - Testing R-squared Score (R2):      {r2_test:.4f} ({r2_test*100:.2f}%)")
    
    # Comparison sample table
    comp_df = pd.DataFrame({
        'Actual_Marks': y_test.values,
        'Predicted_Marks': y_test_pred.round(2),
        'Residual_Error': (y_test.values - y_test_pred).round(2)
    })
    print("\nSample 10 Actual vs. Predicted Test Records:")
    print(comp_df.head(10))
    
    return y_test_pred, comp_df

# ---------------------------------------------------------
# STEP 6: PREDICTION PIPELINE FOR NEW STUDENTS
# ---------------------------------------------------------
def predict_student_performance(model, study_hours, attendance, prev_marks, assignment_score, sleep_hours):
    input_data = pd.DataFrame([{
        'Study_Hours': float(study_hours),
        'Attendance': float(attendance),
        'Previous_Marks': float(prev_marks),
        'Assignment_Score': float(assignment_score),
        'Sleep_Hours': float(sleep_hours)
    }])
    
    raw_pred = model.predict(input_data)[0]
    capped_pred = min(max(raw_pred, 0.0), 100.0)
    
    if capped_pred >= 90.0:
        status = "Outstanding (Grade A+)"
    elif capped_pred >= 75.0:
        status = "Good / Meritorious (Grade A/B)"
    elif capped_pred >= 50.0:
        status = "Average / Pass (Grade C)"
    else:
        status = "Needs Remedial Support (At-Risk)"
        
    return round(raw_pred, 2), round(capped_pred, 2), status

# ---------------------------------------------------------
# MAIN EXECUTION ROUTINE
# ---------------------------------------------------------
if __name__ == '__main__':
    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        'dataset', 'student_performance_dataset.csv'
    )
    
    # Pipeline execution
    df = load_and_inspect_data(dataset_path)
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    model = train_linear_regression(X_train, y_train)
    y_test_pred, comp_df = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # Run test cases
    print("\n" + "=" * 70)
    print("STEP 7: SYSTEM TEST SUITE VERIFICATION (5 FORMAL CASES)")
    print("=" * 70)
    test_suite = [
        ("TC1: High Performer", 8.5, 95.0, 92.0, 94.0, 7.5),
        ("TC2: Average Student", 5.0, 78.0, 70.0, 72.0, 6.5),
        ("TC3: Low Attendance / High Study", 7.5, 65.0, 75.0, 80.0, 6.0),
        ("TC4: At-Risk Student", 2.0, 62.0, 45.0, 55.0, 5.0),
        ("TC5: Steady Consistent Student", 6.0, 88.0, 82.0, 85.0, 8.0)
    ]
    
    for case_name, sh, att, pm, ascore, slh in test_suite:
        raw_p, capped_p, cat = predict_student_performance(model, sh, att, pm, ascore, slh)
        print(f"\n{case_name}")
        print(f"  Inputs: Study={sh}h, Att={att}%, Prev={pm}, Assign={ascore}, Sleep={slh}h")
        print(f"  Predicted Marks: {capped_p} / 100 (Raw Model Output: {raw_p}) | Classification: {cat}")
