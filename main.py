"""
Health Risk Detection from Industrial Gas Exposure
===================================================
Training Script - Preprocesses data, trains multiple ML models,
evaluates performance, and saves the best model.

Author: Health Risk Detection Project
"""

# ============================================
# STEP 1: Import Required Libraries
# ============================================
import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Try to import XGBoost (optional)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
    print("XGBoost is available and will be used.")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed. Skipping XGBoost model.")

warnings.filterwarnings('ignore')

## Dataset generation removed for production use

# ============================================
# STEP 3: Load and Explore Dataset
# ============================================
def load_data(filepath):
    """
    Load the dataset from CSV file and print columns.
    """
    print("\n" + "="*50)
    print("Loading Dataset...")
    print("="*50)
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumn names: {list(df.columns)}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nBasic statistics:\n{df.describe()}")
    return df

# ============================================
# STEP 4: Data Preprocessing
# ============================================
def preprocess_data(df):
    """
    Preprocess the data for refinery risk prediction.
    - Handle missing values
    - Encode categorical columns
    - Scale numeric features
    - Split into train and test sets
    """
    print("\n" + "="*50)
    print("Preprocessing Data...")
    print("="*50)

    # Define required columns for exact disease prediction
    feature_columns = [
        'Industry_Type', 'Unit',
        'SO2_ppm', 'NOx_ppm', 'CO_ppm', 'H2S_ppm',
        'VOC_ppm', 'Benzene_ppm', 'Toluene_ppm', 'PM2_5_ugm3',
        'Exposure_Hours_Day', 'Exposure_Years'
    ]
    target_column = 'Disease'

    # Print columns and check for missing columns
    print(f"\nDataset columns: {list(df.columns)}")
    missing_features = [col for col in feature_columns if col not in df.columns]
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is missing from dataset!")
    if missing_features:
        print(f"WARNING: Missing feature columns: {missing_features}")
    else:
        print("All required feature columns are present.")

    # Drop rows with missing target
    before = len(df)
    df = df.dropna(subset=[target_column])
    after = len(df)
    if before != after:
        print(f"Dropped {before - after} rows with missing target label.")

    # Fill missing values
    categorical_cols = ['Industry_Type', 'Unit']
    numeric_cols = [col for col in feature_columns if col not in categorical_cols]
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median = df[col].median()
            df[col].fillna(median, inplace=True)
            print(f"Filled missing values in numeric column '{col}' with median: {median}")
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode = df[col].mode()[0]
            df[col].fillna(mode, inplace=True)
            print(f"Filled missing values in categorical column '{col}' with mode: {mode}")

    # Prepare X and y
    X = df[feature_columns].copy()
    y = df[target_column].copy()

    # Encode categorical features
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
        print(f"Label encoding for {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Encode target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"\nLabel encoding mapping for target:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"  {label} -> {i}")

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    print(f"\nFeatures scaled using StandardScaler for: {numeric_cols}")

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    print(f"\nTrain set size: {len(X_train)} (70%)")
    print(f"Test set size: {len(X_test)} (30%)")

    return X_train, X_test, y_train, y_test, scaler, label_encoder, feature_columns, encoders

# ============================================
# STEP 5: Train Multiple Models
# ============================================
def train_models(X_train, y_train):
    """
    Train multiple machine learning models:
    - Logistic Regression
    - Random Forest
    - XGBoost (if available)
    """
    print("\n" + "="*50)
    print("Training Models...")
    print("="*50)

    models = {}

    # Model 1: Logistic Regression
    print("\n[1] Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    models['Logistic Regression'] = lr_model
    print("    Logistic Regression trained successfully!")

    # Model 2: Random Forest
    print("\n[2] Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    models['Random Forest'] = rf_model
    print("    Random Forest trained successfully!")

    # Model 3: XGBoost (if available)
    if XGBOOST_AVAILABLE:
        print("\n[3] Training XGBoost...")
        xgb_model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        )
        xgb_model.fit(X_train, y_train)
        models['XGBoost'] = xgb_model
        print("    XGBoost trained successfully!")

    return models

# ============================================
# STEP 6: Evaluate Models
# ============================================
def evaluate_models(models, X_test, y_test, label_encoder):
    """
    Evaluate all trained models and compare their performance.
    Returns the best model based on accuracy.
    """
    print("\n" + "="*50)
    print("Evaluating Models...")
    print("="*50)

    results = {}
    best_accuracy = 0
    best_model_name = None
    best_model = None

    for name, model in models.items():
        print(f"\n{'='*40}")
        print(f"Model: {name}")
        print('='*40)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy

        print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

        # Confusion Matrix
        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        # Classification Report
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred,
                                    target_names=label_encoder.classes_))

        # Class distribution
        print(f"\nClass distribution in test set:")
        unique, counts = np.unique(y_test, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  {label_encoder.classes_[u]}: {c}")

        # Track best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            best_model = model

    # Summary
    print("\n" + "="*50)
    print("MODEL COMPARISON SUMMARY")
    print("="*50)
    for name, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {accuracy:.4f} ({accuracy*100:.2f}%)")

    print(f"\n*** Best Model: {best_model_name} with accuracy {best_accuracy:.4f} ***")

    return best_model, best_model_name, best_accuracy

# ============================================
# STEP 7: Save Model and Preprocessors
# ============================================
def save_model(model, scaler, label_encoder, feature_columns, encoders, filepath='model.pkl'):
    """
    Save the trained model, scaler, label encoder, and feature encoders.
    """
    print("\n" + "="*50)
    print("Saving Model...")
    print("="*50)
    model_data = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_columns': feature_columns,
        'encoders': encoders
    }
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    # Save scaler, encoders, label_encoder separately if needed
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('encoder.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Model saved to: {filepath}")
    print("Saved components:")
    print("  - Trained model")
    print("  - StandardScaler (scaler.pkl)")
    print("  - Feature encoders (encoder.pkl)")
    print("  - LabelEncoder (label_encoder.pkl)")
    print("  - Feature column names")

# ============================================
# STEP 8: Prediction Function
# ============================================
def predict_disease(model_data, co, no2, so2, o3, benzene, toluene, xylene, exposure_duration):
    """
    Predict disease based on gas exposure values.

    Parameters:
    - model_data: Dictionary containing model, scaler, and label_encoder
    - co: Carbon Monoxide level (ppm)
    - no2: Nitrogen Dioxide level (ppb)
    - so2: Sulfur Dioxide level (ppb)
    - o3: Ozone level (ppb)
    - benzene: Benzene level (ppb)
    - toluene: Toluene level (ppb)
    - xylene: Xylene level (ppb)
    - exposure_duration: Duration of exposure (hours)

    Returns:
    - predicted_disease: Name of predicted disease
    - confidence: Probability of the prediction
    - all_probabilities: Probabilities for all classes
    """
    # Extract components
    model = model_data['model']
    scaler = model_data['scaler']
    label_encoder = model_data['label_encoder']

    # Create input array
    input_data = np.array([[co, no2, so2, o3, benzene, toluene, xylene, exposure_duration]])

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Get probability (if available)
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(input_scaled)[0]
        confidence = probabilities[prediction]
        all_probs = {label_encoder.classes_[i]: prob
                     for i, prob in enumerate(probabilities)}
    else:
        confidence = None
        all_probs = None

    # Decode prediction to disease name
    predicted_disease = label_encoder.inverse_transform([prediction])[0]

    return predicted_disease, confidence, all_probs

# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  HEALTH RISK DETECTION FROM INDUSTRIAL GAS EXPOSURE")
    print("="*60)

    # Step 1: Load dataset
    dataset_path = 'Refinery_Exact_Disease_Dataset_100000.csv'
    df = load_data(dataset_path)

    # Step 2: Preprocess data
    X_train, X_test, y_train, y_test, scaler, label_encoder, feature_columns, encoders = preprocess_data(df)

    # Step 3: Train models
    models = train_models(X_train, y_train)

    # Step 4: Evaluate models and get the best one
    best_model, best_model_name, best_accuracy = evaluate_models(
        models, X_test, y_test, label_encoder
    )

    # Step 5: Save the best model and preprocessors
    save_model(best_model, scaler, label_encoder, feature_columns, encoders, 'model.pkl')

    print("\n" + "="*60)
    print("  TRAINING COMPLETE!")
    print("  Run 'streamlit run app.py' to launch the web app")
    print("="*60)
