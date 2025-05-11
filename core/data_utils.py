import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

def load_and_preprocess_data(filepath, target_column):
    """
    Load dataset, preprocess features, and split into train/test sets.
    """
    print("Loading dataset and preprocessing features...")
    
    # Define column names (if needed)
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country', 'income'
    ]
    
    # Load dataset
    df = pd.read_csv(filepath, skipinitialspace=True)
    
    # Define features and target
    features = df.drop(target_column, axis=1)
    target = df[target_column]
    
    # Identify feature types
    categorical_features = features.select_dtypes(include=['object']).columns
    numeric_features = features.select_dtypes(include=['int64', 'float64']).columns
    
    # Preprocessing pipeline
    feature_transformer = ColumnTransformer([
        ('numeric_scaler', StandardScaler(), numeric_features),
        ('categorical_encoder', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    
    # Encode target variable
    target_encoder = LabelEncoder()
    target_encoded = target_encoder.fit_transform(target)
    class_names = target_encoder.classes_
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        features, target_encoded, test_size=0.3, random_state=42, stratify=target_encoded
    )
    
    # Transform features
    X_train = feature_transformer.fit_transform(X_train)
    X_test = feature_transformer.transform(X_test)
    
    return X_train, X_test, y_train, y_test, class_names, feature_transformer, target_encoder

def create_batches(X_train, y_train, n_splits=5):
    """
    Create batches for simulating incremental learning scenarios.
    """
    kfold_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    batch_indices = [test_idx for _, test_idx in kfold_splitter.split(X_train, y_train)]
    return batch_indices