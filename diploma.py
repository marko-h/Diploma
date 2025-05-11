import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

# Traditional models
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB

# Gradient boosting libraries
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Create directory for model snapshots and results
os.makedirs('./model_snapshots', exist_ok=True)
os.makedirs('./comparison_results', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────────
# Data Preparation
# ─────────────────────────────────────────────────────────────────────────────────

print("Loading dataset and preprocessing features...")
columns = [
    'age','workclass','fnlwgt','education','education-num','marital-status',
    'occupation','relationship','race','sex','capital-gain','capital-loss',
    'hours-per-week','native-country','income'
]
df = pd.read_csv('./datasets/adult/adult_full.csv', skipinitialspace=True)
features = df.drop('income', axis=1)
target = df['income']

# Identify feature types
categorical_features = features.select_dtypes(include=['object']).columns
numeric_features = features.select_dtypes(include=['int64','float64']).columns

# Preprocessing pipeline
feature_transformer = ColumnTransformer([
    ('numeric_scaler', StandardScaler(), numeric_features),
    ('categorical_encoder', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Encode target variable
target_encoder = LabelEncoder()
target_encoded = target_encoder.fit_transform(target)
target_classes = np.unique(target_encoded)
class_names = target_encoder.classes_

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    features, target_encoded, test_size=0.3, random_state=42, stratify=target_encoded
)

# Transform features
X_train = feature_transformer.fit_transform(X_train)
X_test = feature_transformer.transform(X_test)

# Create batches for simulating incremental learning scenarios
kfold_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
batch_indices = [test_idx for _, test_idx in kfold_splitter.split(X_train, y_train)]

# ─────────────────────────────────────────────────────────────────────────────────
# Model Definitions - Wide range of model types, not just neural networks
# ─────────────────────────────────────────────────────────────────────────────────

models = [
    # Traditional sklearn models
    ("MLPClassifier",       MLPClassifier(hidden_layer_sizes=(128,32), max_iter=50, random_state=42)),
    ("LogisticRegression",  LogisticRegression(solver='saga', warm_start=True, max_iter=500, random_state=42)),
    ("KNeighborsClassifier", KNeighborsClassifier(n_neighbors=5)),
    ("DecisionTreeClassifier", DecisionTreeClassifier(random_state=42)),
    
    # Models with partial_fit support
    ("SGDClassifier",       SGDClassifier(max_iter=500, random_state=42)),
    ("PassiveAggressive",   PassiveAggressiveClassifier(max_iter=500, random_state=42)),
    ("BernoulliNB",         BernoulliNB()), 
    #("MultinomialNB",       MultinomialNB()),
    
    # Models with warm_start support
    ("RandomForestClassifier", RandomForestClassifier(n_estimators=50, warm_start=True, random_state=42)),
    ("GradientBoostingClassifier", GradientBoostingClassifier(n_estimators=50, warm_start=True, random_state=42)),
    
    # Boosting libraries with init_model capability
    ("XGBoost",             XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric='mlogloss', random_state=42)),
    ("LightGBM",            LGBMClassifier(n_estimators=50, random_state=42)),
    ("CatBoost",            CatBoostClassifier(iterations=50, verbose=0, random_state=42))
]

# ─────────────────────────────────────────────────────────────────────────────────
# Learning Paradigm Evaluations - Compare different approaches 
# ─────────────────────────────────────────────────────────────────────────────────

def evaluate_traditional(model_name, model, X_train, y_train, X_test, y_test):
    """
    Evaluate traditional batch learning (control case)
    """
    print(f"\n=== Traditional Learning: {model_name} ===")
    start_time = time.time()
    
    # Fit on full training data at once
    model_instance = model.__class__(**model.get_params())
    model_instance.fit(X_train, y_train)
    
    # Evaluate
    predictions = model_instance.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')
    training_time = time.time() - start_time
    
    print(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Time: {training_time:.2f}s")
    print(classification_report(y_test, predictions, target_names=class_names))
    
    return {
        'model': model_name,
        'learning_type': 'Traditional',
        'accuracy': accuracy,
        'f1_score': f1,
        'training_time': training_time
    }

def evaluate_cross_validation(model_name, model, X_train, y_train, cv_strategy):
    """
    Evaluate using k-fold cross-validation and return the best model
    """
    print(f"\n=== Cross-Validation: {model_name} ===")
    start_time = time.time()
    
    # Run cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, n_jobs=-1)
    cv_predictions = cross_val_predict(model, X_train, y_train, cv=cv_strategy, n_jobs=-1)
    print(classification_report(y_train, cv_predictions, target_names=class_names))
    # Evaluate CV performance
    avg_accuracy = np.mean(cv_scores)
    cv_f1 = f1_score(y_train, cv_predictions, average='weighted')
    cv_time = time.time() - start_time
    
    print(f"CV Accuracies: {[f'{s:.3f}' for s in cv_scores]}")
    print(f"Average Accuracy: {avg_accuracy:.4f}, F1: {cv_f1:.4f}, Time: {cv_time:.2f}s")
    
    return {
        'model': model_name,
        'learning_type': 'CrossValidation',
        'accuracy': avg_accuracy, # CV accuracy, not test accuracy
        'f1_score': cv_f1,
        'training_time': cv_time,
        'cv_scores': cv_scores.tolist(),
    }

def evaluate_partial_fit(model_name, model, X_train, y_train, X_test, y_test, batch_indices):
    """
    Evaluate incremental learning using partial_fit
    """
    if not hasattr(model, "partial_fit"):
        print(f"\n=== Partial Fit: {model_name} - NOT SUPPORTED ===")
        return None
    
    print(f"\n=== Partial Fit: {model_name} ===")
    start_time = time.time()
    
    # Create a new model instance
    incremental_model = model.__class__(**model.get_params())
    accuracies = []
    kfold_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    batch_indices = [test_idx for _, test_idx in kfold_splitter.split(X_train, y_train)]
    
    # Train incrementally on batches
    for i, batch_idx in enumerate(batch_indices, 1):
        batch_features = X_train[batch_idx]
        batch_targets = y_train[batch_idx]
        
        # For first batch with MultinomialNB, ensure non-negative features
        if model_name == "MultinomialNB" and i == 1:
            if hasattr(batch_features, "toarray"):  # If sparse
                # Convert to dense and ensure non-negative
                dense_features = batch_features.toarray()
                dense_features[dense_features < 0] = 0
                incremental_model.partial_fit(dense_features, batch_targets, classes=target_classes)
            else:
                batch_features[batch_features < 0] = 0
                incremental_model.partial_fit(batch_features, batch_targets, classes=target_classes)
        else:
            incremental_model.partial_fit(batch_features, batch_targets, classes=target_classes)
        
        # Evaluate after each batch
        batch_predictions = incremental_model.predict(X_test)
        batch_accuracy = accuracy_score(y_test, batch_predictions)
        accuracies.append(batch_accuracy)
        print(f"  Batch {i}/{len(batch_indices)}: Accuracy = {batch_accuracy:.4f}")
    
    # Final evaluation
    final_predictions = incremental_model.predict(X_test)
    final_accuracy = accuracy_score(y_test, final_predictions)
    final_f1 = f1_score(y_test, final_predictions, average='weighted')
    training_time = time.time() - start_time
    
    print(f"Final Accuracy: {final_accuracy:.4f}, F1: {final_f1:.4f}, Time: {training_time:.2f}s")
    print(classification_report(y_test, final_predictions, target_names=class_names))
    
    return {
        'model': model_name,
        'learning_type': 'PartialFit',
        'accuracy': final_accuracy,
        'f1_score': final_f1,
        'training_time': training_time,
        'accuracy_curve': accuracies
    }

def evaluate_warm_start(model_name, model, X_train, y_train, X_test, y_test):
    """
    Evaluate incremental learning using warm_start
    """
    warm_start_models = [
        "MLPClassifier", "LogisticRegression", "RandomForestClassifier", 
        "GradientBoostingClassifier", "PassiveAggressive"
    ]
    
    if model_name not in warm_start_models and not getattr(model, "warm_start", False):
        print(f"\n=== Warm Start: {model_name} - NOT SUPPORTED ===")
        return None
    
    print(f"\n=== Warm Start: {model_name} ===")
    start_time = time.time()
    
    # Create a new model instance with warm_start=True and limited iterations
    warm_model = model.__class__(**model.get_params())
    
    # Set appropriate parameter for limited learning per iteration
    if hasattr(warm_model, "max_iter"):
        warm_model.set_params(max_iter=1, warm_start=True)
    elif hasattr(warm_model, "n_estimators"):
        # For tree-based methods, increase trees gradually
        initial_estimators = 10
        warm_model.set_params(n_estimators=initial_estimators, warm_start=True)
    
    accuracies = []
    epochs = 10
    
    # Train incrementally
    for epoch in range(epochs):
        if hasattr(warm_model, "n_estimators") and epoch > 0:
            # Increase trees for next iteration
            warm_model.set_params(n_estimators=initial_estimators * (epoch + 1))
            
        warm_model.fit(X_train, y_train)
        
        # Evaluate after each epoch
        epoch_predictions = warm_model.predict(X_test)
        epoch_accuracy = accuracy_score(y_test, epoch_predictions)
        accuracies.append(epoch_accuracy)
        print(f"  Epoch {epoch+1}/{epochs}: Accuracy = {epoch_accuracy:.4f}")
    
    # Final evaluation
    final_predictions = warm_model.predict(X_test)
    final_accuracy = accuracy_score(y_test, final_predictions)
    final_f1 = f1_score(y_test, final_predictions, average='weighted')
    training_time = time.time() - start_time
    
    print(f"Final Accuracy: {final_accuracy:.4f}, F1: {final_f1:.4f}, Time: {training_time:.2f}s")
    print(classification_report(y_test, final_predictions, target_names=class_names))
    
    return {
        'model': model_name,
        'learning_type': 'WarmStart',
        'accuracy': final_accuracy,
        'f1_score': final_f1,
        'training_time': training_time,
        'accuracy_curve': accuracies
    }

def evaluate_init_model_continuation(model_name, model, X_train, y_train, X_test, y_test, batch_indices):
    """
    Evaluate continuation learning with init_model or similar (for boosting libraries)
    """
    if model_name not in ["XGBoost", "LightGBM", "CatBoost"]:
        return None
    
    print(f"\n=== Init Model Continuation: {model_name} ===")
    start_time = time.time()

    # Use StratifiedKFold for stratified batches
    kfold_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    batch_indices = [test_idx for _, test_idx in kfold_splitter.split(X_train, y_train)]

    # Get first batch for initial training
    initial_features = X_train[batch_indices[0]]
    initial_targets = y_train[batch_indices[0]]
    
    # Create and train initial model
    if model_name == "XGBoost":
        initial_model = XGBClassifier(n_estimators=20, use_label_encoder=False, eval_metric='mlogloss', random_state=42, learning_rate=0.1)
        model_path = f"./model_snapshots/{model_name}_model.json"
    elif model_name == "LightGBM":
        initial_model = LGBMClassifier(n_estimators=20, random_state=42, learning_rate=0.05)
        model_path = f"./model_snapshots/{model_name}_model.txt"
    else:  # CatBoost
        initial_model = CatBoostClassifier(iterations=20, verbose=0, random_state=42, learning_rate=0.05)
        model_path = f"./model_snapshots/{model_name}_model.cbm"
    
    # Train initial model on first batch
    initial_model.fit(initial_features, initial_targets)
    
    # Save model
    if model_name == "XGBoost":
        initial_model.save_model(model_path)
    elif model_name == "LightGBM":
        initial_model.booster_.save_model(model_path)
    else:  # CatBoost
        initial_model.save_model(model_path)
    
    # Evaluate initial model
    initial_predictions = initial_model.predict(X_test)
    initial_accuracy = accuracy_score(y_test, initial_predictions)
    accuracies = [initial_accuracy]
    print(f"  Batch 1/{len(batch_indices)}: Accuracy = {initial_accuracy:.4f}")
    
    # Train on remaining batches
    for i, batch_idx in enumerate(batch_indices[1:], 2):
        batch_features = X_train[batch_idx]
        batch_targets = y_train[batch_idx]
        
        if model_name == "XGBoost":
            # Load and continue training
            continued_model = XGBClassifier()
            continued_model.load_model(model_path)
            continued_model.fit(batch_features, batch_targets, xgb_model=model_path)
            continued_model.save_model(model_path)
        
        elif model_name == "LightGBM":
            # For LightGBM, need to specify n_estimators for continued model
            continued_model = LGBMClassifier(n_estimators=20*i)
            continued_model.fit(batch_features, batch_targets, init_model=model_path)
            continued_model.booster_.save_model(model_path)
        
        else:  # CatBoost
            continued_model = CatBoostClassifier()
            continued_model.load_model(model_path)
            continued_model.fit(batch_features, batch_targets, init_model=model_path)
            continued_model.save_model(model_path)
        
        # Evaluate model after this batch
        batch_predictions = continued_model.predict(X_test)
        batch_accuracy = accuracy_score(y_test, batch_predictions)
        accuracies.append(batch_accuracy)
        print(f"  Batch {i}/{len(batch_indices)}: Accuracy = {batch_accuracy:.4f}")
    
    # Load final model for evaluation
    if model_name == "XGBoost":
        final_model = XGBClassifier()
        final_model.load_model(model_path)
    elif model_name == "LightGBM":
        final_model = LGBMClassifier()
        final_model.fit(initial_features, initial_targets, init_model=model_path)
    else:
        final_model = CatBoostClassifier()
        final_model.load_model(model_path)
    
    # Final evaluation
    final_predictions = final_model.predict(X_test)
    final_accuracy = accuracy_score(y_test, final_predictions)
    final_f1 = f1_score(y_test, final_predictions, average='weighted')
    training_time = time.time() - start_time
    
    print(f"Final Accuracy: {final_accuracy:.4f}, F1: {final_f1:.4f}, Time: {training_time:.2f}s")
    print(classification_report(y_test, final_predictions, target_names=class_names))
    
    return {
        'model': model_name,
        'learning_type': 'InitModel',
        'accuracy': final_accuracy,
        'f1_score': final_f1,
        'training_time': training_time,
        'accuracy_curve': accuracies
    }

# ─────────────────────────────────────────────────────────────────────────────────
# Main Evaluation Loop - Run all evaluations
# ─────────────────────────────────────────────────────────────────────────────────

def run_model_evaluations():
    """Run all model evaluations and collect results"""
    print("\n" + "="*80)
    print("Starting Model Evaluations - Comparing Learning Approaches")
    print("="*80)
    
    results = []
    cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    for model_name, model in models:
        model_results = []
        
        # 1. Traditional (control) training
        result = evaluate_traditional(model_name, model, 
                                     X_train, y_train, 
                                     X_test, y_test)
        if result:
            model_results.append(result)
            results.append(result)
        
        # 2. Cross-validation (FIXED)
        cv_result = evaluate_cross_validation(model_name, model, 
                                             X_train, y_train, cv10)
        if cv_result:
            model_results.append(cv_result)
            results.append(cv_result)
        
        # 3. Partial Fit (incremental learning)
        result = evaluate_partial_fit(model_name, model, 
                                    X_train, y_train, 
                                    X_test, y_test, 
                                    batch_indices)
        if result:
            model_results.append(result)
            results.append(result)
        
        # 4. Warm Start (incremental learning)
        result = evaluate_warm_start(model_name, model, 
                                   X_train, y_train, 
                                   X_test, y_test)
        if result:
            model_results.append(result)
            results.append(result)
        
        # 5. Init Model (continuation learning)
        result = evaluate_init_model_continuation(model_name, model, 
                                                X_train, y_train, 
                                                X_test, y_test, 
                                                batch_indices)
        if result:
            model_results.append(result)
            results.append(result)
    
    return results

# ─────────────────────────────────────────────────────────────────────────────────
# Analysis and Visualization - Compare results
# ─────────────────────────────────────────────────────────────────────────────────

def generate_comparative_analysis(all_results):
    """Generate comparative analysis of all results"""
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save raw results
    results_df.to_csv('./comparison_results/all_results.csv', index=False)
    
    # Create a capability matrix showing which model supports which learning approach
    capabilities = {}
    capabilities['Model'] = list(results_df['model'].unique())
    
    for learning_type in ['Traditional', 'CrossValidation', 'PartialFit', 'WarmStart', 'InitModel']:
        temp_df = results_df[results_df['learning_type'] == learning_type]
        supported_models = temp_df['model'].unique()
        
        capabilities[learning_type] = [
            '✓' if model in supported_models else '✗' 
            for model in capabilities['Model']
        ]
    
    capabilities_df = pd.DataFrame(capabilities)
    capabilities_df.to_csv('./comparison_results/model_capabilities.csv', index=False)
    print("\nModel capability matrix saved to ./comparison_results/model_capabilities.csv")
    
    # Generate comparative visualizations
    try:
        # Accuracy comparison across learning types
        plt.figure(figsize=(14, 10))
        
        # Create pivot table for easier plotting
        pivot_df = results_df.pivot_table(
            values='accuracy',
            index='model',
            columns='learning_type'
        )
        
        # Sort by traditional accuracy for better visualization
        if 'Traditional' in pivot_df.columns:
            pivot_df = pivot_df.sort_values('Traditional', ascending=False)
            
        # Plot
        pivot_df.plot(kind='barh', ax=plt.gca())
        plt.title('Accuracy by Model and Learning Type')
        plt.xlabel('Accuracy')
        plt.ylabel('Model')
        plt.grid(True, axis='x')
        plt.tight_layout()
        plt.savefig('./comparison_results/accuracy_comparison.png')
        
        # Training time comparison
        plt.figure(figsize=(14, 10))
        time_pivot = results_df.pivot_table(
            values='training_time',
            index='model',
            columns='learning_type'
        )
        
        if 'Traditional' in time_pivot.columns:
            time_pivot = time_pivot.sort_values('Traditional', ascending=True)
            
        time_pivot.plot(kind='barh', ax=plt.gca())
        plt.title('Training Time by Model and Learning Type')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Model')
        plt.grid(True, axis='x')
        plt.tight_layout()
        plt.savefig('./comparison_results/training_time_comparison.png')
        
        # Plot learning curves for incremental approaches
        for learning_type in ['PartialFit', 'WarmStart', 'InitModel']:
            curves_df = results_df[results_df['learning_type'] == learning_type]
            curves_df = curves_df[curves_df['accuracy_curve'].notnull()]
            
            if len(curves_df) > 0:
                plt.figure(figsize=(10, 6))
                
                for _, row in curves_df.iterrows():
                    model_name = row['model']
                    accuracy_curve = row['accuracy_curve']
                    iterations = list(range(1, len(accuracy_curve) + 1))
                    plt.plot(iterations, accuracy_curve, marker='o', label=model_name)
                
                plt.title(f'Learning Curves - {learning_type}')
                plt.xlabel('Iteration')
                plt.ylabel('Accuracy')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.savefig(f'./comparison_results/learning_curve_{learning_type}.png')
        
        print("Visualizations saved to ./comparison_results/ directory")
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")
    
    # Summary statistics
    print("\nSummary of results by learning type:")
    summary = results_df.groupby('learning_type').agg({
        'accuracy': ['mean', 'std', 'min', 'max'],
        'f1_score': ['mean', 'std', 'min', 'max'],
        'training_time': ['mean', 'std', 'min', 'max']
    })
    print(summary)
    
    # Best models by approach
    print("\nBest models by learning approach:")
    for learning_type in results_df['learning_type'].unique():
        type_df = results_df[results_df['learning_type'] == learning_type]
        best_model_idx = type_df['accuracy'].idxmax()
        best_model = type_df.loc[best_model_idx]
        print(f"{learning_type}: {best_model['model']} (Accuracy: {best_model['accuracy']:.4f})")

# ─────────────────────────────────────────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    start_time = time.time()
    
    # Run all evaluations
    results = run_model_evaluations()
    
    # Analyze and visualize results
    generate_comparative_analysis(results)
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print("\nAll evaluations complete. Results saved to ./comparison_results/ directory")