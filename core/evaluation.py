import os
import time
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score, cross_val_predict
from xgboost import XGBClassifier

def evaluate_traditional(model_name, model, X_train, y_train, X_test, y_test, class_names):
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

def evaluate_cross_validation(model_name, model, X_train, y_train, cv_strategy, class_names):
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

def evaluate_partial_fit(model_name, model, X_train, y_train, X_test, y_test, batch_indices, class_names, target_classes):
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

def evaluate_warm_start(model_name, model, X_train, y_train, X_test, y_test , class_names):
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

def evaluate_init_model_continuation(model_name, model, X_train, y_train, X_test, y_test, batch_indices, class_names):
    """
    Evaluate continuation learning with init_model or similar (for boosting libraries)
    """
    if model_name not in ["XGBoost", "LightGBM", "CatBoost"]:
        return None
    
    print(f"\n=== Init Model Continuation: {model_name} ===")
    start_time = time.time()
    
    os.makedirs("./model_snapshots", exist_ok=True)
    # Get first batch for initial training
    initial_features = X_train[batch_indices[0]]
    initial_targets = y_train[batch_indices[0]]
    
    # Create and train initial model
    if model_name == "XGBoost":
        initial_model = XGBClassifier(n_estimators=20, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        model_path = f"./model_snapshots/{model_name}_model.json"
    elif model_name == "LightGBM":
        initial_model = LGBMClassifier(n_estimators=20, random_state=42)
        model_path = f"./model_snapshots/{model_name}_model.txt"
    else:  # CatBoost
        initial_model = CatBoostClassifier(iterations=20, verbose=0, random_state=42)
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