import os
import time

import numpy as np
from core.data_utils import load_and_preprocess_data, create_batches
from core.models import get_models
from core.evaluation import (
    evaluate_traditional,
    evaluate_cross_validation,
    evaluate_partial_fit,
    evaluate_warm_start,
    evaluate_init_model_continuation
)
from core.visualizations import generate_comparative_analysis
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":
    # Create directories for results
    os.makedirs('./comparison_results', exist_ok=True)

    # Load and preprocess data
    X_train, X_test, y_train, y_test, class_names, feature_transformer, target_encoder = load_and_preprocess_data(
        './datasets/adult/adult_full.csv', target_column='income'
    )

    target_classes = np.unique(y_train)
    # Create batches for incremental learning
    batch_indices = create_batches(X_train, y_train)

    # Get models
    models = get_models()

    # Run evaluations
    results = []
    cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    start_time = time.time()

    for model_name, model in models:
        print(f"\nRunning evaluations for: {model_name}")

        # 1. Traditional evaluation
        traditional_result = evaluate_traditional(model_name, model, X_train, y_train, X_test, y_test, class_names)
        if traditional_result:
            results.append(traditional_result)

        # 2. Cross-validation
        cv_result = evaluate_cross_validation(model_name, model, X_train, y_train, cv_strategy, class_names)
        if cv_result:
            results.append(cv_result)

        # 3. Partial fit (incremental learning)
        partial_fit_result = evaluate_partial_fit(model_name, model, X_train, y_train, X_test, y_test, batch_indices, class_names, target_classes)
        if partial_fit_result:
            results.append(partial_fit_result)

        # 4. Warm start
        warm_start_result = evaluate_warm_start(model_name, model, X_train, y_train, X_test, y_test, class_names)
        if warm_start_result:
            results.append(warm_start_result)

        # 5. Init model continuation
        init_model_result = evaluate_init_model_continuation(model_name, model, X_train, y_train, X_test, y_test, batch_indices, class_names)
        if init_model_result:
            results.append(init_model_result)

    
    # Generate analysis and visualizations
    generate_comparative_analysis(results)
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print("\nAll evaluations complete. Results saved to ./comparison_results/ directory")