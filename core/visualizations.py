import pandas as pd
import matplotlib.pyplot as plt

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