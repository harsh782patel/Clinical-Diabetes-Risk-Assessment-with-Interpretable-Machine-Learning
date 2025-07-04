import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.metrics import (classification_report, roc_auc_score, 
                            confusion_matrix, roc_curve, precision_recall_curve)

def evaluate_model(model, X_test, y_test, model_name="Model", save_path="reports/figures/"):
    os.makedirs(save_path, exist_ok=True)

    # Generate predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Classification report
    print(f"\n{model_name} Performance:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-diabetic', 'Diabetic'],
                yticklabels=['Non-diabetic', 'Diabetic'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f"{save_path}/{model_name.lower()}_confusion_matrix.png", dpi=300)
    plt.close()
    
    # ROC Curve
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc_score(y_test, y_proba):.2f})', 
             linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig(f"{save_path}/{model_name.lower()}_roc_curve.png", dpi=300)
    plt.close()
    
    # Precision-Recall Curve
    plt.figure(figsize=(10, 6))
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.plot(recall, precision, label=model_name, linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.savefig(f"{save_path}/{model_name.lower()}_pr_curve.png", dpi=300)
    plt.close()
    
    # Feature Importance
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(12, 8))
        importance = pd.Series(model.feature_importances_, index=X_test.columns)
        importance.sort_values().plot(kind='barh', color='skyblue')
        plt.title(f'{model_name} Feature Importance')
        plt.xlabel('Importance Score')
        plt.savefig(f"{save_path}/{model_name.lower()}_feature_importance.png", dpi=300)
        plt.close()
    
    return {
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }

def compare_models(results_dict, save_path="reports/figures/"):
    """
    Compare multiple models' performance
    
    Args:
        results_dict: Dictionary of model evaluation results
        save_path: Path to save comparison plot
    """
    # Prepare comparison data
    metrics = ['precision', 'recall', 'f1-score', 'roc_auc']
    comparison = {}
    
    for model_name, results in results_dict.items():
        comparison[model_name] = {
            'precision': results['classification_report']['weighted avg']['precision'],
            'recall': results['classification_report']['weighted avg']['recall'],
            'f1-score': results['classification_report']['weighted avg']['f1-score'],
            'roc_auc': results['roc_auc']
        }
    
    # Plot comparison
    plt.figure(figsize=(14, 8))
    for metric in metrics:
        values = [comparison[model][metric] for model in comparison]
        plt.plot(list(comparison.keys()), values, 'o-', label=metric)
    
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.ylim(0.5, 1.0)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{save_path}/model_comparison.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    # Example usage
    from train import train_model
    
    # Train and evaluate model
    model, X_test, y_test = train_model('data/diabetes.csv')
    results = evaluate_model(model, X_test, y_test)
    