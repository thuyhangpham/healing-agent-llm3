#!/usr/bin/env python3
"""
NLP Performance Evaluation Script

Evaluates the accuracy of the Sentiment Analysis Agent by comparing
system predictions against human ground truth labels.

Usage:
    python scripts/evaluate_nlp_performance.py --generate  # Create template for manual labeling
    python scripts/evaluate_nlp_performance.py --eval       # Calculate metrics after labeling
"""

import sys
import os
import json
import csv
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import Counter

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  Warning: scikit-learn not available. Install with: pip install scikit-learn")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️  Warning: matplotlib/seaborn not available. Confusion matrix will be text-only.")


class NLPEvaluator:
    """Evaluator for NLP sentiment analysis performance"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.opinions_dir = self.project_root / "data" / "production" / "opinions"
        self.evaluation_dir = self.project_root / "data" / "evaluation"
        self.reports_dir = self.project_root / "data" / "reports"
        
        self.ground_truth_file = self.evaluation_dir / "ground_truth.csv"
        self.metrics_file = self.reports_dir / "nlp_performance_metrics.json"
        
        # Ensure directories exist
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Sentiment labels (must match system labels)
        self.sentiment_labels = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
    
    def generate_ground_truth_template(self) -> bool:
        """
        Generate a ground truth template CSV file for manual labeling.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            print("="*80)
            print("GENERATING GROUND TRUTH TEMPLATE")
            print("="*80)
            print(f"Scanning opinion articles in: {self.opinions_dir}")
            
            # Find all JSON files
            opinion_files = list(self.opinions_dir.glob("*.json"))
            
            if not opinion_files:
                print(f"❌ No opinion articles found in {self.opinions_dir}")
                return False
            
            print(f"Found {len(opinion_files)} opinion articles")
            
            # Prepare data for CSV
            rows = []
            analyzed_count = 0
            not_analyzed_count = 0
            
            for article_file in opinion_files:
                try:
                    with open(article_file, 'r', encoding='utf-8') as f:
                        article = json.load(f)
                    
                    article_id = article.get('id', article_file.stem)
                    title = article.get('title', 'No title')
                    system_prediction = article.get('sentiment_label', 'NOT_ANALYZED')
                    
                    if system_prediction == 'NOT_ANALYZED':
                        not_analyzed_count += 1
                    else:
                        analyzed_count += 1
                    
                    rows.append({
                        'article_id': article_id,
                        'title': title,
                        'system_prediction': system_prediction,
                        'human_label': '',  # Empty for user to fill
                        'is_verified': 'False'
                    })
                    
                except Exception as e:
                    print(f"⚠️  Warning: Failed to process {article_file.name}: {e}")
                    continue
            
            # Write CSV file
            print(f"\nWriting ground truth template to: {self.ground_truth_file}")
            print(f"   Analyzed articles: {analyzed_count}")
            print(f"   Not analyzed articles: {not_analyzed_count}")
            
            with open(self.ground_truth_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'article_id',
                    'title',
                    'system_prediction',
                    'human_label',
                    'is_verified'
                ])
                writer.writeheader()
                writer.writerows(rows)
            
            print(f"\n✅ Ground truth template created successfully!")
            print(f"   Total articles: {len(rows)}")
            print(f"   File: {self.ground_truth_file}")
            print(f"\n📝 Next steps:")
            print(f"   1. Open the CSV file in Excel or a text editor")
            print(f"   2. Fill in the 'human_label' column with: POSITIVE, NEGATIVE, or NEUTRAL")
            print(f"   3. Set 'is_verified' to 'True' for verified labels")
            print(f"   4. Run: python scripts/evaluate_nlp_performance.py --eval")
            
            return True
            
        except Exception as e:
            print(f"❌ Error generating ground truth template: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def calculate_metrics(self) -> bool:
        """
        Calculate performance metrics from ground truth labels.
        
        Returns:
            True if successful, False otherwise
        """
        if not SKLEARN_AVAILABLE:
            print("❌ scikit-learn is required for metrics calculation")
            print("   Install with: pip install scikit-learn")
            return False
        
        try:
            print("="*80)
            print("CALCULATING NLP PERFORMANCE METRICS")
            print("="*80)
            
            # Check if ground truth file exists
            if not self.ground_truth_file.exists():
                print(f"❌ Ground truth file not found: {self.ground_truth_file}")
                print(f"   Run with --generate first to create the template")
                return False
            
            # Read ground truth CSV
            print(f"Reading ground truth file: {self.ground_truth_file}")
            rows = []
            with open(self.ground_truth_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
            
            # Filter rows with human labels
            labeled_rows = [
                row for row in rows 
                if row.get('human_label', '').strip() and 
                   row.get('human_label', '').strip().upper() in self.sentiment_labels
            ]
            
            if not labeled_rows:
                print(f"❌ No labeled data found in ground truth file")
                print(f"   Please fill in the 'human_label' column with POSITIVE, NEGATIVE, or NEUTRAL")
                return False
            
            print(f"Found {len(labeled_rows)} labeled articles out of {len(rows)} total")
            
            # Extract predictions and labels
            y_true = []
            y_pred = []
            article_ids = []
            
            for row in labeled_rows:
                human_label = row['human_label'].strip().upper()
                system_pred = row['system_prediction'].strip().upper()
                
                # Skip if system prediction is NOT_ANALYZED
                if system_pred == 'NOT_ANALYZED':
                    continue
                
                # Validate labels
                if human_label not in self.sentiment_labels:
                    print(f"⚠️  Warning: Invalid human label '{human_label}' for article {row['article_id']}, skipping")
                    continue
                
                if system_pred not in self.sentiment_labels:
                    print(f"⚠️  Warning: Invalid system prediction '{system_pred}' for article {row['article_id']}, skipping")
                    continue
                
                y_true.append(human_label)
                y_pred.append(system_pred)
                article_ids.append(row['article_id'])
            
            if not y_true:
                print(f"❌ No valid labeled data after filtering")
                return False
            
            print(f"Evaluating {len(y_true)} valid labeled articles")
            
            # Calculate metrics
            print("\n" + "="*80)
            print("PERFORMANCE METRICS")
            print("="*80)
            
            # Accuracy
            accuracy = accuracy_score(y_true, y_pred)
            print(f"\n📊 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            # Precision (Macro & Weighted)
            precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
            precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            print(f"📊 Precision (Macro): {precision_macro:.4f}")
            print(f"📊 Precision (Weighted): {precision_weighted:.4f}")
            
            # Recall (Macro & Weighted)
            recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
            recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            print(f"📊 Recall (Macro): {recall_macro:.4f}")
            print(f"📊 Recall (Weighted): {recall_weighted:.4f}")
            
            # F1-Score (Macro & Weighted)
            f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
            f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            print(f"📊 F1-Score (Macro): {f1_macro:.4f}")
            print(f"📊 F1-Score (Weighted): {f1_weighted:.4f}")
            
            # Per-class metrics
            print("\n" + "="*80)
            print("PER-CLASS METRICS")
            print("="*80)
            print(classification_report(y_true, y_pred, target_names=self.sentiment_labels, zero_division=0))
            
            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred, labels=self.sentiment_labels)
            print("\n" + "="*80)
            print("CONFUSION MATRIX")
            print("="*80)
            self._print_confusion_matrix(cm, self.sentiment_labels)
            
            # Save confusion matrix visualization if possible
            if PLOTTING_AVAILABLE:
                self._save_confusion_matrix_plot(cm, self.sentiment_labels)
            
            # Prepare metrics dictionary
            metrics = {
                'evaluation_timestamp': datetime.now().isoformat(),
                'total_labeled_articles': len(y_true),
                'accuracy': float(accuracy),
                'precision': {
                    'macro': float(precision_macro),
                    'weighted': float(precision_weighted)
                },
                'recall': {
                    'macro': float(recall_macro),
                    'weighted': float(recall_weighted)
                },
                'f1_score': {
                    'macro': float(f1_macro),
                    'weighted': float(f1_weighted)
                },
                'confusion_matrix': {
                    'labels': self.sentiment_labels,
                    'matrix': cm.tolist()
                },
                'per_class_metrics': self._calculate_per_class_metrics(y_true, y_pred, self.sentiment_labels)
            }
            
            # Save metrics to JSON
            print(f"\n💾 Saving metrics to: {self.metrics_file}")
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Metrics calculation completed successfully!")
            print(f"   Metrics saved to: {self.metrics_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error calculating metrics: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _print_confusion_matrix(self, cm, labels: List[str]):
        """Print confusion matrix in a readable format"""
        print("\nPredicted →")
        print("Actual ↓", end="")
        for label in labels:
            print(f"\t{label[:8]}", end="")
        print()
        
        for i, label in enumerate(labels):
            print(f"{label[:8]}", end="")
            for j in range(len(labels)):
                print(f"\t{cm[i][j]:4d}", end="")
            print()
        
        print("\nLegend:")
        print("  Rows = Actual labels (Ground Truth)")
        print("  Columns = Predicted labels (System)")
    
    def _save_confusion_matrix_plot(self, cm, labels: List[str]):
        """Save confusion matrix as a visualization"""
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=labels,
                yticklabels=labels,
                cbar_kws={'label': 'Count'}
            )
            plt.title('Sentiment Analysis Confusion Matrix', fontsize=14, fontweight='bold')
            plt.ylabel('Actual Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.tight_layout()
            
            plot_file = self.reports_dir / "confusion_matrix.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"\n📊 Confusion matrix visualization saved to: {plot_file}")
            
        except Exception as e:
            print(f"⚠️  Warning: Failed to save confusion matrix plot: {e}")
    
    def _calculate_per_class_metrics(
        self, 
        y_true: List[str], 
        y_pred: List[str], 
        labels: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate per-class precision, recall, and F1-score"""
        per_class = {}
        
        for label in labels:
            # Calculate TP, FP, FN for this class
            tp = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred == label)
            fp = sum(1 for true, pred in zip(y_true, y_pred) if true != label and pred == label)
            fn = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred != label)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            per_class[label] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'support': int(tp + fn)  # Number of actual instances of this class
            }
        
        return per_class


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Evaluate NLP Sentiment Analysis Performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate ground truth template for manual labeling
  python scripts/evaluate_nlp_performance.py --generate
  
  # Calculate metrics after labeling
  python scripts/evaluate_nlp_performance.py --eval
        """
    )
    
    parser.add_argument(
        '--generate',
        action='store_true',
        help='Generate ground truth template CSV for manual labeling'
    )
    
    parser.add_argument(
        '--eval',
        action='store_true',
        help='Calculate performance metrics from labeled ground truth'
    )
    
    args = parser.parse_args()
    
    if not args.generate and not args.eval:
        parser.print_help()
        print("\n❌ Error: Must specify either --generate or --eval")
        sys.exit(1)
    
    evaluator = NLPEvaluator()
    
    if args.generate:
        success = evaluator.generate_ground_truth_template()
        sys.exit(0 if success else 1)
    
    if args.eval:
        success = evaluator.calculate_metrics()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

