import time
import numpy as np
from dataclasses import dataclass
from data_loader import DataLoader
from hybrid_fraud_detector import HybridFraudDetector
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, 
    f1_score, confusion_matrix, roc_auc_score
)


@dataclass
class AlgorithmResult:
   # Stores metrics/results of different algorithms
    name: str
    accuracy: float
    recall: float
    precision: float
    f1_score: float
    roc_auc: float
    train_time_ms: float
    predict_time_ms: float
    avg_time_per_tx_ms: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    frauds_caught: int
    frauds_missed: int
    total_frauds: int


class FraudDetectionComparison:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.results = []
        
    def test_logistic_regression(self):
        # Train and test Logistic Regression
        print("\nTesting Logistic Regression...")
        
        logistic_regression = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
        )
        
        # Train and Predict
        # Record time it takes
        train_start = time.perf_counter()
        logistic_regression.fit(self.X_train, self.y_train)
        train_time = (time.perf_counter() - train_start) * 1000
        
        predict_start = time.perf_counter()
        predictions = logistic_regression.predict(self.X_test)
        predict_time = (time.perf_counter() - predict_start) * 1000
        
        # Get probabilities for ROC AUC
        probs = logistic_regression.predict_proba(self.X_test)[:, 1]
        roc_auc = roc_auc_score(self.y_test, probs)
        
        # Calculate metrics
        result = self.calculate_metrics(
            "Logistic Regression",
            predictions,
            train_time,
            predict_time,
            roc_auc
        )
        
        self.results.append(result)
        
        return result
    
    def test_decision_tree(self):
        # Train and test Decision Tree
        print("\nTesting Decision Tree...")
        
        decision_tree = DecisionTreeClassifier(
            max_depth=15,
            class_weight='balanced',
            random_state=42
        )
        
        # Train and Predict
        # Record time it takes
        train_start = time.perf_counter()
        decision_tree.fit(self.X_train, self.y_train)
        train_time = (time.perf_counter() - train_start) * 1000

        predict_start = time.perf_counter()
        predictions = decision_tree.predict(self.X_test)
        predict_time = (time.perf_counter() - predict_start) * 1000
        
        probs = decision_tree.predict_proba(self.X_test)[:, 1]
        roc_auc = roc_auc_score(self.y_test, probs)
        
        result = self.calculate_metrics(
            "Decision Tree",
            predictions,
            train_time,
            predict_time,
            roc_auc
        )
        
        self.results.append(result)
        
        return result
    
    def test_naive_bayes(self):
        # Train and test Naive Bayes
        print("\nTesting Naive Bayes...")
        
        naive_bayes = GaussianNB()
        
        # Train and Predict
        # Record time it takes
        train_start = time.perf_counter()
        naive_bayes.fit(self.X_train, self.y_train)
        train_time = (time.perf_counter() - train_start) * 1000
        
        predict_start = time.perf_counter()
        predictions = naive_bayes.predict(self.X_test)
        predict_time = (time.perf_counter() - predict_start) * 1000
        
        probs = naive_bayes.predict_proba(self.X_test)[:, 1]
        roc_auc = roc_auc_score(self.y_test, probs)
        
        result = self.calculate_metrics(
            "Naive Bayes",
            predictions,
            train_time,
            predict_time,
            roc_auc
        )
        
        self.results.append(result)
        
        return result
    
    def test_random_forest(self):
        # Train and test Random Forest
        print("\nTesting Random Forest...")
        
        random_forest = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Train and Predict
        # Record time it takes
        train_start = time.perf_counter()
        random_forest.fit(self.X_train, self.y_train)
        train_time = (time.perf_counter() - train_start) * 1000
        
        predict_start = time.perf_counter()
        predictions = random_forest.predict(self.X_test)
        predict_time = (time.perf_counter() - predict_start) * 1000
        
        probs = random_forest.predict_proba(self.X_test)[:, 1]
        roc_auc = roc_auc_score(self.y_test, probs)
        
        result = self.calculate_metrics(
            "Random Forest",
            predictions,
            train_time,
            predict_time,
            roc_auc
        )
        
        self.results.append(result)
        
        return result
    
    def test_gradient_boosting(self):
        # Train and test Gradient Boosting
        print("\nTesting Gradient Boosting...")
        
        gradient_boosting = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )

        # Train and Predict
        # Record time it takes
        train_start = time.perf_counter()
        gradient_boosting.fit(self.X_train, self.y_train)
        train_time = (time.perf_counter() - train_start) * 1000
        
        predict_start = time.perf_counter()
        predictions = gradient_boosting.predict(self.X_test)
        predict_time = (time.perf_counter() - predict_start) * 1000
        
        probs = gradient_boosting.predict_proba(self.X_test)[:, 1]
        roc_auc = roc_auc_score(self.y_test, probs)
        
        result = self.calculate_metrics(
            "Gradient Boosting",
            predictions,
            train_time,
            predict_time,
            roc_auc
        )
        
        self.results.append(result)
        
        return result
    
    def test_hybrid_detector(self, hybrid_detector):
        # Test Hybrid Detector 
        print("\nTesting Hybrid Detector...")
        
        # Hybrid is already trained just predict
        predict_start = time.perf_counter()
        predictions, classifications = hybrid_detector.predict(self.X_test)
        predict_time = (time.perf_counter() - predict_start) * 1000
        
        # Hybrid doesn't have true probabilities, use predictions as proxy
        try:
            # Try to get RF probabilities for ROC AUC
            ml_idx = [i for i, c in enumerate(classifications) if c != 'BLOCK']
            if ml_idx:
                X_ml = self.X_test.iloc[ml_idx]
                ml_probs = hybrid_detector.rf_model.predict_proba(X_ml)[:, 1]
                
                # Create full probability array
                probs = np.zeros(len(self.X_test))
                probs[[i for i, c in enumerate(classifications) if c == 'BLOCK']] = 1.0
                for idx, prob in zip(ml_idx, ml_probs):
                    probs[idx] = prob
                
                roc_auc = roc_auc_score(self.y_test, probs)
            else:
                roc_auc = 0.0
        except:
            roc_auc = 0.0
        
        result = self.calculate_metrics(
            "Hybrid Approach",
            predictions,
            0,
            predict_time,
            roc_auc
        )
        
        self.results.append(result)
        
        return result
    
    # Calculates the metrics of all algorithms
    def calculate_metrics(self, name, predictions, train_time, predict_time, roc_auc):

        accuracy = accuracy_score(self.y_test, predictions)
        recall = recall_score(self.y_test, predictions)
        precision = precision_score(self.y_test, predictions, zero_division=0)
        f1 = f1_score(self.y_test, predictions)
        
        cm = confusion_matrix(self.y_test, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        return AlgorithmResult(
            name=name,
            accuracy=accuracy,
            recall=recall,
            precision=precision,
            f1_score=f1,
            roc_auc=roc_auc,
            train_time_ms=train_time,
            predict_time_ms=predict_time,
            avg_time_per_tx_ms=predict_time / len(self.X_test),
            true_positives=int(tp),
            false_positives=int(fp),
            true_negatives=int(tn),
            false_negatives=int(fn),
            frauds_caught=int(tp),
            frauds_missed=int(fn),
            total_frauds=int(tp + fn)
        )
    
    # Prints and compares the results of different algorithms
    def compare_all(self, hybrid_detector=None):
        # Test all algorithms
        self.test_logistic_regression()
        self.test_decision_tree()
        self.test_naive_bayes()
        self.test_random_forest()
        self.test_gradient_boosting()
        
        # Test hybrid if provided
        if hybrid_detector is not None:
            self.test_hybrid_detector(hybrid_detector)
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        # Prints a summary table of the results
        print("\n" + "="*100)
        print("COMPARISON TABLE (sorted by recall)")
        print("="*100)
        print(f"{'Algorithm':<25} {'Accuracy':>10} {'Recall':>10} {'Precision':>10} {'F1':>10} {'ROC AUC':>10} {'Speed(ms)':>12}")
        print("-"*100)
        
        # Sort by recall
        sorted_results = sorted(self.results, key=lambda x: x.recall, reverse=True)
        
        for result in sorted_results:
            roc_str = f"{result.roc_auc:.4f}" if result.roc_auc > 0 else "N/A"
            print(f"{result.name:<25} "
                  f"{result.accuracy*100:>9.2f}% "
                  f"{result.recall*100:>9.2f}% "
                  f"{result.precision*100:>9.2f}% "
                  f"{result.f1_score*100:>9.2f}% "
                  f"{roc_str:>10} "
                  f"{result.avg_time_per_tx_ms:>11.4f}")
        
        print("="*100)
    
    def print_detailed_results(self):
            # Prints detailed results of algorithm
    
            sorted_results = sorted(self.results, key=lambda x: x.recall, reverse=True)
    
            print("\n" + "="*70)
            print("DETAILED RESULTS")
            print("="*70)
    
            for result in sorted_results:
                print(f"\n{result.name}:")
    
                print(f"  Performance:")
                print(f"    Accuracy:  {result.accuracy*100:.2f}%")
                print(f"    Recall:    {result.recall*100:.2f}%")
                print(f"    Precision: {result.precision*100:.2f}%")
                print(f"    F1 Score:  {result.f1_score*100:.2f}%")
                if result.roc_auc > 0:
                    print(f"    ROC AUC:   {result.roc_auc:.4f}")
    
                print(f"  Fraud Detection:")
                print(f"    Caught: {result.frauds_caught}/{result.total_frauds} ({result.recall*100:.2f}%)")
                print(f"    Missed: {result.frauds_missed}/{result.total_frauds}")
    
                print(f"  Confusion Matrix:")
                print(f"    True Positives:  {result.true_positives:,}")
                print(f"    False Positives: {result.false_positives:,}")
                print(f"    True Negatives:  {result.true_negatives:,}")
                print(f"    False Negatives: {result.false_negatives:,}")
    
                print(f"  Timing:")
                if result.train_time_ms > 0:
                    print(f"    Training:        {result.train_time_ms/1000:.2f} seconds")
                print(f"    Prediction:      {result.predict_time_ms:.2f} ms total")
                print(f"    Per transaction: {result.avg_time_per_tx_ms:.4f} ms")

if __name__ == "__main__":
    # Load and split the dataset (comparison needs train/test split for fair eval)
    loader = DataLoader('data/creditcardfraud/creditcard.csv')
    X_train, X_test, y_train, y_test = loader.load_and_split()

    # Load the trained hybrid detector
    detector = HybridFraudDetector(
        layer1_model_path='models/statistical_thresholds.pkl',
        layer2_model_path='models/random_forest_model.pkl'
    )

    # Run comparison
    comparison = FraudDetectionComparison(X_train, X_test, y_train, y_test)
    comparison.compare_all(hybrid_detector=detector)
    comparison.print_detailed_results()
