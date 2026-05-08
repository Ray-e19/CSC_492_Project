import pickle
import time
import pandas as pd
import numpy as np
from data_loader import DataLoader
from typing import Tuple, List
from pathlib import Path
from sklearn.metrics import (accuracy_score, recall_score,
                             precision_score, f1_score,
                             confusion_matrix, roc_auc_score)

class HybridFraudDetector:

    # ML probability thresholds for layer 2
    THRESHOLD_FLAG = 0.20
    THRESHOLD_PASS = 0.40

    # Layer 1 thresholds 
    Z_SCORE_BLOCK = 6
    EXTREME_COUNT_BLOCK = 5
    EXTREME_COUNT_FLAG = 3
    AMOUNT_MULTIPLIER = 2.5

    def __init__(self, layer1_model_path: str, layer2_model_path: str):
        # Loads the statistical thresholds calculated from training data
        with open(layer1_model_path, 'rb') as f:
            self.stat_thresholds = pickle.load(f)

        # Loads the random forest model
        with open(layer2_model_path, 'rb') as f:
            self.rf_model = pickle.load(f)

        # Pre-convert the threshold dictionary into numpy arrays for vectorization
        self.setup_threshold_arrays()

    def setup_threshold_arrays(self):
        # Retrieves the stats for V1-V28 and stores them as numpy arrays
        v_features = []
        for i in range(1, 29):
            v_features.append(f'V{i}')       
        
        means = []
        stds  = []
        p01s  = []
        p99s  = []

        for f in v_features:
            means.append(self.stat_thresholds[f'{f}_mean'])
            stds.append(self.stat_thresholds[f'{f}_std'])
            p01s.append(self.stat_thresholds[f'{f}_p01'])
            p99s.append(self.stat_thresholds[f'{f}_p99'])

        self.means = np.array(means)
        self.stds  = np.array(stds)
        self.p01s  = np.array(p01s)
        self.p99s  = np.array(p99s)

        self.amount_max  = self.stat_thresholds['amount_max']
        self.amount_99th = self.stat_thresholds['amount_99th']

    # Predicts each transaction and conducts classification
    def predict(self, X: pd.DataFrame) -> Tuple[pd.Series, List[str]]:
        num_transactions = len(X)

        # Grabs the V features and amounts as numpy arrays
        # .values converts from pandas to numpy so we can do fast math on it
        v_features = [f'V{i}' for i in range(1, 29)]
        X_v = X[v_features].values
        amounts = X['Amount'].values

        # Layer 1 statistical screening
        # Checks if the amount is way too high
        is_extreme_amount = amounts > (self.AMOUNT_MULTIPLIER * self.amount_max)

        # Checks if any features have a high z-score
        z_scores = np.abs((X_v - self.means) / self.stds)
        is_extreme_z = np.any(z_scores > self.Z_SCORE_BLOCK, axis=1)

        # Checks how many features are outside either 1st or 99th percentile
        below_p01 = X_v < self.p01s
        above_p99 = X_v > self.p99s
        extreme_counts = np.sum(below_p01 | above_p99, axis=1)

        # Combine the checks into BLOCK, FLAG, and PASS 
        is_block = is_extreme_amount | is_extreme_z | (extreme_counts >= self.EXTREME_COUNT_BLOCK)
        is_flag = (~is_block) & ((extreme_counts >= self.EXTREME_COUNT_FLAG) | (amounts > self.amount_99th))
        is_pass = ~(is_block | is_flag)

        # Assigns and stores the layer 1 labels into an array
        classifications = np.empty(num_transactions, dtype=object)
        classifications[is_block] = 'BLOCK'
        classifications[is_flag] = 'FLAG'
        classifications[is_pass] = 'PASS'

        # Blocked transactions are automatically fraud 
        # Does not need to be processed in second layer 
        predictions = np.zeros(num_transactions, dtype=int)
        predictions[is_block] = 1

        # Layer 2 Random forest 
        # Processes FLAG or PASS transactions
        ml_idx = np.where(is_flag | is_pass)[0]

        if len(ml_idx) > 0:
            X_ml = X.iloc[ml_idx]
            fraud_probs = self.rf_model.predict_proba(X_ml)[:, 1]

            # Flagged transactions get a lower threshold (0.20) because they are already suspicious
            # Passed transactions get the normal threshold (0.40)S
            is_flag = is_flag[ml_idx]
            thresholds = np.where(is_flag, self.THRESHOLD_FLAG, self.THRESHOLD_PASS)

            # Compare probabilities against contextual thresholds
            ml_predictions = (fraud_probs >= thresholds).astype(int)
            predictions[ml_idx] = ml_predictions

        # Returns the prediction and classification of each transaction
        return pd.Series(predictions, index=X.index), classifications.tolist()

    # Returns a dataframe of transactions that were classified fraud but were actually legit
    def get_false_positives(self, X: pd.DataFrame, y_true: pd.Series) -> pd.DataFrame:
        predictions, classifications = self.predict(X)

        # False positive where the transaction was predicted as fraud (1) but was actually legit (0)
        is_false_postive = (predictions == 1) & (y_true == 0)
        false_postive_idx = np.where(is_false_postive)[0]

        if len(false_postive_idx) == 0:
            return pd.DataFrame()

        fp_transactions = X.iloc[false_postive_idx].copy()
        fp_transactions['True_Label'] = 0
        fp_transactions['Predicted_Label'] = 1
        
        layer1_labels = []
        for i in false_postive_idx:
            layer1_labels.append(classifications[i])
        fp_transactions['Layer1_Classification'] = layer1_labels
        
        # Get the ML probability for each false positive
        # Blocked transactions dont have a real probability so we just set it to 1.0
        ml_probability = []
        for idx in false_postive_idx:
            if classifications[idx] == 'BLOCK':
                ml_probability.append(1.0)
            else:
                prob = self.rf_model.predict_proba(X.iloc[[idx]])[0, 1]
                ml_probability.append(prob)
        fp_transactions['ML_Probability'] = ml_probability

        # Record which threshold was used for each transaction
        thresholds = []
        for idx in false_postive_idx:
            if classifications[idx] == 'BLOCK':
                thresholds.append('N/A (BLOCK)')
            elif classifications[idx] == 'FLAG':
                thresholds.append(0.20)
            else:
                thresholds.append(0.40)
        fp_transactions['Threshold_Used'] = thresholds

        # Determine the reason for each false positive
        reasons = []
        for idx in false_postive_idx:
            if classifications[idx] == 'BLOCK':
                reasons.append(self.explain_block_reason(X.iloc[idx]))
            else:
                prob = ml_probability[false_postive_idx.tolist().index(idx)]
                thresh = thresholds[false_postive_idx.tolist().index(idx)]
                if isinstance(thresh, float):
                    reasons.append(f"ML probability {prob:.4f} >= threshold {thresh}")
                else:
                    reasons.append("statistical outlier")
        fp_transactions['Reason'] = reasons

        return fp_transactions

    # Returns a dataframe of transactions that were classified legit but were actually fraud
    def get_false_negatives(self, X: pd.DataFrame, y_true: pd.Series) -> pd.DataFrame:
        predictions, classifications = self.predict(X)

        # False negative where the transaction was predicted as legitimate (0) but was actually fraud (1)
        is_false_negative = (predictions == 0) & (y_true == 1)
        false_negative_idx = np.where(is_false_negative)[0]

        if len(false_negative_idx) == 0:
            return pd.DataFrame()

        fn_transactions = X.iloc[false_negative_idx].copy()
        fn_transactions['True_Label'] = 1
        fn_transactions['Predicted_Label'] = 0
        
        layer1_labels = []
        for i in false_negative_idx:
            layer1_labels.append(classifications[i])
        fn_transactions['Layer1_Classification'] = layer1_labels
        
        # Get the ML probability for each missed fraud
        ml_probability = []
        for idx in false_negative_idx:
            prob = self.rf_model.predict_proba(X.iloc[[idx]])[0, 1]
            ml_probability.append(prob)
        fn_transactions['ML_Probability'] = ml_probability

        # Figure out which threshold was applied
        thresholds = []
        for idx in false_negative_idx:
            if classifications[idx] == 'FLAG':
                thresholds.append(0.20)
            else:
                thresholds.append(0.40)
        fn_transactions['Threshold_Used'] = thresholds

        # Determine the reason for each false negative
        reasons = []
        for idx in false_negative_idx:
            prob = ml_probability[false_negative_idx.tolist().index(idx)]
            thresh = thresholds[false_negative_idx.tolist().index(idx)]
            diff = thresh - prob
            reasons.append(f"ML probability {prob:.4f} < threshold {thresh} (missed by {diff:.4f})")
        fn_transactions['Reason'] = reasons

        # Distance from threshold to show how close to catching it
        distance_from_threshold = []
        for probability, threshold in zip(ml_probability, thresholds):
            distance_from_threshold.append(threshold - probability)

        fn_transactions['Distance_From_Threshold'] = distance_from_threshold

        return fn_transactions

    # Explains why a transaction was blocked by layer 1
    def explain_block_reason(self, transaction: pd.Series) -> str:
        # Checks if amount was the problem
        if transaction['Amount'] > self.AMOUNT_MULTIPLIER * self.amount_max:
            return (f"extreme amount: ${transaction['Amount']:.2f} > "
                    f"${self.AMOUNT_MULTIPLIER * self.amount_max:.2f}")

        # Checks if a z-score was the problem
        v_features = [f'V{i}' for i in range(1, 29)]
        values = transaction[v_features].values
        z_scores = np.abs((values - self.means) / self.stds)

        extreme_z_idx = np.where(z_scores > self.Z_SCORE_BLOCK)[0]
        if len(extreme_z_idx) > 0:
            feature = v_features[extreme_z_idx[0]]
            z_val = z_scores[extreme_z_idx[0]]
            return f"extreme z-score in {feature}: {z_val:.2f} > {self.Z_SCORE_BLOCK}"

        # Checks if too many extreme features was the problem
        extreme_count = np.sum((values < self.p01s) | (values > self.p99s))
        if extreme_count >= self.EXTREME_COUNT_BLOCK:
            return f"too many extreme features: {extreme_count} >= {self.EXTREME_COUNT_BLOCK}"

    # Prints basic summary to terminal and saves detailed results to csv files
    def print_error_report(self, X: pd.DataFrame, y_true: pd.Series,
                           predictions: pd.Series, classifications: list,
                           processing_time_ms: float,
                           output_dir: str = 'results'):
 
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
 
        fp = self.get_false_positives(X, y_true)
        fn = self.get_false_negatives(X, y_true)
        total_frauds = int(y_true.sum())
 
        # Unpacks the confusion matrix into individual variables
        tn, fp_count, fn_count, tp = confusion_matrix(y_true, predictions).ravel()
 
        # Calculates all the metrics
        accuracy  = accuracy_score(y_true, predictions) * 100
        recall    = recall_score(y_true, predictions) * 100
        precision = precision_score(y_true, predictions, zero_division=0) * 100
        f1 = f1_score(y_true, predictions, zero_division=0) * 100
        avg_ms = processing_time_ms / len(X)
 
        # Calculates ROC AUC
        try:
            cls_arr = np.array(classifications)
            fraud_probs = np.zeros(len(X))
 
            block_idx = np.where(cls_arr == 'BLOCK')[0]
            non_block_idx = np.where(cls_arr != 'BLOCK')[0]
 
            fraud_probs[block_idx] = 1.0
            if len(non_block_idx) > 0:
                fraud_probs[non_block_idx] = self.rf_model.predict_proba(
                    X.iloc[non_block_idx])[:, 1]
 
            roc_auc = roc_auc_score(y_true, fraud_probs)
        except Exception:
            roc_auc = 0.0
 
        cls_counts = pd.Series(classifications).value_counts()
 
        # Create the metrics summary csv
        metrics_rows = [
            ['-----PERFORMANCE METRICS-----', ''],
            ['Accuracy',     f'{accuracy:.2f}%'],
            ['Recall',       f'{recall:.2f}%'],
            ['Precision',    f'{precision:.2f}%'],
            ['F1 Score',     f'{f1:.2f}%'],
            ['ROC AUC',      f'{roc_auc:.4f}'],
 
            ['', ''],
            ['-----PROCESSING TIME-----', ''],
            ['Total Time',              f'{processing_time_ms:.2f} ms'],
            ['Avg Per Transaction',     f'{avg_ms:.4f} ms'],
            ['Total Transactions',      f'{len(X)}'],
 
            ['', ''],
            ['-----CONFUSION MATRIX-----', ''],
            ['True Positives', f'{tp}'],
            ['False Positives', f'{fp_count}'],
            ['True Negatives', f'{tn}'],
            ['False Negatives', f'{fn_count}'],
 
            ['', ''],
            ['-----FRAUD DETECTION-----', ''],
            ['Total Frauds',     f'{total_frauds}'],
            ['Frauds Caught',    f'{tp}'],
            ['Frauds Missed',    f'{fn_count}'],
            ['Catch Rate',       f'{tp / total_frauds * 100:.2f}%'],
 
            ['', ''],
            ['-----LAYER 1 DISTRIBUTION-----', ''],
        ]
 
        for label, count in cls_counts.items():
            pct = count / len(X) * 100
            metrics_rows.append([f'{label}', f'{count} ({pct:.2f}%)'])
 
        with open(output_path / 'metrics_summary.csv', 'w') as f:
            f.write('Metric,Value\n')
            for row in metrics_rows:
                if row == ['', '']:
                    f.write('\n')
                elif row[1] == '':
                    f.write(f'{row[0]}\n')
                else:
                    f.write(f'{row[0]},{row[1]}\n')  
 
        # Build the error report csv
        # Combines false positives and false negatives
        if len(fp) > 0:
            fp_rows = fp[['Amount', 'Layer1_Classification', 'ML_Probability',
                          'Threshold_Used', 'Reason']].copy()
            fp_rows['Distance_From_Threshold'] = None
            fp_rows['Error_Type'] = 'False Positive'
            fp_rows['Transaction_Number'] = fp.index
        else:
            fp_rows = pd.DataFrame(columns=['Amount', 'Layer1_Classification',
                                            'ML_Probability', 'Threshold_Used', 'Reason',
                                            'Distance_From_Threshold', 'Error_Type',
                                            'Transaction_Number'])
 
        if len(fn) > 0:
            fn_rows = fn[['Amount', 'Layer1_Classification', 'ML_Probability',
                          'Threshold_Used', 'Distance_From_Threshold', 'Reason']].copy()
            fn_rows['Error_Type'] = 'False Negative / Missed Fraud'
            fn_rows['Transaction_Number'] = fn.index
        else:
            fn_rows = pd.DataFrame(columns=['Amount', 'Layer1_Classification',
                                            'ML_Probability', 'Threshold_Used',
                                            'Distance_From_Threshold', 'Reason',
                                            'Error_Type', 'Transaction_Number'])
 
        report = pd.concat([fp_rows, fn_rows], ignore_index=True)
        report = report[['Transaction_Number', 'Error_Type', 'Amount',
                         'Layer1_Classification', 'ML_Probability',
                         'Threshold_Used', 'Distance_From_Threshold', 'Reason']]
        report.to_csv(output_path / 'error_report.csv', index=False)

        print(f"\nSaved to {output_path}/metrics_summary.csv")
        print(f"Saved to {output_path}/error_report.csv")

if __name__ == "__main__":
    # Load and split data into 70% training and 30% testing
    loader = DataLoader('data/creditcardfraud/creditcard.csv')
    X_train, X_test, y_train, y_test = loader.load_and_split()

    # Load the trained models
    detector = HybridFraudDetector(
        layer1_model_path='models/statistical_thresholds.pkl',
        layer2_model_path='models/random_forest_model.pkl'
    )

    # Run predictions on the test set and time it
    print("\nRunning predictions on dataset...")
    start = time.perf_counter()
    predictions, classifications = detector.predict(X_test)
    elapsed_ms = (time.perf_counter() - start) * 1000

    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    total_frauds = int(y_test.sum())

    # Print all the results to the terminal
    print("\n-----Results-----")
    print(f"Accuracy: {accuracy_score(y_test, predictions)*100:.2f}%")
    print(f"Recall: {recall_score(y_test, predictions)*100:.2f}%")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0)*100:.2f}%")
    print(f"F1 Score: {f1_score(y_test, predictions, zero_division=0)*100:.2f}%")

    print("\n-----Confusion Matrix-----")
    print(f"True Positives: {tp:,}")
    print(f"False Positives: {fp:,}")
    print(f"True Negatives: {tn:,}")
    print(f"False Negatives: {fn:,}")

    print("\n-----Fraud Detection-----")
    print(f"Frauds Caught: {tp}/{total_frauds} ({tp/total_frauds*100:.2f}%)")
    print(f"Frauds Missed: {fn}/{total_frauds} ({fn/total_frauds*100:.2f}%)")

    print("\n-----Processing Time-----")
    print(f"Total: {elapsed_ms:.2f} ms")
    print(f"Per Transaction: {elapsed_ms/len(X_test):.4f} ms")

    # Save metrics and error report to the results folder
    detector.print_error_report(
        X_test, y_test,
        predictions=predictions,
        classifications=classifications,
        processing_time_ms=elapsed_ms,
        output_dir='results'
    )