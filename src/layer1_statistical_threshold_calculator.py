import pickle
from data_loader import DataLoader  

class StatisticalThresholdCalculator:
    def __init__(self):
        self.thresholds = {}
        
    # Calculates statistical thresholds from legitimate transactions and returns a model
    def fit(self, X_train, y_train):
        # Filters only legitimate transactions 
        legit_transactions = X_train[y_train == 0]
        
        print(f"Analyzing {len(legit_transactions)} legitimate transactions...")
        
        # Amount thresholds
        self.thresholds['amount_max'] = legit_transactions['Amount'].max()
        self.thresholds['amount_99th'] = legit_transactions['Amount'].quantile(0.99)
        
        # Finds mean, standard deviation, first percentile, and 99th percentile
        # from features V1-V28
        v_features = [col for col in X_train.columns if col.startswith('V')]
        
        for feature in v_features:
            self.thresholds[f'{feature}_mean'] = legit_transactions[feature].mean()
            self.thresholds[f'{feature}_std'] = legit_transactions[feature].std()
            self.thresholds[f'{feature}_p01'] = legit_transactions[feature].quantile(0.01)
            self.thresholds[f'{feature}_p99'] = legit_transactions[feature].quantile(0.99)
        
        print(f"Calculated thresholds for {len(v_features)} features")
        print(f"Amount Max: ${self.thresholds['amount_max']:.2f}")
        print(f"Amount 99th: ${self.thresholds['amount_99th']:.2f}")
        
        return self
    
    # Saves thresholds to file
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.thresholds, f)
        print(f"Saved threshold file to {filepath}")

if __name__ == "__main__":
    # Load and split data
    loader = DataLoader('data/creditcardfraud/creditcard.csv')
    X_train, X_test, y_train, y_test = loader.load_and_split()
    
    # Train Layer 1 and produce model
    layer1 = StatisticalThresholdCalculator()
    layer1.fit(X_train, y_train)
    
    # Save model
    layer1.save('models/statistical_thresholds.pkl')