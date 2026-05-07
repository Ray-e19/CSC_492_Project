from data_loader import DataLoader
from sklearn.ensemble import RandomForestClassifier
import pickle

class RandomForestTrainer:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
    # Build a forest of trees from the training set and returns a model
    def fit(self, X_train, y_train):
        print(f"Training Random Forest with {X_train.shape[0]} transactions...")
        
        # Trains Random Forest
        self.model.fit(X_train, y_train)

        return self
    
    # Saves Random Forest Model
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
            
        print(f"Saved Random Forest model to {filepath}")

if __name__ == "__main__":
    # Load and split data
    loader = DataLoader('data/creditcardfraud/creditcard.csv')
    X_train, X_test, y_train, y_test = loader.load_and_split()
    
    # Train Layer 2 and produce model
    layer2 = RandomForestTrainer()
    layer2.fit(X_train, y_train)
    
    # Save model
    layer2.save('models/random_forest_model.pkl')