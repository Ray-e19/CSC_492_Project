import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
    
    # Load and split data into 70% training and 30% testing
    def load_and_split(self, test_size=0.3, random_state=42):
        
        print(f"Loading File: {self.filepath}")

        df = pd.read_csv(self.filepath)
        
        # Add Hour feature
        df['Hour'] = (df['Time'] / 3600) % 24
        
        # Separate features and labels
        # Drops Class and Time after deriving Hour
        X = df.drop(['Class', 'Time'], axis=1)  
        
        # Contains DataFrame of Class
        y = df['Class']
        
        # Split 30% data for testing 
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            stratify=y, 
            random_state=random_state
        )
        
        print(f"Training set: {X_train.shape[0]} transactions ({y_train.sum()} frauds)")
        print(f"Test set: {X_test.shape[0]} transactions ({y_test.sum()} frauds)")

        return X_train, X_test, y_train, y_test

    def load_full(self):
        # Loads entire CSV without splitting or training
        print(f"Loading File: {self.filepath}")

        df = pd.read_csv(self.filepath)

        # Add Hour feature derived from Time
        df['Hour'] = (df['Time'] / 3600) % 24

        # Separate features and labels
        X = df.drop(['Class', 'Time'], axis=1)
        y = df['Class']

        print(f"Total: {X.shape[0]} transactions ({y.sum()} frauds)")

        return X, y