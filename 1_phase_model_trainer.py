import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from micromlgen import port
import os

def main():
    print("Loading 1-Phase Features...")
    df = pd.read_csv("data/1_phase_features.csv")
    
    X = df[["RMS", "Peak", "Mean", "Variance"]].values
    y = df["Label"].values
    
    print("Training Decision Tree Model...")
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X, y)
    
    print(f"Accuracy on training set: {clf.score(X, y)}")
    
    print("Porting model to C++ header (micromlgen)...")
    c_code = port(clf)
    
    os.makedirs("firmware", exist_ok=True)
    with open("firmware/model.h", "w") as f:
        f.write(c_code)
        
    print("Model successfully exported to firmware/model.h")

if __name__ == "__main__":
    main()
