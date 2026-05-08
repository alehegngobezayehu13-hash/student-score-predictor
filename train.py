import pandas as pd
from sklearn.model_selection import train_test_split
data = pd.read_csv("data/student_scores.csv")
X = data[["hours", "sleep", "attendance"]]
y = data["score"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Training data:")
print(X_train)
print("\nTesting data:")
print(X_test)