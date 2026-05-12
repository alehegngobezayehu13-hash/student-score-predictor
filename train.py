import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

data = pd.read_csv("data/student_scores.csv")

X = data[["hours", "sleep", "attendance"]]
y = data["score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)


predictions = model.predict(X_test)

results = pd.DataFrame({
    "Actual Score": y_test,
    "Predicted Score": predictions
})

print(results)

mae = mean_absolute_error(y_test, predictions)

print("\nMean Absolute Error:", mae)

plt.scatter(y_test, predictions)

plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")
plt.title("Actual vs Predicted Scores")

plt.show()