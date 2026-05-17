import pandas as pd
import pickle

with open("models/score_model.pkl", "rb") as f:
    model = pickle.load(f)

print("\nEnter student details:\n")

hours = float(input("Hours Studied: "))
attendance = float(input("Attendance: "))
sleep = float(input("Sleep Hours: "))
previous = float(input("Previous Score: "))
tutoring = float(input("Tutoring Sessions: "))
activity = float(input("Physical Activity: "))

data = pd.DataFrame([[
    hours,
    attendance,
    sleep,
    previous,
    tutoring,
    activity
]], columns=[
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity"
])

prediction = model.predict(data)

print("\nPredicted Exam Score:", prediction[0])