import pickle
import pandas as pd

model_dict = pickle.load(open("Models/model_dict.pkl", "rb"))
model = model_dict["gs_lr"]


def input_values():

    stats = {
        "Pregnancies": 0,
        "Glucose:": 0,
        "BloodPressure": 0,
        "SkinThickness": 0,
        "Insulin": 0,
        "BMI": 0,
        "DiabetesPedigreeFunction": 0,
        "Age": 0
    }

    for name, value in stats.items():
        stats[name] = input("Enter the {}".format(name))

    dataframe = pd.DataFrame.from_dict(stats, orient='columns')
    return dataframe


def output():
    data = input_values()
    pred = model.predict(data)

    if pred == 0:
        return "not Diabetic."
    elif pred == 1:
        return "Diabetic."

print(f"The patient is {output()}.")


