import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_healthcare_data(path=None):
    if path is None:
        # path from project root
        path = os.path.join("data", "KaggleV2-May-2016.csv")

    df = pd.read_csv(path)

    df['No-show'] = df['No-show'].apply(lambda x: 1 if x == 'Yes' else 0)

    # Dates
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
    df['WaitingTime'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
    df['ApptDayOfWeek'] = df['AppointmentDay'].dt.dayofweek

    df = df[(df['WaitingTime'] >= 0) & (df['Age'] >= 0) & (df['Age'] <= 120)]

    features = ['Age', 'WaitingTime', 'Scholarship', 'Hipertension', 'Diabetes', 'ApptDayOfWeek']
    X = df[features]
    y = df['No-show']

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)

    return X_scaled.to_numpy(), y.to_numpy()
