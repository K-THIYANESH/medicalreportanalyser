import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from app.config import DATASET_PATH


class Prediction:
    def __init__(self):
        self.df = pd.read_csv(DATASET_PATH)
        self.feature_columns = ["Hemoglobin", "WBC", "RBC", "Platelets"]
        self.X = self.df[self.feature_columns]
        self.targets = [
            "Anemia",
            "Infection",
            "Thrombocytopenia",
            "Leukocytosis",
        ]
        self.models = {}
        self.train()

    def train(self):
        for disease in self.targets:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(self.X, self.df[disease])
            self.models[disease] = model

    def predict(self, sample):
        frame = pd.DataFrame(sample, columns=self.feature_columns)
        result = {}
        for disease, model in self.models.items():
            prob = model.predict_proba(frame)[0][1]
            result[disease] = round(prob * 100, 2)
        return result
