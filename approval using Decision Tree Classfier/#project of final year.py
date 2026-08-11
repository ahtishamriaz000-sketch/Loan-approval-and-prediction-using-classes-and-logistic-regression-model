import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class DatasetLoader:
    def load(self, file):
        return pd.read_csv("C:/Users/Ahtsham-Riaz/Downloads/train_u6lujuX_CVtuZ9i.csv")

class DataInspector:
    def show(self, data):
        print("\n===== DATA PREVIEW =====")
        print(data.head())
        print("\nShape:", data.shape)
        print("\nMissing Values:\n", data.isnull().sum())

class DataCleaner:
    def clean(self, data):
        data = data.drop("Loan_ID", axis=1)
        for col in data.columns:
            if data[col].dtype in ['int64', 'float64']:
                data[col] = data[col].fillna(data[col].median())
            else:
                data[col] = data[col].fillna(data[col].mode()[0])
        return data

class DataVisualizer:
    def plot(self, data):
        sns.countplot(x="Loan_Status", data=data)
        plt.title("Loan Status Distribution")
        plt.show(block=False)
        plt.pause(34)
        plt.close()

class Encoder:
    def encode(self, data):
        # Get dummies - yeh sabse reliable tarika hai
        data = pd.get_dummies(data, drop_first=True)
        # Sab columns ko int mein convert karo
        for col in data.columns:
            data[col] = data[col].astype(int)
        return data

class DataSplitter:
    def split(self, data):
        # Loan_Status_Y column dhundo
        target_col = [c for c in data.columns if 'Loan_Status' in c][0]
        X = data.drop(target_col, axis=1)
        y = data[target_col]
        print("Target column:", target_col)
        print("Unique y values:", y.unique())
        return train_test_split(X, y, test_size=0.2, random_state=42)

class ModelTrainer:
    def train(self, X_train, y_train):
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, solver='lbfgs'),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier()
        }
        trained_models = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            trained_models[name] = model
        return trained_models

class ModelEvaluator:
    def evaluate(self, models, X_test, y_test):
        best_model = None
        best_score = 0
        for name, model in models.items():
            pred = model.predict(X_test)
            acc = accuracy_score(y_test, pred)
            print(f"\n===== {name} =====")
            print("Accuracy:", acc)
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, pred))
            print("\nClassification Report:")
            print(classification_report(y_test, pred))
            if acc > best_score:
                best_score = acc
                best_model = model
        print("\nBest Model Selected")
        return best_model

class Predictor:
    def predict(self, model, sample):
        return model.predict([sample])

class LoanSystem:
    def run(self):
        loader = DatasetLoader()
        data = loader.load("train_u6lujuX_CVtuZ9i.csv")

        inspector = DataInspector()
        inspector.show(data)

        cleaner = DataCleaner()
        data = cleaner.clean(data)

        viz = DataVisualizer()
        viz.plot(data)

        encoder = Encoder()
        data = encoder.encode(data)

        splitter = DataSplitter()
        X_train, X_test, y_train, y_test = splitter.split(data)

        trainer = ModelTrainer()
        models = trainer.train(X_train, y_train)

        evaluator = ModelEvaluator()
        best_model = evaluator.evaluate(models, X_test, y_test)

        predictor = Predictor()
        sample = X_test.iloc[0].values
        result = predictor.predict(best_model, sample)

        print("\nFinal Prediction:", "Approved" if result[0] == 1 else "Rejected")

app = LoanSystem()
app.run()