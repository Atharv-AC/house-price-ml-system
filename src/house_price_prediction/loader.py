import pandas as pd
import joblib
from house_price_prediction.config import settings
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent  # goes up 1 level `house_price_prediction` folder
PROJECT_DIR = BASE_DIR.parent  # goes up 2 levels `src` folder

DATA_DIR = PROJECT_DIR.parent / "data"  # goes up 3 levels `House-Price-Prediction` folder
MODELS_DIR = PROJECT_DIR.parent / "models" # goes up 3 levels `House-Price-Prediction` folder

def load_csv():
    csv_path = DATA_DIR / "Housing.csv"
    return pd.read_csv(csv_path)


def save_model(model, model_path: Path):
    # model_path = MODELS_DIR / filename

    joblib.dump(model, model_path)



def load_model():
    return joblib.load(settings.get_model_path())
    # return joblib.load(os.path.join(base_path, "..", "models", "latest.joblib"))

# df = load_csv()
# print(df.info())
# print(df.describe())
# print(df.shape)
# print(df.head(65))
# print(df.isnull().sum())
# print(df["sex"].head())
# pd.get_dummies(csv_path)
# print(df.columns)

