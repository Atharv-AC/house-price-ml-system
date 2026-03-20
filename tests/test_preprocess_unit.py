import pandas as pd
import numpy as np
from house_price_prediction.preprocess import build_preprocessor
from sklearn.compose import ColumnTransformer

def test_build_preprocessor_returns_expected_objects():
    data = {
        "price": [100000, 200000],
        "area": [2000, 2500],
        "bedrooms": [3, 4],
        "bathrooms": [2, 3],
        "stories": [2, 2],
        "parking": [1, 2],
        "mainroad": ["yes", "no"],
        "guestroom": ["no", "yes"],
        "basement": ["no", "yes"],
        "hotwaterheating": ["no", "no"],
        "airconditioning": ["yes", "yes"],
        "prefarea": ["yes", "no"],
        "furnishingstatus": ["furnished", "semi-furnished"],
    }

    df = pd.DataFrame(data)

    features, target, preprocessor = build_preprocessor(df)

    assert "price" not in features.columns
    assert isinstance(preprocessor, ColumnTransformer)

    # check log transform
    assert np.allclose(target.values, np.log(df["price"]).values)