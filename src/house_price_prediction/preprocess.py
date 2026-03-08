from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np


def build_preprocessor(df):

    # Separate features
    # features are the columns which not contain only price column but all other columns
    # axis=1 means columns axis is 1 i.e. columns are the features and price is the target
    features = df.drop("price", axis=1)

    # Separate target
    # log transform stabilized variance
    target = np.log(df["price"])


    # Identify column types
    numerical_cols = [
        "area",
        "bedrooms",
        "bathrooms",
        "stories",
        "parking"
    ]

    categorical_cols = [
        "mainroad",
        "guestroom",
        "basement",
        "hotwaterheating",
        "airconditioning",
        "prefarea",
        "furnishingstatus"
    ]

    # Build ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            # handle_unknown means ignore the unknown values
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    return features, target, preprocessor


# this is the new concept it includes column transformer, onehotencoder and standard scaler
#-------------------------------------------------------------------------------------
# column transformer is used to combine multiple transformers,
#  transformer means to transform the data in simple words it
#  is a function that transforms the data  into a new format 
# that can be used by a machine learning model
#-------------------------------------------------------------------------------------
# onehotencoder is used to convert categorical variables into binary variables 
#-------------------------------------------------------------------------------------
# standard scaler is used to normalize the data
#-------------------------------------------------------------------------------------
