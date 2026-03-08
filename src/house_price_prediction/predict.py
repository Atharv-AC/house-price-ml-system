#----------------------------
# For API call
#----------------------------


import numpy as np
import pandas as pd


# def predict_price_api(input_dict):
#     model = load_model()
#     df = pd.DataFrame([input_dict])
#     log_pred = model.predict(df)
#     price = np.exp(log_pred)
#     return float(price[0])

required_features = {
    "bedrooms": "int",
    "bathrooms": "int",
    "stories": "int",
    "parking": "int",
    "area": "int",
    "mainroad": "str",
    "guestroom": "str" ,
    "basement": "str",
    "hotwaterheating": "str",
    "airconditioning": "str",
    "prefarea": "str",
    "furnishingstatus": "str"}

class HousePriceModel:

    def __init__(self, model):
        self.model = model

    def validate(self, input_dict: dict):
        """
        handles missing or unexpected features

        Checks:
            data type
            data range
        """
        missing = [f for f in required_features if f not in input_dict]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
       
        
        for f in input_dict:
            if f not in required_features:
                raise ValueError(f"Unexpected feature: {f}")

        num_feature = ["bedrooms","bathrooms","stories","parking","area"]

        for f in num_feature:
            try:
                input_dict[f] = float(input_dict[f])
            except:
                raise ValueError(f"{f} must be a number")
            
            if input_dict[f] < 0:
                raise ValueError(f"{f} Cannot be negative")

        cat_feature = ["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea","furnishingstatus"]

        for f in cat_feature:
            if input_dict[f].lower() not in ["yes", "no"]:
                raise ValueError(f"{f} must be a yes or no")

            input_dict[f] = input_dict[f].lower()  

        return input_dict
    
    # This function converts the input dictionary to a dataframe to be used by the model
    def to_dataframe(self, input_dict):
        df = pd.DataFrame([input_dict])
        return df
           

    # The dict means just a type hint:
            # input_dict is expected to be a dictionary.
            # -> float means returns a float
    def predict(self, input_dict: dict) -> float:
        """
        Predict house price from validated input features.

        Parameters:
            input_dict (dict): Feature dictionary.

        Returns:
            float: Predicted house price in original scale.
        """
        validate_input = self.validate(input_dict)
        df = self.to_dataframe(validate_input)
        log_pred = self.model.predict(df)
        price = np.exp(log_pred)
      
        return float(price[0])
