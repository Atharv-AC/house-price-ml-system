import numpy as np
from house_price_prediction.predict import HousePriceModel
import pytest


# this is a fake model for testing prediction 
class FakeModel:
    def predict(self, df):
        return np.array([np.log(100000)])

# this is a fake model for testing prediction failure    
class BrokenModel:
    def predict(self, df):
        raise RuntimeError("Model failed")


valid_input_dict =  {
  "bedrooms": 4,
  "bathrooms": 2,
  "stories": 2,
  "parking": 3,
  "area": 4235,
  "mainroad": "yes",
  "guestroom": "no",
  "basement": "yes",
  "hotwaterheating": "no",
  "airconditioning": "yes",
  "prefarea": "yes",
  "furnishingstatus": "no"
}


def test_predict_success():
    model = HousePriceModel(FakeModel())
    result = model.predict(valid_input_dict)
    assert result == pytest.approx(100000)





# Using invalid input for all process is not worth
invalid_input_dict =  {
  "bedrooms": -4,
  "bathrooms": "abc",
  "stories": 2,
  "parking": 0,  #? Since "area" is missing → it raises ValueError immediately.So execution stops here.
  "mainroad": "no",
  "something": "no",
  "basement": "yes",
  "hotwaterheating": "hmm", #? Since "area" is missing → it was never executed
  "airconditioning": "yes",
  "prefarea": "yes",
  "furnishingstatus": "no"
}

def test_validation_failure():
    model = HousePriceModel(BrokenModel())
    with pytest.raises(RuntimeError):
        model.predict(valid_input_dict)
    

def test_missing_feature():
    model = HousePriceModel(FakeModel())
    bad_input = valid_input_dict.copy()
    bad_input.pop("area")
    with pytest.raises(ValueError):
        model.predict(bad_input)

def test_unexpected_feature():
    model = HousePriceModel(FakeModel())
    bad_input = valid_input_dict.copy()
    bad_input["extra"] = 123
    with pytest.raises(ValueError):
        model.predict(bad_input)

def test_numeric_type_failure():
    model = HousePriceModel(FakeModel())
    bad_input = valid_input_dict.copy()
    bad_input["bedrooms"] = "abc"
    with pytest.raises(ValueError):
        model.predict(bad_input)


def test_negative_value():
    model = HousePriceModel(FakeModel())
    bad_input = valid_input_dict.copy()
    bad_input["bedrooms"] = -1

    with pytest.raises(ValueError):
        model.predict(bad_input)


def test_invalid_category():
    model = HousePriceModel(FakeModel())
    bad_input = valid_input_dict.copy()
    bad_input["mainroad"] = "maybe"

    with pytest.raises(ValueError):
        model.predict(bad_input)


