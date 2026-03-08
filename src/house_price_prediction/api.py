from fastapi import FastAPI, HTTPException, Depends
from house_price_prediction.predict import HousePriceModel
from pydantic import BaseModel
import json
import logging
from house_price_prediction.loader import load_model
from contextlib import asynccontextmanager
from house_price_prediction.config import settings
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse



# Initialize logging for API 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)
logger.info("API startup")



# lifespan will run before the first request. It is used to load the model and save it in app state
@asynccontextmanager
async def lifespans(app: FastAPI):
    try:
        if settings.app_env != "test":
            model = load_model()
            app.state.model = HousePriceModel(model)
            app.state.model_loaded = True
            logger.info("Model loaded successfully")
        else:
            logger.info("Skip")
    except Exception:
        # exc_info=True will log the full traceback
        logger.error("Model failed to load: ", exc_info=True)

    yield

    logger.info("Shutting down")

app = FastAPI(lifespan=lifespans)



# This tells FastAPI: /static/* → serve files from static folder
# http://localhost:8000/static/style.css
app.mount("/static", StaticFiles(directory="src/house_price_prediction/static"), name="static")

# app.state.model = None
# app.state.model_loaded = False


# this is a pydantic model for prediction input data. This is used to validate the input data
class House(BaseModel):
    bedrooms: int
    bathrooms: int
    stories: int
    parking: int
    area: int
    mainroad: str
    guestroom: str
    basement: str
    hotwaterheating: str
    airconditioning: str
    prefarea: str
    furnishingstatus: str

# this lets you open ui at http://localhost:8000
@app.get("/")
def home():
    return FileResponse("src/house_price_prediction/static/index.html")


# get request (root) this is the root endpoint which returns a message for testing purposes
# @app.get("/")
# def root():
#     return {"message": "House Price Prediction API! Running"}


# this function loads the model and returns it
def get_model():
    if not app.state.model_loaded:
        # 503 Service Unavailable
        raise HTTPException(status_code=503, detail="Model not loaded")
    return app.state.model


# post request for prediction. we have use here dependency injection 
# which is a feature of FastAPI that allows us to inject a dependency into a function 
# to be used as a parameter and also allows us to override the dependency
@app.post("/predict")
def predict_price(features: House, model = Depends(get_model)):
    price = model.predict(features.model_dump())
    return {'Prediction' : float(price)}

    
# print(predict_price)
# predict_price(House, app.state.model)


# get request for health check it ensures that the model is loaded and returns a true/false
@app.get("/health")
def health_info():
    if not app.state.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
            "status": "ok",
            "model_loaded": True
            }



# get request for model metadata it will not return anything if the model is not loaded
@app.get("/model-info")
def load_model_metadata():
    from pathlib import Path
    REPORT_PATH = Path("reports/model_summary.json")

    try:
        with open(REPORT_PATH) as fi:
            model_summary = json.load(fi)
            return model_summary
    except Exception:
        logger.error("Model Metadata failed to load: ", exc_info=True)
        # 500 Internal Server Error
        raise HTTPException(status_code=500, detail="Model Metadata not found")


# print("MODEL PATH:", settings.model_path)
# {
#   "bedrooms": 4,
#   "bathrooms": 2,
#   "stories": 2,
#   "parking": 3,
#   "area": 4235,
#   "mainroad": "yes",
#   "guestroom": "no",
#   "basement": "yes",
#   "hotwaterheating": "no",
#   "airconditioning": "yes",
#   "prefarea": "yes",
#   "furnishingstatus": "no"
# }
