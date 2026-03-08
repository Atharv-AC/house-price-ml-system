from pydantic_settings import BaseSettings
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]    # goes up 2 levels `House-Price-Prediction`
MODEL_PATH = ROOT_DIR / "models" / "latest.joblib"
REPORT_PATH = ROOT_DIR / "reports" 


# Defined a Pydantic model to represent the settings 
class Settings(BaseSettings):
    app_env: str = "dev"                 #* dev, test, docker, prod this can be defined in .env and passed to the container
    log_level: str = "INFO"              
    model_path: str | None = None       

    def get_model_path(self):
        if self.model_path:
            return self.model_path

        if self.app_env == "docker":
            return "/app/models/latest.joblib"

        return "models/latest.joblib"
    # model_path: str = "/app/models/latest.joblib"
    
    # model_info: Path = REPORT_PATH / "train_summary.json"

settings = Settings()