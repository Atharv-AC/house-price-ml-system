from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import yaml

def build_pipeline(preprocessor, model_type="linear"):

    if model_type == "linear":
        model = LinearRegression()

    elif model_type == "ridge":
        # if alpha = 1000.0 means in simple words we are using l2 regularization
        model = Ridge(alpha= 1.0)

    elif model_type == "lasso":
        # if alpha = 1.0 means in simple words we are using l1 regularization
        model = Lasso(alpha=1.0)
    
    elif model_type == "rf":
        model = RandomForestRegressor(random_state=42)

    else:
        raise ValueError("Invalid model type")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )


    return pipeline

#=======================================================================
# A Pipeline is:
# A machine that performs multiple steps in order, automatically.
#=======================================================================

#----------------------------------------------------
#  **Linear Regression + Penalty.**
# Normal Linear Regression tries to: Minimize error.
#----------------------------------------------------

#-------------------------------------------------------
# Ridge tries to:
# Minimize error + penalty for large coefficients.

# Ridge:
# Shrinks coefficients to make them stable.

#-------------------------------------------------------


#----------------------------------------------------------------
# Lasso tries to:
# Minimize error + penalty that can force some coefficients to 0.

# Lasso can:
# Make some coefficients exactly 0.
# If a feature is useless → coefficient becomes 0.
#----------------------------------------------------------------


#==============================================================================
# Alpha is a hyperparameter that controls the strength of the regularization.

# If alpha increases:
# Coefficients shrink more
# Model becomes simpler
# Variance reduces
# Overfitting decreases

# So:
# 👉 If overfitting → increase alpha
# 👉 If underfitting → decrease alpha

# Important concept:
# Alpha ↑ → Bias ↑, Variance ↓
# Alpha ↓ → Bias ↓, Variance ↑

# This is bias-variance tradeoff in action.
#==============================================================================


def model_linear(preprocessor, X_train, y_train, X_test, y_test):
        pipeline = build_pipeline(preprocessor, model_type="linear")
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        model = pipeline
        rsq = model.score(X_train, y_train)
        rsq2 = model.score(X_test, y_test)
        print("-" * 35)
        print("Model trained successfully....")
        print(f"Training score: {rsq * 100:.2f}%")
        print(f"Testing score: {rsq2 * 100:.2f}%")
        print()
        print("-" * 35, "\n")

        return pred, model


# This is a function to load the config.yaml file which contains the hyperparameters
def load_config():
    import os
    base_path = os.path.dirname(__file__)
    config_path = os.path.join(base_path, "config.yaml")

    with open(config_path, "r") as fi:
         config = yaml.safe_load(fi)

    return config


def model_ridge(preprocessor, X_train, y_train, X_test, y_test):
        config = load_config()
        alpha_grid = config["model"]["alpha_grid"]
        param_grid = {"model__alpha": alpha_grid}
        pipeline = build_pipeline(preprocessor, model_type="ridge")
        model = pipeline

        # param_grid = {"model__alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        grid_search = GridSearchCV(
        estimator=model,
        param_grid = param_grid,
        cv=5,                # 5-fold cross-validation
        scoring= "r2",  # evaluation metric
        n_jobs=-1            # use all CPU cores
        )

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        pred = best_model.predict(X_test)
        model = best_model
        # print("-" * 35)
        # print("Best alpha:", grid_search.best_params_)
        # print("Best CV score:", grid_search.best_score_)
        # print("Test R2:", best_model.score(X_test, y_test))
        # print("-" * 35 , "\n")

        r2 = best_model.score(X_test, y_test)
        Tr2 = best_model.score(X_train, y_train)

        return pred, model, grid_search.best_params_, grid_search.best_score_, r2, Tr2


def model_lasso(preprocessor, X_train, y_train, X_test, y_test):
        pipeline = build_pipeline(preprocessor, model_type="lasso")
        model = pipeline
        param_grid = {"model__alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        grid_search = GridSearchCV(
            estimator=model,
            param_grid = param_grid,
            cv=5,                # 5-fold cross-validation
            scoring= "r2",  # evaluation metric
            n_jobs=-1            # use all CPU cores
            )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        pred = best_model.predict(X_test)
        model = best_model
        print("-" * 35)
        print("Best alpha:", grid_search.best_params_)
        print("Best CV score:", grid_search.best_score_)
        print("Test R2:", best_model.score(X_test, y_test))
        print("-" * 35 , "\n")

        return pred, model

def model_rf(preprocessor, X_train, y_train, X_test, y_test):
        pipeline = build_pipeline(preprocessor, model_type="rf")
        pipeline.fit(X_train, y_train)
        model = pipeline
        pred = model.predict(X_test)
        rsq = model.score(X_train, y_train)
        rsq2 = model.score(X_test, y_test)
        print("-" * 35)
        print("Model trained successfully....")
        print(f"Training score: {rsq * 100:.2f}%")
        print(f"Testing score: {rsq2 * 100:.2f}%")
        print()
        print("-" * 35, "\n")

        return pred, model



#--------------------
# For API
#--------------------


def train_ridge_model(config):
        import logging
        from datetime import datetime, timezone
        import json
        from pathlib import Path

        BASE_DIR = Path(__file__).resolve().parents[2] #* goes to `House-Price-Prediction`
        MODELS_DIR = BASE_DIR / "models"
        VERSIONS_DIR = MODELS_DIR / "versions"
        REPORTS_DIR = BASE_DIR / "reports"

        #* create models and versions directory if it doesn't exist
        MODELS_DIR.mkdir(exist_ok=True)
        VERSIONS_DIR.mkdir(exist_ok=True)
        REPORTS_DIR.mkdir(exist_ok=True)

        # base_path = os.path.dirname(__file__)
        # os.makedirs(os.path.join(base_path, "..", "reports"), exist_ok=True) 

        # save_path = os.path.join(base_path, "..", "reports", "train_summary.json")
        save_path = REPORTS_DIR /"train_summary.json"
        # save_model_path = os.path.join(base_path, "..", "reports", "model_summary.json")
        save_model_path = REPORTS_DIR / "model_summary.json"


        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # getiing parameters from config.yaml file
        test_size = config["train"]["test_size"]
        randomstate = config["train"]["random_state"]
        

        from house_price_prediction.loader import load_csv, save_model
        from house_price_prediction.preprocess import build_preprocessor
        from sklearn.model_selection import train_test_split

        df = load_csv()

        logger.info("Training started")

        features, target, preprocessor = build_preprocessor(df)

        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=randomstate
            )

        _ , model, best_params, cv_scores, r2, Tr2 = model_ridge(preprocessor, X_train, y_train, X_test, y_test)
        
        time_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_filename = f"ridge_{time_stamp}.joblib"
        verions_filename = "latest.joblib"

        verions_path = VERSIONS_DIR / model_filename
        models_path = MODELS_DIR / verions_filename
        
        save_model(model, verions_path)
        save_model(model, models_path)


        model_summary = {
              "model_file": model_filename
        }

        logger.info("Best alpha: %s", best_params)
        logger.info("Best CV score: %s", cv_scores)
        logger.info("Test R2: %s", r2)
        logger.info("Training R2: %s", Tr2)

        summary = {
            "model_type": "ridge",
            "best_alpha": best_params["model__alpha"],
            "cv_score": cv_scores,
            "test_r2": r2,
            "train_r2": Tr2,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
            
        with open (save_path, "w") as file:
            json.dump(summary, file, indent=4)

        logger.info("Training completed")

        with open(save_model_path, "w") as file:
              json.dump(model_summary, file, indent=4)
            
        logger.info("Model saved successfully")
        

def train():
    # geting model type from config.yaml
    config = load_config()
    model_type = config["model"]["type"]

    if model_type == "ridge":
        train_ridge_model(config)
    
    elif model_type == "lasso":
        pass
    
    elif model_type == "rf":
        pass
    
    else:
        raise ValueError("Invalid model type")




if __name__ == "__main__":
    train()
