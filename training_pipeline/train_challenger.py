import os
import pandas as pd
import pathlib
import pickle
from dotenv import load_dotenv
from sklearn.feature_extraction import DictVectorizer
import math
import optuna
from optuna.samplers import TPESampler
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
from prefect import flow, task
import mlflow.pyfunc

# =======================
# Tasks
# =======================

@task(name="Read Data")
def read_data(file_path: str) -> pd.DataFrame:
    """Read data into DataFrame"""
    df = pd.read_parquet(file_path)

    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    return df

@task(name="Add Features")
def add_features(df_train: pd.DataFrame, df_val: pd.DataFrame):
    """Add features to the model"""
    df_train["PU_DO"] = df_train["PULocationID"] + "_" + df_train["DOLocationID"]
    df_val["PU_DO"] = df_val["PULocationID"] + "_" + df_val["DOLocationID"]

    categorical = ["PU_DO"]  #'PULocationID', 'DOLocationID']
    numerical = ["trip_distance"]

    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values
    return X_train, X_val, y_train, y_val, dv

@task(name="RandomForestRegressor Model")
def forest_run(X_train, X_val, y_train, y_val, dv):
    def objective(trial: optuna.trial.Trial):
    # Hiperparámetros muestreados por Optuna
        params = {
            "max_features": trial.suggest_float("max_features", math.exp(-5), math.exp(-1), log=True),
            "n_estimators": trial.suggest_int("n_estimators", 5, 100),
            "criterion": "absolute_error",
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 0.5),
            "random_state": 42,
            "n_jobs": -1
        }

        # Run anidado de MLflow
        with mlflow.start_run(nested=True):
            mlflow.set_tag("model_family", "RandomForestRegressor")
            mlflow.log_params(params)

            # Entrenamiento
            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)

            # Predicción y métrica
            y_pred = model.predict(X_val)
            rmse = root_mean_squared_error(y_val, y_pred)

            # Registrar métrica en MLflow
            mlflow.log_metric("rmse", rmse)

            # Guardar modelo en MLflow
            signature = infer_signature(X_val, y_pred)
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                input_example=X_val[:5],
                signature=signature
            )

        # Optuna minimizará este valor
        return rmse
    
    mlflow.sklearn.autolog(log_models=False)

    # ------------------------------------------------------------
    # Crear el estudio de Optuna
    # ------------------------------------------------------------
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # ------------------------------------------------------------
    # Ejecutar la optimización
    # ------------------------------------------------------------
    with mlflow.start_run(run_name="RandomForest Hyperparameter Optimization (Optuna)"):
        study.optimize(objective, n_trials=3)

        # --------------------------------------------------------
        # Recuperar y registrar los mejores hiperparámetros
        # --------------------------------------------------------
        best_params = study.best_params
        best_params["random_state"] = 42
        best_params["n_jobs"] = -1

        mlflow.log_params(best_params)

        # Etiquetas del run "padre"
        mlflow.set_tags({
            "project": "NYC Taxi Time Prediction Project",
            "optimizer_engine": "optuna",
            "model_family": "RandomForestRegressor",
            "feature_set_version": 1,
        })

        final_modelrf = RandomForestRegressor(**best_params)
        final_modelrf.fit(X_train, y_train)

        # Evaluar en validación
        y_pred = final_modelrf.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        pathlib.Path("preprocessor").mkdir(exist_ok=True)
        with open("preprocessor/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("preprocessor/preprocessor.b", artifact_path="preprocessor")

        # --------------------------------------------------------
        # Registrar el modelo final en MLflow
        # --------------------------------------------------------
        feature_names = dv.get_feature_names_out()
        input_example = pd.DataFrame(X_val[:5].toarray(), columns=feature_names)
        signature = infer_signature(input_example, y_val[:5])

        mlflow.sklearn.log_model(
            sk_model=final_modelrf,
            artifact_path="model",
            input_example=input_example,
            signature=signature
        )
    return None

@task(name="GradientBoosting Model")
def gradient_run(X_train, X_val, y_train, y_val, dv):

    def objective(trial: optuna.trial.Trial):
    # Hiperparámetros a optimizar
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "random_state": 42
        }

        # Run anidado en MLflow
        with mlflow.start_run(nested=True):
            mlflow.set_tag("model_family", "GradientBoostingRegressor")
            mlflow.log_params(params)

            # Entrenamiento
            model = GradientBoostingRegressor(**params)
            model.fit(X_train, y_train)

            # Predicción y métrica
            y_pred = model.predict(X_val)
            rmse = root_mean_squared_error(y_val, y_pred)
            mlflow.log_metric("rmse", rmse)

            # Guardar modelo
            signature = infer_signature(X_val, y_pred)
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=X_val[:5],
                signature=signature
            )

        return rmse
    
    mlflow.sklearn.autolog(log_models=False)

    # ------------------------------------------------------------
    # Crear el estudio de Optuna
    # ------------------------------------------------------------
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # ------------------------------------------------------------
    # Ejecutar la optimización
    # ------------------------------------------------------------
    with mlflow.start_run(run_name="GradientBoosting Hyperparameter Optimization (Optuna)"):
        study.optimize(objective, n_trials=5)

        # --------------------------------------------------------
        # Recuperar y registrar los mejores hiperparámetros
        # --------------------------------------------------------
        best_params = study.best_params
        best_params["random_state"] = 42
        mlflow.log_params(best_params)

        # Etiquetas del run padre
        mlflow.set_tags({
            "project": "NYC Taxi Time Prediction Project",
            "optimizer_engine": "optuna",
            "model_family": "GradientBoostingRegressor",
            "feature_set_version": 1,
        })

        # --------------------------------------------------------
        # Entrenar modelo final con los mejores parámetros
        # --------------------------------------------------------
        final_modelgb = GradientBoostingRegressor(**best_params)
        final_modelgb.fit(X_train, y_train)

        # Evaluar en validación
        y_pred = final_modelgb.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        # --------------------------------------------------------
        # Guardar artefactos adicionales (preprocesador)
        # --------------------------------------------------------
        pathlib.Path("preprocessor").mkdir(exist_ok=True)
        with open("preprocessor/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("preprocessor/preprocessor.b", artifact_path="preprocessor")

        # --------------------------------------------------------
        # Registrar el modelo final en MLflow
        # --------------------------------------------------------
        feature_names = dv.get_feature_names_out()
        input_example = pd.DataFrame(X_val[:5].toarray(), columns=feature_names)
        signature = infer_signature(input_example, y_val[:5])

        mlflow.sklearn.log_model(
            sk_model=final_modelgb,
            artifact_path="model",
            input_example=input_example,
            signature=signature
        )

    return None

@task(name='Register Model')
def register(EXPERIMENT_NAME):

    model_name = "workspace.default.nyc-taxi-model-prefect"
    # Load the Client for model registration
    client = MlflowClient()

    runs = mlflow.search_runs(
    experiment_names=[EXPERIMENT_NAME],
    order_by=["metrics.rmse ASC"],
    output_format="list"
    )

    # Obtener el mejor run
    if len(runs) > 0:
        best_run = runs[0]
        print("🏆 Champion Run encontrado:")
        print(f"Run ID: {best_run.info.run_id}")
        print(f"RMSE: {best_run.data.metrics['rmse']}")
        print(f"Params: {best_run.data.params}")
    else:
        print("⚠️ No se encontraron runs con métrica RMSE.")

    result = mlflow.register_model(
    model_uri=f"runs:/{best_run.info.run_id}/model",
    name=model_name
    )   

    model_version = result.version
    new_alias = "Champion"

    client.set_registered_model_alias(
        name=model_name,
        alias=new_alias,
        version=result.version
    )


@task(name='March eval')
def eval(EXPERIMENT_NAME, MX_train, MX_val, My_train, My_val, Mdv):

    model_name = "workspace.default.nyc-taxi-model-prefect"




    model_version_uri = f"models:/{model_name}@champion"

    champion_version = mlflow.pyfunc.load_model(model_version_uri)



    model_version_uri = f"models:/{model_name}@challenger"

    challenger_version = mlflow.pyfunc.load_model(model_version_uri)


    y_pred_champion = champion_version.predict(MX_train)
    y_pred_challenger = challenger_version.predict(MX_train)

    rmse_champion = root_mean_squared_error(My_train, y_pred_champion)
    rmse_challenger = root_mean_squared_error(My_train, y_pred_challenger)

    print(f"Champion RMSE:  {rmse_champion:.4f}")
    print(f"Challenger RMSE: {rmse_challenger:.4f}")




@flow(name="Main Challenger Flow")
def main_flow(year: int, month_train: str, month_val: str, march: str) -> None:
    """Main training pipeline for Challenger Models (RandomForest and Gradient Boosting)"""

    train_path = f"../data/green_tripdata_{year}-{month_train}.parquet"
    val_path = f"../data/green_tripdata_{year}-{month_val}.parquet"
    marzo_path = f"../data/green_tripdata_{year}-{march}.parquet"
    # Load .env file and experiment for Databricks
    load_dotenv(override=True)  
    EXPERIMENT_NAME = "/Users/oscar.josue2204@gmail.com/nyc-taxi-experiment-prefect"

    # Set MLFlow tracking to Databricks
    mlflow.set_tracking_uri("databricks")
    experiment = mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)

    # Load Data
    df_train = read_data(train_path)
    df_val = read_data(val_path)
    df_marzo = read_data(marzo_path)

    # Transform Data
    X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val)
    
    MX_train, MX_val, My_train, My_val, Mdv = add_features(df_train, dv)

    add_features(df_marzo, dv)
    # Tune and Train RandomForest
    forest_run(X_train, X_val, y_train, y_val, dv)
    
    # Tune and Train Gradient Boosting
    gradient_run(X_train, X_val, y_train, y_val, dv)

    # Model Registry
    register(EXPERIMENT_NAME)

    eval(MX_train, MX_val, My_train, My_val, Mdv)
# =======================
# Run
# =======================

if __name__ == "__main__":
    main_flow(year=2025, month_train="01", month_val="02", march="03")

