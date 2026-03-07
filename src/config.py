import os
from dotenv import load_dotenv

load_dotenv()

class Config:

    # Project Paths
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    DATA_PATH = os.path.join(PROJECT_ROOT, "data")
    MODEL_PATH = os.path.join(PROJECT_ROOT, "artifacts/models")
    VECTOR_DB_PATH = os.path.join(PROJECT_ROOT, "artifacts/vector_store/chroma_store")

    # MLflow Configuration
    MLFLOW_TRACKING_URI = os.getenv(
        "MLFLOW_TRACKING_URI",
        os.path.join(PROJECT_ROOT, "artifacts/mlruns")
    )

    MLFLOW_EXPERIMENT_REGRESSION = "regression_experiment"
    MLFLOW_EXPERIMENT_DL = "deeplearning_experiment"

    # Chatbot / LLM Config
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Model Names
    BEST_REG_PIPE_FILE = "best_reg_pipe.joblib"
    BEST_CLASSIFICATION_PIPE_FILE = "best_classification_pipe.joblib"
    REGRESSION_MODEL_FILE = "regression_best_model.pkl"
    DL_MODEL_FILE = "model3_cnn_vit_tuned_best.model.keras"

    # Streamlit App Config
    STREAMLIT_PORT = 5000