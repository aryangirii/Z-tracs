from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    environment: str = "development"
    sequence_length: int = 20
    horizon: int = 5

    model_path: str = "models/urbanx_lstm_model.keras"
    scaler_path: str = "models/urbanx_scaler.pkl"

    use_graph_propagation: bool = False
    graph_alpha: float = 0.3
    graph_iterations: int = 1
    adjacency_matrix_path: str = "models/adjacency_matrix.npy"

    model_version: str = "1.0"

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()