import numpy as np
import tensorflow as tf
import joblib
from typing import List, Sequence, Dict, Any
import logging

logger = logging.getLogger("forecasting")


class ForecastEngine:
    """Singleton wrapper around the TensorFlow model and scaler.

    NOTE: This class is intentionally pure-temporal. It performs only LSTM
    forecasting, scaling, and the multi-horizon sliding-window rollout.
    Spatial propagation was moved to the `services` layer to enforce
    separation of concerns.
    """

    _instance = None
    sequence_length: int = 20

    def __init__(self) -> None:
        from app.config import settings

        self.sequence_length = settings.sequence_length
        self.horizon = settings.horizon
        # model loading and scaler behavior unchanged
        self.model = tf.keras.models.load_model(settings.model_path)
        self.scaler = joblib.load(settings.scaler_path)

    @classmethod
    def get_instance(cls) -> "ForecastEngine":
        if cls._instance is None:
            cls._instance = ForecastEngine()
        return cls._instance

    def _prepare_batch(self, arr: np.ndarray) -> np.ndarray:
        """Ensure input is 3D: (batch, seq_len, features)."""
        if arr.ndim == 2:
            return arr.reshape(1, arr.shape[0], arr.shape[1])
        if arr.ndim == 3:
            return arr
        raise ValueError("input must be 2D (seq,feat) or 3D (batch,seq,feat)")

    def _inverse_scale_single(self, scaled_values: Sequence[float], feature_index: int = 0) -> List[float]:
        scaled_vals = np.array(scaled_values).reshape(-1, 1)
        n_features = self.scaler.n_features_in_
        if n_features == 1:
            inv = self.scaler.inverse_transform(scaled_vals)
            return list(inv.flatten())

        filler = np.zeros((scaled_vals.shape[0], n_features))
        filler[:, feature_index:feature_index + 1] = scaled_vals
        inv = self.scaler.inverse_transform(filler)
        return list(inv[:, feature_index])

    def _scale_input(self, arr: np.ndarray) -> np.ndarray:
        shape = arr.shape
        flat = arr.reshape(-1, shape[-1])
        scaled = self.scaler.transform(flat)
        return scaled.reshape(shape)

    def predict(self, data: np.ndarray, horizon: int | None = None) -> Dict[str, Any]:
        """Predict next `horizon` timesteps for a single sequence or batch.

        Returns a dictionary: {"predictions": List[List[float]]} where each inner list
        is the horizon predictions for a single input sequence. This method does NOT
        perform any spatial propagation.
        """

        if horizon is None:
            horizon = self.horizon

        arr = np.asarray(data)
        if arr.ndim == 2:
            if arr.shape[0] != self.sequence_length:
                raise ValueError(f"input length {arr.shape[0]} != {self.sequence_length}")
        elif arr.ndim == 3:
            if arr.shape[1] != self.sequence_length:
                raise ValueError(f"input length {arr.shape[1]} != {self.sequence_length}")
        else:
            raise ValueError("input must be 2D or 3D ndarray")

        batch = self._prepare_batch(arr)

        # scale inputs only (no leakage from future)
        scaled_batch = self._scale_input(batch)

        batch_preds_scaled: List[List[float]] = []

        for seq in scaled_batch:
            seq_preds_scaled: List[float] = []
            window = seq.copy()
            for _ in range(horizon):
                x = window.reshape(1, window.shape[0], window.shape[1])
                y_scaled = self.model.predict(x, verbose=0)
                next_scaled = float(y_scaled[0][0])
                seq_preds_scaled.append(next_scaled)
                n_feat = window.shape[1]
                new_row = np.zeros((1, n_feat))
                new_row[0, 0] = next_scaled
                window = np.vstack([window[1:], new_row])

            batch_preds_scaled.append(seq_preds_scaled)

        batch_preds = [self._inverse_scale_single(p, feature_index=0) for p in batch_preds_scaled]

        return {"predictions": batch_preds}

    def predict_batch(self, batch_data: np.ndarray, horizon: int | None = None) -> Dict[str, Any]:
        return self.predict(batch_data, horizon=horizon)