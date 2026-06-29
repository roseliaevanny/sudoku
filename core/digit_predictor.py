import logging
import os
from typing import List, Optional, Sequence

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

# Resolve relative to this file, not the current working directory, so the
# default path works no matter where the caller's script is run from.
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "digit_model.keras"
)


class DigitPredictor:
    """Loads a trained Keras CNN model and predicts digits for cell images."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        try:
            self.model = tf.keras.models.load_model(model_path)
        except Exception as e:
            raise IOError(f"Failed to load model from {model_path}: {e}") from e
        logger.info("Model loaded from %s", model_path)

    def predict_cells(self, cells: Sequence[Optional[np.ndarray]]) -> List[List[int]]:
        """
        Args:
            cells: 81 cell images in row-major order, each a 28x28 array,
                   or None where OpenCV detected an empty cell.

        Returns:
            9x9 nested list of predicted digits (0 = empty cell).
        """
        if len(cells) != 81:
            raise ValueError(f"Expected 81 cells, got {len(cells)}")

        non_empty_idx = [i for i, cell in enumerate(cells) if cell is not None]
        board_flat = [0] * 81

        if non_empty_idx:
            batch = np.stack([cells[i] for i in non_empty_idx]).astype("float32")

            # Normalize to [0, 1] to match training preprocessing
            # (X_train / 255.0 in the training notebook).
            if batch.max() > 1.0:
                batch /= 255.0

            batch = np.expand_dims(batch, axis=-1)  # (N, 28, 28) -> (N, 28, 28, 1)

            # One batched call instead of 81 individual ones -- model.predict
            # has fixed per-call overhead, so batching is dramatically faster.
            predictions = self.model.predict(batch, verbose=0)
            digits = np.argmax(predictions, axis=1)

            for i, digit in zip(non_empty_idx, digits):
                digit = int(digit)
                if digit == 0:
                    # clean_and_prepare_box already filters out empty cells
                    # before they reach the model, so every cell here is
                    # supposed to contain a real digit (1-9). A predicted 0
                    # means the model misclassified a digit as blank -- log
                    # it so it doesn't silently merge with genuinely empty
                    # cells on the board.
                    logger.warning("Cell %d: model predicted class 0 (blank) "
                                    "for a non-empty cell -- likely misclassification.", i)
                board_flat[i] = digit

        return [board_flat[r * 9 : (r + 1) * 9] for r in range(9)]