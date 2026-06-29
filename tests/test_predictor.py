import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.digit_predictor import DigitPredictor, DEFAULT_MODEL_PATH


def test_predictor_pipeline():
    predictor = DigitPredictor(model_path=DEFAULT_MODEL_PATH)

    # Deterministic mock cells: every 3rd cell is "empty" (None), the rest
    # are random 28x28 uint8 noise standing in for real digit crops. Seeded
    # so a failure is reproducible instead of depending on whatever random
    # values happened to come up on a given run.
    rng = np.random.default_rng(seed=42)
    empty_indices = set(range(0, 81, 3))
    mock_cells = [
        None if i in empty_indices else rng.integers(0, 256, (28, 28), dtype=np.uint8)
        for i in range(81)
    ]

    print("\nRunning batch inference on mock cells...")
    board = predictor.predict_cells(mock_cells)

    # Shape checks
    assert len(board) == 9, "Output board should have 9 rows"
    assert all(len(row) == 9 for row in board), "Each row should have 9 columns"

    for idx in range(81):
        row, col = divmod(idx, 9)
        value = board[row][col]

        if idx in empty_indices:
            assert value == 0, f"Cell {idx} was None but prediction wasn't 0"
        else:
            assert isinstance(value, int), (
                f"Cell {idx} prediction should be a plain int, got {type(value)}"
            )
            assert 0 <= value <= 9, (
                f"Cell {idx} prediction {value} is out of the valid digit range"
            )

    print("Predictor pipeline structural test passed successfully!")


if __name__ == "__main__":
    test_predictor_pipeline()