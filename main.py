import argparse
import os
import sys
import cv2

from core import extract_digit_cells, DigitPredictor, solve_sudoku


def print_board(board):
    """Utility function to print the Sudoku board cleanly to the terminal."""
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - -")
        for j in range(9):
            if j % 3 == 0 and j != 0:
                print(" | ", end="")
            if j == 8:
                print(board[i][j])
            else:
                print(str(board[i][j]) + " ", end="")


def main():
    default_img = os.path.join(os.path.dirname(__file__), "assets", "sudoku.png")
    parser = argparse.ArgumentParser(description="End-to-end AI Sudoku solver.")
    parser.add_argument(
        "image", nargs="?", default=default_img,
        help=f"Path to a Sudoku photo (default: {default_img})",
    )
    img_path = parser.parse_args().image

    if not os.path.exists(img_path):
        print(f"Error: Target image not found at {img_path}")
        sys.exit(1)

    print("\n[1/3] Extracting cells from image")
    try:
        img = cv2.imread(img_path)
        cells = extract_digit_cells(img)
    except Exception as e:
        print(f"CV Extraction failed: {e}")
        sys.exit(1)

    clue_count = sum(1 for cell in cells if cell is not None)
    if clue_count < 17:
        # 17 givens is the mathematical minimum for a uniquely solvable
        # Sudoku -- fewer than that means the grid was very likely
        # misread, not that you have an unusually sparse puzzle.
        print(
            f"Warning: only {clue_count} digits detected (minimum for a "
            "solvable puzzle is 17). The grid detection may have misread "
            "the photo."
        )

    print("[2/3] Loading CNN model and predicting digits")
    try:
        predictor = DigitPredictor()
        board = predictor.predict_cells(cells)
    except Exception as e:
        print(f"AI Prediction failed: {e}")
        sys.exit(1)

    print("\nDetected Initial Board")
    print_board(board)

    print("\n[3/3] Solving puzzle")
    if solve_sudoku(board):
        print_board(board)
        print("Sudoku solved.")
    else:
        print(
            "\nFailed to solve. The puzzle is either invalid or the AI misclassified a digit."
        )


if __name__ == "__main__":
    main()