import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.solver import solve_sudoku


def print_board(board):
    for row in board:
        print(" ".join(str(val) if val != 0 else "." for val in row))
    print()


def test_solver_easy():
    board = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]

    print("Initial Board:")
    print_board(board)

    success = solve_sudoku(board)

    if success:
        print("Solved Board:")
        print_board(board)
    else:
        print("Failed to solve the board.")

    assert success == True
    assert board[0] == [5, 3, 4, 6, 7, 8, 9, 1, 2]


def test_solver_sudoku_png():
    board = [
        [0, 0, 2, 1, 0, 8, 0, 7, 0],
        [0, 8, 0, 0, 0, 0, 1, 0, 4],
        [0, 0, 1, 0, 3, 5, 8, 0, 2],
        [3, 0, 0, 9, 0, 0, 0, 2, 8],
        [9, 5, 0, 8, 0, 2, 0, 1, 0],
        [1, 2, 8, 3, 0, 7, 0, 4, 9],
        [2, 0, 9, 5, 8, 0, 4, 0, 0],
        [6, 0, 0, 0, 0, 0, 2, 8, 0],
        [8, 1, 0, 0, 0, 3, 9, 0, 0]
    ]

    print("Initial Board:")
    print_board(board)

    success = solve_sudoku(board)

    if success:
        print("Solved Board:")
        print_board(board)
    else:
        print("Failed to solve the board.")

    assert success == True


def test_solver_rejects_invalid_board():
    # Two 5s in the same row -- should fail cleanly, not raise or hang.
    board = [
        [5, 5, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]

    success = solve_sudoku(board)

    if success:
        print("Solved Board:")
        print_board(board)
    else:
        print("Failed to solve the board. Initial Board:")
        print_board(board)

    assert success == False


if __name__ == "__main__":
    test_solver_easy()
    test_solver_sudoku_png()
    test_solver_rejects_invalid_board()
    print("All tests passed!")