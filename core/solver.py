from typing import List, Tuple

Board = List[List[int]]


def box_index(r: int, c: int) -> int:
    return (r // 3) * 3 + (c // 3)


def build_constraints(board: Board):
    """Bitmasks of digits already used in each row, column, and box."""
    rows, cols, boxes = [0] * 9, [0] * 9, [0] * 9
    for r in range(9):
        for c in range(9):
            val = board[r][c]
            if val:
                bit = 1 << val
                rows[r] |= bit
                cols[c] |= bit
                boxes[box_index(r, c)] |= bit
    return rows, cols, boxes


def candidates(rows, cols, boxes, r: int, c: int) -> List[int]:
    used = rows[r] | cols[c] | boxes[box_index(r, c)]
    return [v for v in range(1, 10) if not used & (1 << v)]


def is_valid_board(board: Board) -> bool:
    """Sanity check: the given digits (ignoring 0s) contain no conflicts."""
    rows, cols, boxes = [0] * 9, [0] * 9, [0] * 9
    for r in range(9):
        for c in range(9):
            val = board[r][c]
            if val == 0:
                continue
            bit = 1 << val
            b = box_index(r, c)
            if rows[r] & bit or cols[c] & bit or boxes[b] & bit:
                return False
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit
    return True


def solve_sudoku(board: Board) -> bool:
    """
    Solves a Sudoku puzzle in place using backtracking + MRV.

    Args:
        board: 9x9 list of lists of ints, 0 = empty cell.

    Returns:
        True if solved (board is mutated in place with the solution),
        False if no solution exists.
    """
    if not is_valid_board(board):
        return False

    rows, cols, boxes = build_constraints(board)
    empties = [(r, c) for r in range(9) for c in range(9) if board[r][c] == 0]

    def backtrack(remaining: List[Tuple[int, int]]) -> bool:
        if not remaining:
            return True

        # MRV: pick the empty cell with the fewest legal candidates.
        best_i, best_cell, best_cands = -1, None, None
        for i, (r, c) in enumerate(remaining):
            cands = candidates(rows, cols, boxes, r, c)
            if not cands:
                return False  # dead end -- backtrack now
            if best_cands is None or len(cands) < len(best_cands):
                best_i, best_cell, best_cands = i, (r, c), cands
                if len(cands) == 1:
                    break  # can't do better than a forced move

        r, c = best_cell
        rest = remaining[:best_i] + remaining[best_i + 1:]

        for val in best_cands:
            bit = 1 << val
            b = box_index(r, c)
            board[r][c] = val
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit

            if backtrack(rest):
                return True

            board[r][c] = 0
            rows[r] &= ~bit
            cols[c] &= ~bit
            boxes[b] &= ~bit

        return False

    return backtrack(empties)