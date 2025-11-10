# -*- coding: utf-8 -*-
"""
Public API of the package:
- generateSudokus(number, level) -> list[dict]
- sudokus2CSV(sudokus, date, filename=None) -> str (path to the CSV)
"""

from __future__ import annotations
import csv
from datetime import date as Date, datetime, timedelta
from typing import Iterable, List, Dict, Optional, Union

from .sudoku_board import SudokuBoard 

_LEVEL_MAP = {0: "easy", 1: "medium", 2: "hard", 3: "expert"}

def _grid_to_multiline_string(grid, empty_char: str = "#") -> str:
    """Convert the 9x9 matrix (ints) to 9 lines, using '#' for zeros."""
    lines = []
    for r in range(9):
        line = "".join(str(grid[r][c]) if grid[r][c] != 0 else empty_char for c in range(9))
        lines.append(line)
    return "\n".join(lines)

def generateSudokus(number: int, level: int, *, rng_seed: Optional[int] = None) -> List[Dict[str, str]]:
    """
    Generate `number` sudokus of difficulty `level` (0..3) using the ORIGINAL
    `generateGameBoardByDifficulty` generator from SudokuBoard.
    Returns a list of dictionaries:
      { "level": "0|1|2|3", "board": "<9 lines with #>", "solution": "<9 lines with digits>" }
    """
    if level not in _LEVEL_MAP:
        raise ValueError("level must be one of {0,1,2,3}")
    if number <= 0:
        return []

    level_str = _LEVEL_MAP[level]
    out: List[Dict[str, str]] = []

    for _ in range(number):
        sb = SudokuBoard()
        full_board, puzzle_board = sb.generateGameBoardByDifficulty(level=level_str, rng_seed=rng_seed)

        # puzzle_board and full_board are instances of SudokuBoard with .board (9x9 list)
        puzzle_str = _grid_to_multiline_string(puzzle_board.board, empty_char="#")
        solution_str = _grid_to_multiline_string(full_board.board, empty_char="#").replace("#", "")  # 81 digits in 9 lines

        out.append({
            "level": str(level),
            "board": puzzle_str,
            "solution": solution_str,
        })

    return out

def sudokus2CSV(
    sudokus: Iterable[Dict[str, str]],
    date: Union[str, Date],
    *,
    filename: Optional[str] = None
) -> str:
    """
    Write a comma-separated CSV with columns:
      publication_date, level, board, solution
    - `date`: starting date (YYYY-MM-DD or datetime.date). Each row increases by +1 day.
    - `filename`: optional; if not provided, uses 'sudoku_{date}_{N}.csv'.
    Returns the path to the created file.
    """
    # Normalize date
    if isinstance(date, str):
        start_date = datetime.strptime(date, "%Y-%m-%d").date()
    elif isinstance(date, Date):
        start_date = date
    else:
        raise TypeError("date must be 'YYYY-MM-DD' string or datetime.date")

    rows = list(sudokus)
    n = len(rows)

    if filename is None:
        filename = f"sudoku_{start_date.isoformat()}_{n}.csv"

    # CSV with commas and quoting for multiline fields
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["publication_date", "level", "board", "solution"])

        for i, s in enumerate(rows):
            pub_date = start_date + timedelta(days=i)
            writer.writerow([
                pub_date.isoformat(),
                s.get("level", ""),
                s.get("board", ""),
                s.get("solution", ""),
            ])

    return filename
