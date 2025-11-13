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

from .sudoku_board import SudokuBoard, MiniSudokuBoard6x6, MiniSudokuBoard4x4

_LEVEL_MAP = {0: "easy", 1: "medium", 2: "hard", 3: "expert"}

def _grid_to_multiline_string(grid, empty_char: str = "#") -> str:
    """Convert a square matrix (6x6, 9x9, ...) to N lines, using '#' for zeros."""
    size = len(grid)
    lines = []
    for r in range(size):
        line = "".join(str(grid[r][c]) if grid[r][c] != 0 else empty_char for c in range(size))
        lines.append(line)
    return "\n".join(lines)


def generateSudokus(
    number: int,
    level: int,
    *,
    rng_seed: Optional[int] = None,
    size: int = 9,
) -> List[Dict[str, str]]:
    """
    Generate `number` sudokus of difficulty `level` (0..3).

    - size=9 → sudoku clásico 9x9 usando SudokuBoard.generateGameBoardByDifficulty
    - size=6 → mini-sudoku 6x6 (bloques 2x3, números 1-6, 8-12 pistas, solución única)
    - size=4 → mini-sudoku 4x4 (bloques 2x2, números 1-4, 7-9 pistas, solución única)

    Returns a list of dictionaries:
      { "level": "0|1|2|3", "board": "<N lines with #>", "solution": "<N lines with digits>" }
    """
    if level not in _LEVEL_MAP:
        raise ValueError("level must be one of {0,1,2,3}")
    if number <= 0:
        return []
    if size not in (4, 6, 9):
        raise ValueError("size must be one of {4, 6, 9}")

    level_str = _LEVEL_MAP[level]
    out: List[Dict[str, str]] = []

    for i in range(number):
        if size == 9:
            # Sudoku clásico, dejamos TODO igual que antes
            sb = SudokuBoard()
            full_board, puzzle_board = sb.generateGameBoardByDifficulty(
                level=level_str,
                rng_seed=rng_seed,
            )
        elif size == 6:
            # Mini-sudoku 6x6:
            # - Bloques 2x3 (3 columnas de bloques x 2 filas de bloques)
            # - Números 1..6
            # - Entre 8 y 12 pistas
            sb = MiniSudokuBoard6x6()
            per_seed = None
            if rng_seed is not None:
                # pequeña variación por sudoku para no repetir siempre el mismo
                per_seed = rng_seed + i
            full_board, puzzle_board = sb.generate_puzzle(
                clue_min=8,
                clue_max=12,
                rng_seed=per_seed,
            )
        else:
            # Mini-sudoku 4x4:
            # - Bloques 2x2
            # - Números 1..4
            # - Entre 7 y 9 pistas
            sb = MiniSudokuBoard4x4()
            per_seed = None
            if rng_seed is not None:
                per_seed = rng_seed + i
            full_board, puzzle_board = sb.generate_puzzle(
                clue_min=7,
                clue_max=9,
                rng_seed=per_seed,
            )

        # `board` → matriz (4x4, 6x6 o 9x9) a string multilínea con '#'
        puzzle_str = _grid_to_multiline_string(puzzle_board.board, empty_char="#")
        # `solution` → misma matriz pero sin ceros (no debería haberlos)
        solution_str = _grid_to_multiline_string(full_board.board, empty_char="#").replace("#", "")

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
