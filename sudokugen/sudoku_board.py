#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SudokuBoard: solve, generate, and ensure uniqueness for classic 9x9 Sudoku.

Public API:
- generateGameBoard(emptySpaces)
- generateGameBoardByDifficulty(level)
"""

import copy
import random

__author__ = 'Pedro Hernández / Modified by Daniel'
__version__ = '3.0.0'
__maintainer__ = 'Daniel'
__email__ = 'pedro.a.hdez.a@gmail.com'


class SudokuBoard:
    def __init__(self, board=None):
        self.__resetBoard()
        if board:
            for i in range(9):
                for j in range(9):
                    self.board[i][j] = int(board[(i * 9) + j])

    # Basic utilities
    def __resetBoard(self):
        """Fill the board with zeros."""
        self.board = [[0 for _ in range(9)] for _ in range(9)]

    def printBoard(self):
        """Print the board as 9 lines of 9 chars (0 shown as '#')."""
        for i in range(9):
            row_str = ""
            for j in range(9):
                cell = self.board[i][j]
                row_str += f"{cell if cell != 0 else '#'}"
            print(row_str.strip())

    def boardAsString(self):
        """Return the board as a flat 81-char string."""
        return "".join([str(col) for row in self.board for col in row])

    def __findEmptySpace(self, board=None, emptySpace=None):
        """Return (row, col) of an empty cell (0); optional nth empty via emptySpace."""
        b = board.board if board else self.board
        k = 0
        for row in range(9):
            for col in range(9):
                if b[row][col] == 0:
                    if emptySpace is None:
                        return (row, col)
                    if k == emptySpace:
                        return (row, col)
                    k += 1
        return None

    def __checkRules(self, num, space):
        """Check if placing `num` at `space` is valid under Sudoku rules."""
        r, c = space

        # Row
        for col in self.board[r]:
            if col == num:
                return False

        # Column
        for i in range(9):
            if self.board[i][c] == num:
                return False

        # 3x3 box
        br, bc = (r // 3) * 3, (c // 3) * 3
        for i in range(br, br + 3):
            for j in range(bc, bc + 3):
                if self.board[i][j] == num:
                    return False
        return True

    def solve(self, initialCell=None):
        """Plain backtracking over values 1..9."""
        availableSpace = initialCell or self.__findEmptySpace()
        if not availableSpace:
            return True
        r, c = availableSpace
        for n in range(1, 10):
            if self.__checkRules(n, (r, c)):
                self.board[r][c] = n
                if self.solve():
                    return True
                self.board[r][c] = 0
        return False

    def __generateFullBoard(self):
        """Generate a full valid board by seeding diagonal boxes, then solving."""
        self.__resetBoard()
        # Seed diagonal 3x3 boxes to speed up solving
        for box in range(0, 9, 3):
            nums = list(range(1, 10))
            random.shuffle(nums)
            k = 0
            for i in range(3):
                for j in range(3):
                    self.board[box + i][box + j] = nums[k]
                    k += 1
        self.solve()

    # Uniqueness counting
    def __findNumberOfSolutions(self, limit=None):
        """
        Return a list of solution strings (81 digits). If `limit` is given,
        stop once that many solutions are found.
        """
        solutions = []
        b = [row[:] for row in self.board]

        def find_empty(b_):
            for i in range(9):
                for j in range(9):
                    if b_[i][j] == 0:
                        return i, j
            return None

        def is_valid(b_, r, c, n):
            if any(b_[r][j] == n for j in range(9)): return False
            if any(b_[i][c] == n for i in range(9)): return False
            br, bc = (r // 3) * 3, (c // 3) * 3
            for i in range(br, br + 3):
                for j in range(bc, bc + 3):
                    if b_[i][j] == n: return False
            return True

        def backtrack():
            if limit is not None and len(solutions) >= limit:
                return
            empty = find_empty(b)
            if not empty:
                s = ''.join(str(b[i][j]) for i in range(9) for j in range(9))
                solutions.append(s)
                return
            r, c = empty
            for n in range(1, 10):
                if is_valid(b, r, c, n):
                    b[r][c] = n
                    backtrack()
                    if limit is not None and len(solutions) >= limit:
                        b[r][c] = 0
                        return
                    b[r][c] = 0

        backtrack()
        return solutions

    def __has_unique_solution(self):
        """Return True if the puzzle has exactly one solution."""
        return len(self.__findNumberOfSolutions(limit=2)) == 1

    # Generator: remove N cells with uniqueness 
    def generateGameBoard(self, emptySpaces=0):
        """
        Generate a puzzle with `emptySpaces` blanks while preserving uniqueness.
        Return (fullBoard, puzzleBoard). If emptySpaces=0 → (self, None).
        """
        self.__generateFullBoard()
        fullBoard = copy.deepcopy(self)

        emptied = 0
        attempts = 0
        max_attempts = 5000  # safety guard

        while emptied < emptySpaces and attempts < max_attempts:
            attempts += 1
            r, c = random.randint(0, 8), random.randint(0, 8)
            if self.board[r][c] == 0:
                continue
            keep = self.board[r][c]
            self.board[r][c] = 0
            # limit=2 speeds up uniqueness check
            if len(self.__findNumberOfSolutions(limit=2)) != 1:
                self.board[r][c] = keep
                continue
            emptied += 1

        if emptySpaces > 0:
            return fullBoard, self
        return self, None

    # Human-like solver + MRV/LCV backtracking with metrics
    def __box_index(self, r, c):
        return (r // 3) * 3 + (c // 3)

    def __compute_candidates(self, b):
        """Return 9x9 matrix of candidate sets for each empty cell."""
        cand = [[set() for _ in range(9)] for _ in range(9)]
        rows = [set() for _ in range(9)]
        cols = [set() for _ in range(9)]
        boxes = [set() for _ in range(9)]
        for r in range(9):
            for c in range(9):
                v = b[r][c]
                if v:
                    rows[r].add(v)
                    cols[c].add(v)
                    boxes[self.__box_index(r, c)].add(v)
        for r in range(9):
            for c in range(9):
                if b[r][c] == 0:
                    used = rows[r] | cols[c] | boxes[self.__box_index(r, c)]
                    cand[r][c] = set(range(1, 10)) - used
        return cand

    def __apply_naked_singles(self, b, cand):
        """Fill cells that have exactly one candidate."""
        changed = 0
        for r in range(9):
            for c in range(9):
                if b[r][c] == 0 and len(cand[r][c]) == 1:
                    v = next(iter(cand[r][c]))
                    b[r][c] = v
                    changed += 1
        return changed

    def __apply_hidden_singles(self, b, cand):
        """Fill numbers that appear only once in a unit (row/col/box)."""
        changed = 0

        # Rows
        for r in range(9):
            pos = {n: [] for n in range(1, 10)}
            for c in range(9):
                if b[r][c] == 0:
                    for n in cand[r][c]:
                        pos[n].append((r, c))
            for n, cells in pos.items():
                if len(cells) == 1:
                    rr, cc = cells[0]
                    b[rr][cc] = n
                    changed += 1

        # Columns
        for c in range(9):
            pos = {n: [] for n in range(1, 10)}
            for r in range(9):
                if b[r][c] == 0:
                    for n in cand[r][c]:
                        pos[n].append((r, c))
            for n, cells in pos.items():
                if len(cells) == 1:
                    rr, cc = cells[0]
                    b[rr][cc] = n
                    changed += 1

        # Boxes
        for br in range(0, 9, 3):
            for bc in range(0, 9, 3):
                pos = {n: [] for n in range(1, 10)}
                for r in range(br, br + 3):
                    for c in range(bc, bc + 3):
                        if b[r][c] == 0:
                            for n in cand[r][c]:
                                pos[n].append((r, c))
                for n, cells in pos.items():
                    if len(cells) == 1:
                        rr, cc = cells[0]
                        b[rr][cc] = n
                        changed += 1

        return changed

    def __apply_locked_candidates(self, b, cand):
        """
        Locked candidates (pointing & claiming):
        - Pointing: within a box, if all candidates of n lie in one row/col, remove n from that row/col outside the box.
        - Claiming: within a row/col, if all candidates of n lie in one box, remove n from the rest of that box.
        """
        changed = 0

        # Pointing (box -> row/col)
        for br in range(0, 9, 3):
            for bc in range(0, 9, 3):
                box_cells = [(r, c) for r in range(br, br + 3) for c in range(bc, bc + 3)]
                for n in range(1, 10):
                    spots = [(r, c) for (r, c) in box_cells if b[r][c] == 0 and n in cand[r][c]]
                    if not spots:
                        continue
                    rows = {r for (r, _) in spots}
                    cols = {c for (_, c) in spots}
                    if len(rows) == 1:
                        r0 = next(iter(rows))
                        for c in range(9):
                            if not (bc <= c < bc + 3) and b[r0][c] == 0 and n in cand[r0][c]:
                                cand[r0][c].remove(n)
                                changed += 1
                    if len(cols) == 1:
                        c0 = next(iter(cols))
                        for r in range(9):
                            if not (br <= r < br + 3) and b[r][c0] == 0 and n in cand[r][c0]:
                                cand[r][c0].remove(n)
                                changed += 1

        # Claiming (row/col -> box)
        # Rows
        for r in range(9):
            for n in range(1, 10):
                spots = [c for c in range(9) if b[r][c] == 0 and n in cand[r][c]]
                if not spots:
                    continue
                boxes = {(r // 3) * 3 + (c // 3) for c in spots}
                if len(boxes) == 1:
                    box = next(iter(boxes))
                    br = (box // 3) * 3
                    bc = (box % 3) * 3
                    for rr in range(br, br + 3):
                        for cc in range(bc, bc + 3):
                            if rr == r:
                                continue
                            if b[rr][cc] == 0 and n in cand[rr][cc]:
                                cand[rr][cc].remove(n)
                                changed += 1
        # Columns
        for c in range(9):
            for n in range(1, 10):
                spots = [r for r in range(9) if b[r][c] == 0 and n in cand[r][c]]
                if not spots:
                    continue
                boxes = {(r // 3) * 3 + (c // 3) for r in spots}
                if len(boxes) == 1:
                    box = next(iter(boxes))
                    br = (box // 3) * 3
                    bc = (box % 3) * 3
                    for rr in range(br, br + 3):
                        for cc in range(bc, bc + 3):
                            if cc == c:
                                continue
                            if b[rr][cc] == 0 and n in cand[rr][cc]:
                                cand[rr][cc].remove(n)
                                changed += 1

        return changed

    def __apply_naked_pairs(self, b, cand):
        """Remove pair values from peers when exactly two cells share the same two candidates."""
        changed = 0

        def sweep_cells(cells):
            nonlocal changed
            pairs = {}
            for (r, c) in cells:
                if b[r][c] == 0 and len(cand[r][c]) == 2:
                    key = tuple(sorted(cand[r][c]))
                    pairs.setdefault(key, []).append((r, c))
            for (a, b2), spots in pairs.items():
                if len(spots) == 2:
                    for (r, c) in cells:
                        if (r, c) not in spots and b[r][c] == 0:
                            if a in cand[r][c]:
                                cand[r][c].remove(a); changed += 1
                            if b2 in cand[r][c]:
                                cand[r][c].remove(b2); changed += 1

        # Rows
        for r in range(9):
            sweep_cells([(r, c) for c in range(9)])
        # Columns
        for c in range(9):
            sweep_cells([(r, c) for r in range(9)])
        # Boxes
        for br in range(0, 9, 3):
            for bc in range(0, 9, 3):
                sweep_cells([(r, c) for r in range(br, br + 3) for c in range(bc, bc + 3)])

        return changed

    def __human_solve(self, b):
        """
        Apply human techniques until no progress.
        Return:
        {
          "solved_without_guess": bool,
          "usage": {"naked_singles":n, "hidden_singles":n, "locked":n, "pairs":n},
          "stuck_after_steps": total
        }
        """
        usage = {"naked_singles": 0, "hidden_singles": 0, "locked": 0, "pairs": 0}
        steps = 0
        while True:
            cand = self.__compute_candidates(b)
            progress = 0

            placed = self.__apply_naked_singles(b, cand)
            if placed:
                usage["naked_singles"] += placed
                steps += placed
                progress += placed
                continue

            placed = self.__apply_hidden_singles(b, cand)
            if placed:
                usage["hidden_singles"] += placed
                steps += placed
                progress += placed
                continue

            eliminated = self.__apply_locked_candidates(b, cand)
            if eliminated:
                usage["locked"] += eliminated
                steps += eliminated
                progress += eliminated
                continue

            eliminated = self.__apply_naked_pairs(b, cand)
            if eliminated:
                usage["pairs"] += eliminated
                steps += eliminated
                progress += eliminated
                continue

            if all(b[r][c] != 0 for r in range(9) for c in range(9)):
                return {"solved_without_guess": True, "usage": usage, "stuck_after_steps": steps}

            return {"solved_without_guess": False, "usage": usage, "stuck_after_steps": steps}

    def __backtrack_with_metrics(self, b):
        """
        Backtracking with MRV (fewest candidates) + LCV (least constraining value).
        Return:
        {
          "solved": bool,
          "depth_needed": int|None,
          "nodes": int,
          "branching": int
        }
        """
        nodes = 0
        best_depth = [0]
        branching_score = 0

        def compute_cand_local(b_):
            return self.__compute_candidates(b_)

        def select_mrv(b_, cand):
            best = None
            best_set = None
            for r in range(9):
                for c in range(9):
                    if b_[r][c] == 0:
                        k = cand[r][c]
                        if not k:
                            return None
                        if best is None or len(k) < len(best_set):
                            best = (r, c)
                            best_set = k
            return best + (best_set,)

        def lcv_order(b_, r, c, cand):
            values = list(cand[r][c])

            def score(v):
                s = 0
                for cc in range(9):
                    if b_[r][cc] == 0 and cc != c and v in cand[r][cc]: s += 1
                for rr in range(9):
                    if b_[rr][c] == 0 and rr != r and v in cand[rr][c]: s += 1
                br, bc = (r // 3) * 3, (c // 3) * 3
                for rr in range(br, br + 3):
                    for cc in range(bc, bc + 3):
                        if b_[rr][cc] == 0 and not (rr == r and cc == c) and v in cand[rr][cc]:
                            s += 1
                return s

            values.sort(key=score)
            return values

        solved_board = None

        def dfs(b_, depth):
            nonlocal nodes, best_depth, branching_score, solved_board
            cand = compute_cand_local(b_)
            if all(b_[r][c] != 0 for r in range(9) for c in range(9)):
                if best_depth[0] == 0:
                    best_depth[0] = depth
                solved_board = [row[:] for row in b_]
                return True
            sel = select_mrv(b_, cand)
            if sel is None:
                return False
            r, c, opts = sel
            branching_score += len(opts)
            for v in lcv_order(b_, r, c, cand):
                nodes += 1
                b_[r][c] = v
                if dfs(b_, depth + 1):
                    return True
                b_[r][c] = 0
            return False

        bcopy = [row[:] for row in b]
        ok = dfs(bcopy, 0)
        return {
            "solved": ok,
            "depth_needed": best_depth[0] if ok else None,
            "nodes": nodes,
            "branching": branching_score
        }

    # Difficulty rating (hybrid) 
    def rateDifficulty(self):
        """
        Return (label, meta) where label ∈ {"easy","medium","hard","expert"}.
        Meta contains human usage and backtracking metrics.
        """
        b = [row[:] for row in self.board]
        human = self.__human_solve(b)
        if human["solved_without_guess"]:
            u = human["usage"]
            human_score = (
                u["naked_singles"] * 1 +
                u["hidden_singles"] * 3 +
                u["locked"] * 2 +
                u["pairs"] * 4
            )
            if human_score <= 40:
                return ("easy", {"human_usage": u, "stuck": False, "guess_depth": 0, "nodes": 0, "branching": 0})
            elif human_score <= 120:
                return ("medium", {"human_usage": u, "stuck": False, "guess_depth": 0, "nodes": 0, "branching": 0})
            else:
                return ("hard", {"human_usage": u, "stuck": False, "guess_depth": 0, "nodes": 0, "branching": 0})

        # Fallback to MRV+LCV when human solver gets stuck
        metr = self.__backtrack_with_metrics(self.board)
        depth = metr["depth_needed"] or 0
        is_expert = (depth >= 2) or (metr["branching"] >= 140)
        label = "expert" if is_expert else "hard"
        return (label, {
            "human_usage": human["usage"],
            "stuck": True,
            "guess_depth": depth,
            "nodes": metr["nodes"],
            "branching": metr["branching"],
        })

    # Symmetry + difficulty predicates 
    def __symmetrize_positions(self):
        """Return central-symmetric cell pairs (unique)."""
        seen = set()
        pairs = []
        for r in range(9):
            for c in range(9):
                if (r, c) in seen:
                    continue
                rr, cc = 8 - r, 8 - c
                seen.add((r, c)); seen.add((rr, cc))
                pairs.append(((r, c), (rr, cc)))
        random.shuffle(pairs)
        return pairs

    def __count_initial_singles(self):
        """Count naked singles in the current puzzle state."""
        b = [row[:] for row in self.board]
        cand = self.__compute_candidates(b)
        return sum(1 for r in range(9) for c in range(9) if b[r][c] == 0 and len(cand[r][c]) == 1)

    def __difficulty_predicate(self, level, meta):
        """Check if meta matches the requested difficulty level."""
        if level == "easy":
            return not meta.get("stuck", False)
        if level == "medium":
            return not meta.get("stuck", False)
        if level == "hard":
            return (not meta.get("stuck", False)) or (meta.get("guess_depth", 0) == 1)
        if level == "expert":
            return meta.get("stuck", False) and (meta.get("guess_depth", 0) >= 2 or meta.get("branching", 0) >= 140)
        return False

    # Difficulty-based generator with symmetry 
    def generateGameBoardByDifficulty(self, level="medium", max_tries=80, rng_seed=None):
        """
        Generate a unique symmetric puzzle filtered by human difficulty.
        Return (fullBoard, puzzleBoard). If exact match fails, return best attempt.
        """
        if rng_seed is not None:
            random.seed(rng_seed)

        level = level.lower()
        # Tunable clue ranges
        clues_ranges = {
            "easy": (36, 49),
            "medium": (30, 38),
            "hard": (26, 32),
            "expert": (20, 28),
        }

        if level not in clues_ranges:
            raise ValueError("Invalid level. Use easy|medium|hard|expert.")

        min_clues, max_clues = clues_ranges[level]

        # Start from a full valid board
        self.__generateFullBoard()
        fullBoard = copy.deepcopy(self)

        best_puzzle = None
        best_label_gap = 99  # 0 means exact match

        for _ in range(max_tries):
            # Reset puzzle to full
            self.board = copy.deepcopy(fullBoard.board)

            # Remove cells symmetrically to target a clue count
            pairs = self.__symmetrize_positions()
            clues_target = random.randint(min_clues, max_clues)
            to_remove = 81 - clues_target

            removed = 0
            for (a, b) in pairs:
                if removed >= to_remove:
                    break
                (ar, ac), (br, bc) = a, b
                if self.board[ar][ac] == 0 or self.board[br][bc] == 0:
                    continue
                keep_a = self.board[ar][ac]
                keep_b = self.board[br][bc]
                self.board[ar][ac] = 0
                self.board[br][bc] = 0

                if not self.__has_unique_solution():
                    self.board[ar][ac] = keep_a
                    self.board[br][bc] = keep_b
                    continue

                removed += 2

            # Fine-tune single removal if needed
            if removed < to_remove:
                positions = [(r, c) for r in range(9) for c in range(9)]
                random.shuffle(positions)
                for (r, c) in positions:
                    if removed >= to_remove:
                        break
                    if self.board[r][c] == 0:
                        continue
                    keep = self.board[r][c]
                    self.board[r][c] = 0
                    if self.__has_unique_solution():
                        removed += 1
                    else:
                        self.board[r][c] = keep

            # Clue range filter
            current_clues = sum(1 for r in range(9) for c in range(9) if self.board[r][c] != 0)
            if not (min_clues <= current_clues <= max_clues):
                continue

            # Avoid trivial “hard/expert”: limit initial singles
            singles0 = self.__count_initial_singles()
            if level in ("hard", "expert") and singles0 > 3:
                continue

            # Rate difficulty and accept if it matches
            label, meta = self.rateDifficulty()
            if self.__difficulty_predicate(level, meta):
                return fullBoard, copy.deepcopy(self)

            # Keep best near-match for fallback
            order = {"easy": 0, "medium": 1, "hard": 2, "expert": 3}
            gap = abs(order.get(label, 0) - order[level])
            if gap < best_label_gap:
                best_label_gap = gap
                best_puzzle = copy.deepcopy(self)

        # Fallback to best attempt or the current state
        if best_puzzle is not None:
            return fullBoard, best_puzzle
        return fullBoard, copy.deepcopy(self)

class MiniSudokuBoard6x6:
    """
    Mini-sudoku 6x6:
    - Size 6x6
    - Blocks of 3x2 (3 rows x 2 columns)
    - Numbers from 1 to 6
    - Generates full boards and puzzles with a unique solution
    """

    SIZE = 6
    BOX_ROWS = 3
    BOX_COLS = 2

    def __init__(self, board=None):
        self.board = [[0 for _ in range(self.SIZE)] for _ in range(self.SIZE)]
        if board:
            if len(board) != self.SIZE * self.SIZE:
                raise ValueError("Board string must have length 36 for 6x6 mini sudoku")
            for i in range(self.SIZE):
                for j in range(self.SIZE):
                    self.board[i][j] = int(board[i * self.SIZE + j])

    # --- Internal utilities ---

    def _find_empty(self, b):
        for r in range(self.SIZE):
            for c in range(self.SIZE):
                if b[r][c] == 0:
                    return r, c
        return None

    def _is_valid(self, b, r, c, n):
        # Row
        if any(b[r][cc] == n for cc in range(self.SIZE)):
            return False
        # Column
        if any(b[rr][c] == n for rr in range(self.SIZE)):
            return False
        # 2x3 block
        br = (r // self.BOX_ROWS) * self.BOX_ROWS
        bc = (c // self.BOX_COLS) * self.BOX_COLS
        for rr in range(br, br + self.BOX_ROWS):
            for cc in range(bc, bc + self.BOX_COLS):
                if b[rr][cc] == n:
                    return False
        return True

    def _solve_backtracking(self, b):
        empty = self._find_empty(b)
        if not empty:
            return True
        r, c = empty
        nums = list(range(1, self.SIZE + 1))
        random.shuffle(nums)
        for n in nums:
            if self._is_valid(b, r, c, n):
                b[r][c] = n
                if self._solve_backtracking(b):
                    return True
                b[r][c] = 0
        return False

    # --- Generate full board ---

    def generate_full_board(self, rng_seed=None):
        if rng_seed is not None:
            random.seed(rng_seed)
        self.board = [[0 for _ in range(self.SIZE)] for _ in range(self.SIZE)]
        self._solve_backtracking(self.board)

    # --- Solution counting to ensure uniqueness ---

    def _count_solutions(self, limit=None):
        solutions = 0
        b = [row[:] for row in self.board]

        def backtrack():
            nonlocal solutions
            if limit is not None and solutions >= limit:
                return
            empty = self._find_empty(b)
            if not empty:
                solutions += 1
                return
            r, c = empty
            for n in range(1, self.SIZE + 1):
                if self._is_valid(b, r, c, n):
                    b[r][c] = n
                    backtrack()
                    b[r][c] = 0

        backtrack()
        return solutions

    def has_unique_solution(self):
        """True if the puzzle has exactly one solution."""
        return self._count_solutions(limit=2) == 1

    # --- Generate 6x6 puzzle with 8-12 clues and unique solution ---

    def generate_puzzle(self, clue_min=8, clue_max=12, max_tries=200, rng_seed=None):
        """
        Generate a 6x6 puzzle with:
        - unique solution
        - number of clues between clue_min and clue_max (default 8-12)
        Returns (fullBoard, puzzleBoard), both MiniSudokuBoard6x6.
        """
        if clue_min > clue_max:
            raise ValueError("clue_min must be <= clue_max")

        last_full = None

        for attempt in range(max_tries):
            if rng_seed is not None:
                # vary the seed per attempt to avoid identical puzzles
                random.seed(rng_seed + attempt)

            self.generate_full_board()
            full = copy.deepcopy(self)
            last_full = full

            positions = [(r, c) for r in range(self.SIZE) for c in range(self.SIZE)]
            random.shuffle(positions)
            clues = self.SIZE * self.SIZE  # start fully filled (36 clues)

            for (r, c) in positions:
                if clues <= clue_min:
                    break
                keep = self.board[r][c]
                if keep == 0:
                    continue
                self.board[r][c] = 0

                if self.has_unique_solution():
                    clues -= 1
                    # continue trying to remove more, but not below clue_min
                    # and trying to approach clue_max or less
                    if clue_min <= clues <= clue_max:
                        # don't force-stop; let the loop continue
                        # while uniqueness holds and we don't go below clue_min
                        continue
                else:
                    # would break uniqueness -> undo
                    self.board[r][c] = keep

            final_clues = sum(
                1 for r in range(self.SIZE) for c in range(self.SIZE) if self.board[r][c] != 0
            )

            if clue_min <= final_clues <= clue_max and self.has_unique_solution():
                return full, copy.deepcopy(self)

        # Fallback: if we didn't achieve the range after max_tries,
        # return the last attempt (it will still have a unique solution).
        if last_full is None:
            # very unlikely, but safe fallback
            self.generate_full_board()
            last_full = copy.deepcopy(self)
        return last_full, copy.deepcopy(self)

class MiniSudokuBoard4x4:
    """
    Mini-sudoku 4x4:
    - Size 4x4
    - Blocks of 2x2
    - Numbers from 1 to 4
    - Generates full boards and puzzles with a unique solution
    """

    SIZE = 4
    BOX_ROWS = 2
    BOX_COLS = 2

    def __init__(self, board=None):
        # board: optional string of length 16 with digits 0-4
        self.board = [[0 for _ in range(self.SIZE)] for _ in range(self.SIZE)]
        if board:
            if len(board) != self.SIZE * self.SIZE:
                raise ValueError("Board string must have length 16 for 4x4 mini sudoku")
            for i in range(self.SIZE):
                for j in range(self.SIZE):
                    self.board[i][j] = int(board[i * self.SIZE + j])

    # --- Internal utilities ---

    def _find_empty(self, b):
        for r in range(self.SIZE):
            for c in range(self.SIZE):
                if b[r][c] == 0:
                    return r, c
        return None

    def _is_valid(self, b, r, c, n):
        # Row
        if any(b[r][cc] == n for cc in range(self.SIZE)):
            return False
        # Column
        if any(b[rr][c] == n for rr in range(self.SIZE)):
            return False
        # 2x2 block
        br = (r // self.BOX_ROWS) * self.BOX_ROWS
        bc = (c // self.BOX_COLS) * self.BOX_COLS
        for rr in range(br, br + self.BOX_ROWS):
            for cc in range(bc, bc + self.BOX_COLS):
                if b[rr][cc] == n:
                    return False
        return True

    def _solve_backtracking(self, b):
        empty = self._find_empty(b)
        if not empty:
            return True
        r, c = empty
        nums = list(range(1, self.SIZE + 1))  # 1..4
        random.shuffle(nums)
        for n in nums:
            if self._is_valid(b, r, c, n):
                b[r][c] = n
                if self._solve_backtracking(b):
                    return True
                b[r][c] = 0
        return False

    # --- Generate full board ---

    def generate_full_board(self, rng_seed=None):
        if rng_seed is not None:
            random.seed(rng_seed)
        self.board = [[0 for _ in range(self.SIZE)] for _ in range(self.SIZE)]
        self._solve_backtracking(self.board)

    # --- Solution counting to ensure uniqueness ---

    def _count_solutions(self, limit=None):
        solutions = 0
        b = [row[:] for row in self.board]

        def backtrack():
            nonlocal solutions
            if limit is not None and solutions >= limit:
                return
            empty = self._find_empty(b)
            if not empty:
                solutions += 1
                return
            r, c = empty
            for n in range(1, self.SIZE + 1):
                if self._is_valid(b, r, c, n):
                    b[r][c] = n
                    backtrack()
                    b[r][c] = 0

        backtrack()
        return solutions

    def has_unique_solution(self):
        """True if the puzzle has exactly one solution."""
        return self._count_solutions(limit=2) == 1

    # --- Generate 4x4 puzzle with 7-9 clues and unique solution ---

    def generate_puzzle(self, clue_min=7, clue_max=9, max_tries=200, rng_seed=None):
        """
        Generate a 4x4 puzzle with:
        - unique solution
        - number of clues between clue_min and clue_max (default 7-9)
        Returns (fullBoard, puzzleBoard), both MiniSudokuBoard4x4.
        """
        if clue_min > clue_max:
            raise ValueError("clue_min must be <= clue_max")

        last_full = None
        last_puzzle = None

        for attempt in range(max_tries):
            if rng_seed is not None:
                # vary the seed per attempt to avoid identical puzzles
                random.seed(rng_seed + attempt)

            self.generate_full_board()
            full = copy.deepcopy(self)
            last_full = full

            positions = [(r, c) for r in range(self.SIZE) for c in range(self.SIZE)]
            random.shuffle(positions)
            clues = self.SIZE * self.SIZE  # start fully filled (16 clues)

            for (r, c) in positions:
                if clues <= clue_min:
                    break
                keep = self.board[r][c]
                if keep == 0:
                    continue
                self.board[r][c] = 0

                if self.has_unique_solution():
                    clues -= 1
                    # continue trying to remove more, but not below clue_min
                    if clue_min <= clues <= clue_max:
                        # allow the loop to continue while uniqueness holds
                        last_puzzle = copy.deepcopy(self)
                        continue
                else:
                    # would break uniqueness -> undo
                    self.board[r][c] = keep

            final_clues = sum(
                1 for r in range(self.SIZE) for c in range(self.SIZE) if self.board[r][c] != 0
            )

            if clue_min <= final_clues <= clue_max and self.has_unique_solution():
                return full, copy.deepcopy(self)

        if last_full is None:
            self.generate_full_board()
            last_full = copy.deepcopy(self)
        if last_puzzle is None:
            last_puzzle = copy.deepcopy(self)
        return last_full, last_puzzle
