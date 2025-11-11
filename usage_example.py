from sudokugen import generateSudokus, sudokus2CSV

sudokus = generateSudokus(31, 0)

csv_path = sudokus2CSV(sudokus, "2025-12-01")
print(csv_path)
