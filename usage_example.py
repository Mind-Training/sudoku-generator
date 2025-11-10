from sudokugen import generateSudokus, sudokus2CSV

sudokus = generateSudokus(2, 1)

csv_path = sudokus2CSV(sudokus, "2025-11-10")
print(csv_path)
