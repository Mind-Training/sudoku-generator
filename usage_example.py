from sudokugen import generateSudokus, sudokus2CSV

sudokus = generateSudokus(5, 0, size=6)

csv_path = sudokus2CSV(sudokus, "2025-11-04", filename="mini_sudokus.csv")
print(csv_path)
