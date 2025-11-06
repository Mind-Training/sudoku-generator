#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sudoku Board Solver and Generator Console Menu
"""

import os
from sudokuBoard import SudokuBoard


def printHeader():
    print("SUDOKU BOARD SOLVER AND GENERATOR")
    print("Modified by Daniel\n")
    print("| ----------------------------------------------- |")
    print("| Levels: easy | medium | hard | expert           |")
    print("| ----------------------------------------------- |\n")


def clearConsole():
    command = 'clear'
    if os.name in ('nt', 'dos'):
        command = 'cls'
    os.system(command)


def showMenu():
    printHeader()
    print("Which action do you like to perform?")
    print("1: Solve a Sudoku")
    print("2: Generate new board")
    print("3: Read instructions")
    print("4: Generate by difficulty")

    valid_input = False
    usr_input = 0
    while not valid_input:
        usr_input = input("\nSelect an option:  ")
        try:
            usr_input = int(usr_input)
            if usr_input < 1 or usr_input > 4:
                raise Exception
            valid_input = True
        except:
            print("Error, invalid input. Try again.")
            continue
    clearConsole()
    return usr_input


def solveSudokuCase():
    print("SOLVE A SUDOKU\n\n")
    print("Write the 81 cell values below (0 for empty cells)\n")

    valid_input = False
    while not valid_input:
        boardAsString = input(": ")
        charCounter = 0
        for c in boardAsString:
            try:
                int(c)
                charCounter += 1
            except:
                print("Error: input must contain only digits.\n")
                break
        if charCounter != 81:
            print("Error: must be 81 characters long.\n")
            continue
        valid_input = True

    b = SudokuBoard(board=boardAsString)
    print("\n\nORIGINAL BOARD")
    b.printBoard()
    print("\nSOLVED BOARD")
    b.solve()
    b.printBoard()
    input("\nPress Enter to return to the main menu...")


def generateNewBoardCase():
    print("GENERATE NEW BOARD\n\n")
    valid_input = False
    emptySpaces = 0
    while not valid_input:
        emptySpaces = input("How many empty spaces? (0-50):  ")
        try:
            emptySpaces = int(emptySpaces)
            if emptySpaces < 0 or emptySpaces > 50:
                raise Exception
            valid_input = True
        except:
            print("Error, invalid input. Type a number (0-50).")
            continue

    b = SudokuBoard()
    if emptySpaces > 0:
        filled, unfilled = b.generateGameBoard(emptySpaces=emptySpaces)
        print("\nNON-FILLED BOARD")
        unfilled.printBoard()
        print("\nFILLED BOARD")
        filled.printBoard()
    else:
        filled, _ = b.generateGameBoard()
        print("\nFILLED BOARD")
        filled.printBoard()
    input("\nPress Enter to return to the main menu...")


def readInstructionsCase():
    printHeader()
    print("INSTRUCTIONS\n")
    print("1 → Solve a Sudoku (type 81 digits, 0 for empty)")
    print("2 → Generate random Sudoku (choose empty cells)")
    print("3 → Read instructions")
    print("4 → Generate by difficulty (new!)\n")
    input("\nPress Enter to return to the main menu...")


def generateByDifficultyCase():
    print("GENERATE BY DIFFICULTY\n")
    print("Available levels: easy | medium | hard | expert\n")
    level = None
    while level not in ("easy", "medium", "hard", "expert"):
        level = input("Choose difficulty: ").strip().lower()
        if level not in ("easy", "medium", "hard", "expert"):
            print("Invalid level. Try again.\n")

    b = SudokuBoard()
    filled, playable = b.generateGameBoardByDifficulty(level=level)

    print("\nPLAYABLE BOARD")
    playable.printBoard()
    label, score = playable.rateDifficulty()
    print(f"\nDifficulty (measured): {label} | Score: {score}")
    print("\nSOLUTION BOARD")
    filled.printBoard()

    input("\nPress Enter to return to the main menu...")


if __name__ == "__main__":
    while True:
        clearConsole()
        usr_input = showMenu()
        if usr_input == 1:
            solveSudokuCase()
        elif usr_input == 2:
            generateNewBoardCase()
        elif usr_input == 3:
            readInstructionsCase()
        elif usr_input == 4:
            generateByDifficultyCase()
