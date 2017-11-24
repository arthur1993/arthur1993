# Mastermind Solver

## Description 

This Python project generates and solves multiple boards of the famously knonw Mastermind game.
The code works for any given integer for the size of the board and the number of colors. Note that the default values are the same as the official board game, which is played with a board of 4 pegs and with 7 different colors.


## Example 

The aim of the program is to converge towards the answer in a consistently fast way.

```
[[1, 2, 3, 4], [1, 1]]
[[6, 1, 2, 7], [2, 1, 1]]
[[2, 1, 1, 6], [2, 1, 1]]
[[6, 6, 1, 2], [2, 1, 1]]
[[2, 6, 2, 1], [1, 1, 1]]
[[5, 1, 6, 2], [2, 2, 2, 2]]
Board of size 4 which 7 different colors
Average length : 6.0
Variance length : 0.0
Average time : 0.0184760093689
Variance time : 0.0
--- 0.0187108516693 seconds ---
```

Each line corresponds to a guess and the hints linked to the guess. 

### Initial guess

You could see each number as a different color in the real board game. 
```
[1, 2, 3, 4]
```

### Hints attributed to the guess 

The smaller array after the guess corresponds to the hints attributed to the try.
* 1 is when a pegs has the right color but is not located at the correct location
* 2 is when a pegs is correctly placed
Once the hints only displays 2s', the game is finished
```
[[1, 2, 3, 4], [1, 1]]
```

