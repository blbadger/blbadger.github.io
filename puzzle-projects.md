## Puzzle Projects

### Connect Four winner decider

Let's take a series of moves and determine a winner for the classic turn-based game Connect Four. This was one of my favorite games growing up!  It is played by dropping colored disks into a vertical 7 x 6 board such that each disk played rests on the disks (or base) below.  The winner is the first player to connect four disks of their color in a vertical, horizontal, or diagonal row. 

For the list of moves, a series of moves by each player may be represented by a list of strings as follows:
```python
moves_list = ["F_Yellow", "A_Red", "D_Yellow", "D_Red", ... ]
```
Here each player is denoted by the color of their piece, and each move is denoted by the letter 'A' or 'D' etc. of the row that they put their piece into. 

To determine the winner of any given move list, the strategy is simple: add each move to a matrix representing the game board, and check whether any four pieces are connected at each turn.  To start with, define a function that takes in the move list and returns the winner if there is one and write a docstring that tells us what the funciton will do:

```python
def connect_four_winner(moves_list):
    '''A function that takes a list of connect four moves (each
    move in the format of X_color where X is the row of piece addition
    and color is the color of the piece moved) and returns the winning 
    color, or 'Draw' if there is no winner.
    '''
```
The next step is to initialize a board by using a matrix.  The game board is 7 slots wide by 6 slots high, but it turns out that using a bigger board makes finding connected pieces more simple to program.  Here is a 10 x 9 slot list of lists representing a board.  The reason behind making the board larger will become clear in a moment.

```python
    # initialize the board
    board = [[0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], 
        [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0]]
```
Now let's add each move to the board, one by one.  Abitrarily classifying 'Red' as player 1 and 'Yellow' as player 2, each move in the list of moves is positioned by determining who moved.
```python
    for move in moves_list:
        if move[2:] == 'Red':
            piece = 1
        else: 
            piece = 2
```
To determine where each player's piece lands, the letter corresponding to the row played is counted and the piece is added to the first open slot (open slots are denoted '0') in that row.  Our board is upside down, being filled top to bottom!  This does not matter for determining the winner, however.  The code above is omitted using '...' for clarity.

```python
     for move in moves_list:
     ...
        for i, letter in enumerate('ABCDEFG'):
            if move[0] == letter:
                for k in range(6):
                    if board[k][i] == 0:
                        board[k][i] = piece
                        break
```
Note that pieces are added to the 7x6 slots in the lower left hand corner of the entire 10x9 board.  Now it is clear why this is: after adding each piece to the board as above, it is easy to check if there are four slots of a color occupied in a row, column, or diagonal by iterating accross every position of the 7x6 board as a starting point.  10x9 comes from adding 3 to both width and height of the board, such that there are not index errors as one traverses each position of the real board.  The slots outside the real board remain empty ('0'), and so do not influence whether or not a player wins.

```python
     for move in moves_list:
     ...
     ...
         for i in range(6):
            for j in range(7):
            
                # check for a horizontal four
                count = 0
                for k in range(4):
                    if board[i][j+k] == piece: count += 1
                    else: break
                if count == 4:
                    return 'Red' if piece == 1 else 'Yellow'

                # check for a vertical four
                count = 0
                for n in range(4):
                    if board[i+n][j] == piece: count += 1
                    else: break
                if count == 4:
                    return'Red' if piece == 1 else 'Yellow'

                # check for diagonal four right
                count = 0
                for r in range(4):
                    if board[i+r][j+r] == piece: count += 1
                    else: break
                if count == 4:
                    return 'Red' if piece == 1 else 'Yellow'

                # check for diagonal four left
                count = 0
                for s in range(4):
                    if board[i-s+3][j+s] == piece: count += 1
                    else: break
                if count == 4:
                    return 'Red' if piece == 1 else 'Yellow'

    return 'Draw'
```

The trickiest part here is finding backwards diagonals, ie pieces arranged in a \ pattern.  If we start at [0,0], an index error will be thrown because there is no -1 index in this list of lists! Instead, we offset the starting horizontal value by 3 such that all 4 slots tested are within the matrix size.  Looking at a physical board should make it clear why this is a valid way to test for diagonals, even if not all pieces of the 'real' board are being tested.  If there is no winner after all moves are made, 'Draw' is returned.
```python
      for move in moves_list:
         ...
         ...
                ...
                # check for diagonal four left
                count = 0
                for s in range(4):
                    if board[i-s+3][j+s] == piece: count += 1
                    else: break
                if count == 4:
                    return 'Red' if piece == 1 else 'Yellow'

    return 'Draw'
```

Let's test it out!  The following move list results in a win by Red
```python
moves_list = [
"F_Yellow", "G_Red", "D_Yellow", "C_Red", "A_Yellow", "A_Red", "E_Yellow", "D_Red", "D_Yellow", "F_Red", 
"B_Yellow", "E_Red", "C_Yellow", "D_Red", "F_Yellow", "D_Red", "D_Yellow", "F_Red", "G_Yellow", "C_Red", 
"F_Yellow", "E_Red"]
```
and if we print the board after adding each piece as a matrix by inserting the following into the moves loop:
```python
import pprint
pprint.pprint (board)
```
we see that the final board indeed has a diagonal four for red, in fact it has two!

```python
[[2, 2, 1, 2, 2, 2, 1, 0, 0, 0],
 [1, 0, 2, 1, 1, 1, 2, 0, 0, 0],
 [0, 0, 1, 2, 1, 2, 0, 0, 0, 0],
 [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
 [0, 0, 0, 1, 0, 2, 0, 0, 0, 0],
 [0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
```

### Sudoku solver

