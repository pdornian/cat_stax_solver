import numpy as np

#####PUZZLE GENERATION########
# also might change this to a class so that a puzzle is a set of cat colours
# paired with the intial array, then solving is a method on it.


# missing spaces will be set to value "X"
# blocked indices is list of (m,n) tuples where an X is placed at the index m,n
def init_puzzle_grid(max_height, max_width, blocked_indices=None):
    grid = np.full((max_height, max_width), "", dtype=str)

    if blocked_indices != None:
        for index in blocked_indices:
            m, n = index
            grid[m, n] = "X"

    return grid


test = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    "damnit black",
    "why are you like this",
    "at what point do you split this list",
]

# manually defining puzzle grid variables here.
grid1 = init_puzzle_grid(5, 5)
grid2 = init_puzzle_grid(
    8, 6, blocked_indices=((0, 0), (1, 0), (3, 0), (4, 0), (6, 0), (7, 0))
)
grid5 = init_puzzle_grid(6, 7)
grid14 = init_puzzle_grid(7, 6)
grid16 = init_puzzle_grid(8, 7, blocked_indices=((2, 1), (2, 5), (5, 1), (5, 5)))
