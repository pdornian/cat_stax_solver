import numpy as np

#####PUZZLE GENERATION########
# also might change this to a class so that a puzzle is a set of cat colours
# paired with the intial array, then solving is a method on it.


# missing spaces will be set to value "X"
# blocked indices is list of (m,n) tuples where an X is placed at the index m,n


# slice is: ((i_min,  i_max), (j_min, j_max)]
# careful with negatives
# gonna needa change this for the 3d case.
def init_puzzle_grid(
    max_height: int, max_width: int, blocked_slices: list = None
) -> np.array:

    if blocked_slices is None:
        blocked_slices = []
    grid = np.full((max_height, max_width), "", dtype=str)

    # no checking in this for overlapping slices, which would be nice.
    # lazy typing: if slice is tuple[int], read it as index coordinates
    # if slice is tuple[tuple], read as slice (expects min and max in each tuple)
    # could use some other cases

    for slice in blocked_slices:
        # if first element is integer, assume this slice is an index
        if type(slice[0]) is int:
            grid[slice] = "X"
        else:  # could put type check here but meh (assume it's a tuple)
            i_min, i_max = slice[0]
            j_min, j_max = slice[1]
            grid[i_min:i_max, j_min:j_max] = "X"
    return grid


# manually defining all one layer puzzles
grid1 = init_puzzle_grid(5, 5)
grid2 = init_puzzle_grid(
    8,
    6,
    blocked_slices=[
        ((0, 2), (0, 1)),
        ((3, 5), (0, 1)),
        ((6, 8), (0, 1)),
        ((0, 2), (4, 6)),
        ((3, 5), (5, 6)),
        ((6, 8), (4, 6)),
    ],
)
grid3 = init_puzzle_grid(
    7,
    7,
    blocked_slices=[
        ((0, 2), (0, 2)),
        ((5, 7), (0, 2)),
        ((0, 2), (5, 7)),
        ((5, 7), (5, 7)),
    ],
)

grid4 = init_puzzle_grid(6, 6, blocked_slices=[((2, 4), (2, 4))])
grid5 = init_puzzle_grid(6, 7)
grid6 = init_puzzle_grid(10, 6, [((1, -1), (2, 4))])
grid13 = init_puzzle_grid(5, 6, [((1, 4), (2, 4))])
grid14 = init_puzzle_grid(7, 6)
grid15 = init_puzzle_grid(7, 7, blocked_slices=[(0, 3), (3, 0), (3, -1), (-1, 3)])
grid16 = init_puzzle_grid(8, 7, blocked_slices=((2, 1), (2, 5), (5, 1), (5, 5)))
