import numpy as np

# reference of valid cat colour names
all_cats = [
    "white",
    "pink",
    "red",
    "yellow",
    "mint",
    "green",
    "arctic",
    "sky",
    "teal",
    "indigo",
    "violet",
    "black",
]


class Puzzle:
    # Puzzle class is
    # - a grid represented by a numpy array
    # - a (colour label) list of cats
    # - the number of layers.
    # blocked slices list may be passed slices or indices.
    # slice is: ((i_min,  i_max), (j_min, j_max)] (non inclusive)
    # indice is: (i, j)
    # careful with negatives
    def __init__(
        self,
        max_height: int,
        max_width: int,
        cats: list[str] = None,
        blocked_slices: list = None,
        layers: int = 1,
    ):
        self.cats = cats
        self.layers = layers
        self.grid = self._init_puzzle_grid(
            max_height, max_width, blocked_slices, layers
        )

    def _init_puzzle_grid(
        self,
        max_height: int,
        max_width: int,
        blocked_slices: list = None,
        layers: int = 1,
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

        # stack grid for number of layers
        grid = np.vstack([grid.reshape(1, max_height, max_width)] * layers)
        return grid


#####PUZZLE GENERATION########

# manually defining all one layer puzzles

Puzzle1 = Puzzle(5, 5, cats=["black", "mint", "violet", "sky", "teal"])
Puzzle2 = Puzzle(
    8,
    6,
    cats=["green", "indigo", "violet", "teal", "sky", "red"],
    blocked_slices=[
        ((0, 2), (0, 1)),
        ((3, 5), (0, 1)),
        ((6, 8), (0, 1)),
        ((0, 2), (4, 6)),
        ((3, 5), (5, 6)),
        ((6, 8), (4, 6)),
    ],
)
Puzzle3 = Puzzle(
    7,
    7,
    cats=["indigo", "violet", "teal", "mint", "white", "black", "sky"],
    blocked_slices=[
        ((0, 2), (0, 2)),
        ((5, 7), (0, 2)),
        ((0, 2), (5, 7)),
        ((5, 7), (5, 7)),
    ],
)

Puzzle4 = Puzzle(
    6,
    6,
    cats=["indigo", "violet", "teal", "mint", "white", "sky", "red"],
    blocked_slices=[((2, 4), (2, 4))],
)
Puzzle5 = Puzzle(
    6, 7, cats=["arctic", "violet", "teal", "yellow", "black", "sky", "red"]
)
Puzzle6 = Puzzle(
    10,
    6,
    cats=["arctic", "violet", "mint", "teal", "sky", "indigo", "red", "pink"],
    blocked_slices=[((1, -1), (2, 4))],
)

Puzzle7 = Puzzle(4, 2, cats=["white", "indigo", "violet", "sky"], layers=2)
# grid13 = init_puzzle_grid(5, 6, [((1, 4), (2, 4))])
Puzzle14 = Puzzle(
    7, 6, cats=["pink", "arctic", "violet", "green", "mint", "sky", "teal"]
)
Puzzle15 = Puzzle(
    7,
    7,
    cats=["arctic", "violet", "teal", "yellow", "white", "mint", "black", "sky"],
    blocked_slices=[(0, 3), (3, 0), (3, -1), (-1, 3)],
)
Puzzle16 = Puzzle(
    8,
    7,
    cats=[
        "black",
        "arctic",
        "indigo",
        "yellow",
        "violet",
        "mint",
        "red",
        "sky",
        "teal",
    ],
    blocked_slices=((2, 1), (2, 5), (5, 1), (5, 5)),
)

Puzzle18 = Puzzle(
    4,
    5,
    cats=["arctic", "indigo", "teal", "yellow", "white", "sky", "red"],
    blocked_slices=[((1, 3), (2, 3))],
    layers=2,
)
Puzzle23 = Puzzle(
    3,
    6,
    cats=[
        "arctic",
        "violet",
        "teal",
        "yellow",
        "mint",
        "pink",
        "black",
        "sky",
        "red",
    ],
    layers=3,
)

Puzzle24 = Puzzle(
    5,
    4,
    cats=[
        "arctic",
        "indigo",
        "teal",
        "mint",
        "red",
        "yellow",
        "white",
        "sky",
        "green",
        "black",
    ],
    layers=3,
)

Puzzle40 = Puzzle(5, 5, cats=all_cats, blocked_slices=[(2, 2)], layers=3)

Puzzle41 = Puzzle(4, 5, cats=all_cats, blocked_slices=[((0, 1), (0, 2))], layers=4)

Puzzle43 = Puzzle(
    8, 5, cats=all_cats, blocked_slices=[(0, 0), (0, 1), (0, 3), (0, 4)], layers=2
)

Puzzle48 = Puzzle(
    4, 5, cats=all_cats, blocked_slices=[(0, 4), (3, 4)], layers=4
)