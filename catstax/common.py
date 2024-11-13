import matplotlib.pyplot as plt
import numpy as np

# import puzzle class from  puzzles. maybe it should live here.
# dunno. kept it with the hardcoded puzzles.
from catstax.puzzles import Puzzle

# CAT STATE GENERATION
# define each cat piece as an n x m matrix.
# stored in "cat_states" dictionary for indicing/iteration purposes
cat_states = dict()

white = np.full((2, 3), "W", dtype=str)
white[0][1:] = ""
cat_states["white"] = white

pink = np.full((3, 3), "P", dtype=str)
pink[0][1:] = ""
pink[2][1] = ""
cat_states["pink"] = pink

red = np.full((3, 4), "R", dtype=str)
red[0][1:] = ""
red[2][1:3] = ""
cat_states["red"] = red

yellow = np.full((3, 3), "Y", dtype=str)
yellow[0][1:] = ""
cat_states["yellow"] = yellow

mint = np.full((3, 3), "M", dtype=str)
mint[0][0] = ""
mint[0][2] = ""
mint[1][0] = ""
cat_states["mint"] = mint

green = np.full((4, 4), "G", dtype=str)
green[0][0:2] = ""
green[1][1:3] = ""
green[3][1:3] = ""
cat_states["green"] = green

arctic = np.full((3, 4), "A", dtype=str)
arctic[0][0:3] = ""
cat_states["arctic"] = arctic

sky = np.full((1, 2), "S", dtype=str)
cat_states["sky"] = sky

teal = np.full((2, 2), "T", dtype=str)
teal[0][1] = ""
cat_states["teal"] = teal

indigo = np.full((2, 2), "I", dtype=str)
cat_states["indigo"] = indigo

violet = np.full((3, 2), "V", dtype=str)
cat_states["violet"] = violet

black = np.full((3, 3), "B", dtype=str)
black[0][1] = ""
cat_states["black"] = black

# GENERATE ALL POSSIBLE ORIENTATIONS OF EACH PIECE

# this is buildup to a dumb manual function (_remove_dup_arrays)
# for removing duplicate arrays in a list of m x n numpy arrays because:
# - numpy arrays unhashable so the usual python list uniqueness functions don't work.
# - np.unique can remove duplicate arrays in a list,
#   but only if they all have the same dimension.
# so we pad arrays to squares, use np.unique, then unpad them.
# guessing there's a better way of doing this but this works for now.


def _pad_to_square(array: np.array) -> np.array:
    # given m x n numpy array, pads with 0's so it is an m x m or n x n
    # (which ever is bigger) square array.
    # pads w/ "0" by default -- would be more consistent to use empty string
    # but doesn't really matter for private func.
    m, n = array.shape

    if m < n:
        vpad = n - m
        array = np.pad(array, [(0, vpad), (0, 0)], "constant")
    elif m > n:
        hpad = m - n
        array = np.pad(array, [(0, 0), (0, hpad)], "constant")
    # if m == n, nothing happens and square array returned.
    return array


def _remove_padding(array: np.array) -> np.array:
    # cleanup function for recovering original array from padded square
    # removes final column/row if it's '0'
    # only checking -1 indice bc _pad_to_square will
    # add a max of 1 row or column on default tileset, but this isn't forced.

    # if last column is padding
    if (array[:, -1] == "0").all():
        # remove it from array
        array = array[:, 0:-1]
    # if last row is padding
    elif (array[-1, :] == "0").all():
        # remove it from array
        array = array[0:-1, :]
    return array


# given a list of np arrays, returns list containly only the unique arrays.
# used to remove redundant orientations after generating via flips/rotations.
def _remove_dup_arrays(array_list: list[np.array]) -> list[np.array]:
    # pad all arrays to squares
    padded_arrays = [_pad_to_square(array) for array in array_list]
    # get unique arrays
    unique_padded_arrays = list(np.unique([array for array in padded_arrays], axis=0))
    # unpad
    unique_arrays = [_remove_padding(array) for array in unique_padded_arrays]

    return unique_arrays


def _generate_rotations(cat: np.array) -> np.array:
    """Generates four rotation states of cat piece from initial state.

    Args:
        cat (np.array): Numpy array representing cat shape

    Returns:
        list[np.array]: Rotation states of of shape
    """
    states = [cat]
    for _ in range(3):
        # rotate clockwise and add to state list
        cat = np.rot90(cat)
        states.append(cat)
    return states


# using this after generating all rotations of pieces in 2d axis
# to move to 3d array notation.
# 3 possible unique orientations depending on if embedded in x, y or z.
# lil janky, but means i don't have to adjust rot functions.


def _3d_bootstrap(cat: np.array) -> list[np.array]:
    """Given a cat, returns a list of three 3d arrays
        corresponding to it being embedded in each of the x, y, or z dimensions.

    Args:
        cat (np.array): n x m array

    Returns:
        list[np.array]: [1 x n x m array, n x 1 x m array, n x m x 1 array]
    """
    r, c = cat.shape
    # single z-axis orientation
    # single row axis orientation,
    # single column axis orientation
    return [cat.reshape(1, r, c), cat.reshape(r, 1, c), cat.reshape(r, c, 1)]


def generate_states(cat: np.array) -> np.array:
    """Generates up to 24 orientation states of cats
    - rotations,
    - flip followed by rotations,
    - then removes any duplicates,
    - then embeds each in x, y, and z dimension (as 3d array representation)
    - then returns orientation list.

    Args:
        cat (np.array): Numpy array representing cat shape.

    Returns:
        np.array: All orientations of shape induced by rotation and flip
    """

    # get rotation states
    states = _generate_rotations(cat)
    # get flip states
    f_states = _generate_rotations(np.fliplr(cat))

    states.extend(f_states)
    states = _remove_dup_arrays(states)

    xyz_states = []
    for state in states:
        xyz_states.extend(_3d_bootstrap(state))
    return xyz_states


# GENERATE CAT_STATES
# cat_states dictionary of all shape statesw
# used as default argument throughout
for key, val in cat_states.items():
    cat_states[key] = generate_states(val)

# PLOTTING
# used to display outputs on matplotlib voxel grid.
# for colours in plotting
# may not correspond to colour labels
colour_map = {
    "V": "magenta",
    "T": "teal",
    "M": "palegreen",
    "B": "black",
    "R": "red",
    "I": "indigo",
    "S": "cyan",
    "G": "limegreen",
    "A": "paleturquoise",
    "Y": "yellow",
    "P": "hotpink",
    "W": "white",
}


def plot_puzzle(solved_puzzle, colour_map=colour_map, az=0):
    x = solved_puzzle.shape[0]
    y = solved_puzzle.shape[1]
    z = solved_puzzle.shape[2]

    # dumb reshaping/reindexing.
    # arrays are easiest to view with height in first dimension,
    # but this doesn't correspond to height when you slam em into this plotting function
    # so we rotate this shit

    solved_puzzle = np.rot90(solved_puzzle, axes=(2, 0))
    x = solved_puzzle.shape[0]
    y = solved_puzzle.shape[1]
    z = solved_puzzle.shape[2]

    # remove X characters (non-placeable spaces)
    solved_puzzle = np.strings.replace(solved_puzzle, "X", "")

    # pad all puzzles to four voxels high for consistent display
    solved_puzzle = np.pad(
        solved_puzzle,
        [(0, 0), (0, 0), (0, 4 - z)],
        "constant",
        constant_values="",
    )

    # initate puzzle shape array with true val for all non-empty indices
    puzzle_space = solved_puzzle != ""

    colours = np.empty(puzzle_space.shape, dtype=object)
    for label, colour in colour_map.items():
        colour_mask = solved_puzzle == label
        colours[colour_mask] = colour

    ax = plt.figure().add_subplot(projection="3d")
    # we're really just plotting a m x n x 4 grid of voxels for all solutions
    # it's the facecolours array that dictates what it's going to look like.
    ax.voxels(puzzle_space, facecolors=colours, edgecolor="gray", shade=False)

    if az != 0:
        ax.view_init(azim=az)

    ax.axes.set_xlim3d(left=0, right=x)
    ax.axes.set_ylim3d(bottom=0, top=y)
    ax.axes.set_zlim3d(bottom=0, top=4)
    ax.axis("equal")
    ax.axis("off")
    plt.show()

    return ax


# PIECE PLACEMENT


# this is probably dumb and definitely ugly.
# not sure cat_states needs to be passed through all of this
# but lost track of ideal scope for it.
# Cat_Placement objects often referred to as "o_plc" in functions
# for "orientation_placement". yes, this is inconsistent.
class Cat_Placement:
    """Cat Placement is a class that holds the state of a given piece
    during the solving routine. It stores the piece, its current orientation,
    an index representing a possible spot in the puzzle to place it,
    and generator used to iterate through possble placement indices.

    This class tracks:
    ----piece colour.
    ----index of its orientation (relative to cat_states[colour] list).
    ----orientation array representation.
    ----orientation dimensions.
    ----puzzle grid dimensions.
    ----i/j (placement index on puzzle grid).
    ----idx_gen: generator to move through placement/orientation indices

    IT DOES NOT record whether that placement was a success or not.
    (Maybe it should)

    idx_gen is the generator used to handle placement indices.
    Initiated at i,j,k,o = 0,0,0
    Calling 'next' on the generator moves index (i,j,k,o) left to right along each row.
    (i, j, k, o) -> (i, j, k+1, o)
    If all cols in row are exausted, moves down to the next row and leftmost col.
    (i, j, k, o) -> (i, j + 1, 0, o)
    If all rows exhausted, resets row/col index and moves to next depth idx
    (i, j, k, o) -> (i + 1, 0, 0, o)
    If all cols, rows, depths exhausted:
    - move to next orientation index
    - reset i, j, k to 0
    - updates orientation dependent properties
    (i, j, k, o) -> (0, 0, k, o+1)

    Exception generated if next(idx_gen) is called when all of
    i, j, o are at their max values.
    """

    # i forget how that exception is generated
    # or if its supposed to be explicitly defined at the end of the else
    # block. yolo

    # not sure why ijk is initiated at -1, its remapped to 0 in idx_gen_func
    # but i don't feel like changing and troubleshooting it rn
    def __init__(
        self,
        colour,
        grid_init,
        # pretty sure ijk here are redundant
        i=-1,
        j=-1,
        k=-1,
        orientation_idx=-1,
        cat_states=cat_states,
    ):
        self.colour = colour
        self.orientation_idx = orientation_idx
        self.orientations = cat_states[colour]  # all possible orientations
        self.orientation = None
        self.o_dims = None
        self.g_dims = grid_init.shape  # puzzle grid dimensions
        self.i = i  # grid depth placement idx
        self.j = j  # grid row placement idx
        self.k = k  # grid col placement idx
        self.idx_gen = self.make_cat_placement_index_gen()

    def get_next_valid_o(self):
        # gets next valid orientation and updates properties
        # "valid" = piece orientation that fits within puzzle grid
        o = self.orientation_idx
        while True:
            # increment index
            o += 1

            # if index out of orientations range, raise exception
            if o >= len(self.orientations):
                raise StopIteration

            # otherwise get candidate orientation
            candidate = self.orientations[o]

            # if fits within grid dimensions, break.
            # otherwise next index and repeat.
            if np.all(np.array(candidate.shape) <= np.array(self.g_dims)):
                self.orientation_idx = o
                self.orientation = self.orientations[o]
                self.o_dims = self.orientation.shape

                # print(self.orientation)
                break

    def make_cat_placement_index_gen(self):
        def idx_gen_func(self):
            # initiate with first valid orientation
            self.get_next_valid_o()
            # initiate col, row, orientation indices
            i, j, k, o = 0, 0, 0, -1

            # initate max indices
            o_max = len(self.orientations) - 1
            # could make subfunction for this
            i_max = self.g_dims[0] - self.o_dims[0]
            j_max = self.g_dims[1] - self.o_dims[1]
            k_max = self.g_dims[2] - self.o_dims[2]

            while True:
                self.i = i
                self.j = j
                self.k = k

                # i don't know why i'm yielding this stuff anymore
                # its all stored properties in the object

                yield i, j, k, o
                # increment column index check cases
                # print("incrementing column index")

                if k < k_max:
                    # increment row col idx
                    k += 1

                elif j < j_max:
                    # if exceeded max col idx but not max row idx
                    # reset col idx and increment row idx
                    # e.g: try to place in next row
                    # print("exceeded column count")
                    k = 0
                    j += 1

                elif i < i_max:
                    # if also exceeded max row idx but not max depth idx
                    #  reset row/col index to 0,0 and increment depth index
                    # print("exceeded max row idx")
                    i += 1
                    j = 0
                    k = 0
                elif o < o_max:
                    # exceeding all col/row/depth indexes
                    # aka we're shit out of luck, onto the next orientation
                    # reset i,j,k  to 0,0 and try next orientation
                    # print("exceeded max depth idx")
                    self.get_next_valid_o()

                    i_max = self.g_dims[0] - self.o_dims[0]
                    j_max = self.g_dims[1] - self.o_dims[1]
                    k_max = self.g_dims[2] - self.o_dims[2]

                    i = 0
                    j = 0
                    k = 0
                else:
                    # we attempted to move to an o_idx that doesn't exist.
                    return "All orientations exhausted"

        return idx_gen_func(self)


# CHECKS AND NONSENSE HEURISTICS
def _is_valid_grid(grid: np.array) -> np.array:
    # checks if all strings in array are 1 character or less.
    # if anything more, means we've overlapped somewhere.
    return (np.strings.str_len(grid) <= 1).all()


# call before placement attempt to check if top-left piece square of orientation
# and placement index of grid are non-empty.
# only going into full placement routine check if this is false
# can shave a few secs off search
def _check_idx_blocked(grid_state: np.array, o_plc: Cat_Placement) -> bool:
    # given grid state and Cat Placement object
    # checks if both orientation[0,0] and placement index i,j are non-empty.
    o_idx = o_plc.orientation[0, 0, 0]
    i = o_plc.i
    j = o_plc.j
    k = o_plc.k
    g_idx = grid_state[i, j, k]

    # if both grid and orientation indices are non-empty
    if (g_idx != "") and (o_idx != ""):
        return True
    else:
        return False


def _check_invalid_dims(grid_state: np.array, o_plc: Cat_Placement) -> bool:
    # i think this would be unnecessary if k_max was being updated correctly
    # but writing this as a bandaid rn

    # checks that piece size is in bounds of grid
    # probably a better place to put this logic
    i = o_plc.i
    j = o_plc.j
    k = o_plc.k
    cat_size_i, cat_size_j, cat_size_k = o_plc.o_dims

    # if there's room for piece orientation, this will have same shape as piece
    grid_target = grid_state[i : i + cat_size_i, j : j + cat_size_j, k : k + cat_size_k]

    if grid_target.shape != o_plc.orientation.shape:
        # return true, skip iteration via continue.
        return True
    else:
        return False


# PLACEMENT OF PIECES


def _place_o_at_idx(grid_state: np.array, o_plc: Cat_Placement) -> np.array:
    # given a grid state and a Cat_Placement orientation with current index i,j,
    # returns placement matrix of attempting to place orientation at i,j
    # (subset of array with piece slammed on top of it)
    # DOES NOT CHECK IF PLACEMENT IS VALID. just generates placement matrix
    i = o_plc.i
    j = o_plc.j
    k = o_plc.k
    # print(f"attempting to place at {(i,j,k)}")
    cat_size_i, cat_size_j, cat_size_k = o_plc.o_dims
    placement = (
        grid_state[i : i + cat_size_i, j : j + cat_size_j, k : k + cat_size_k]
        + o_plc.orientation
    )

    return placement


def place_orientation(grid_state: np.array, o_plc: Cat_Placement) -> np.array:
    # given an existing grid state and a Cat_Placement object,
    # attempt to place piece by iterating through
    # (up to) all remaining placement and orientation indices.
    # returns first valid grid state found.

    # if no valid placements found, exception raise (from index generator in o_plc:
    # this should probably be more explicit, i don't know shit about exception handling)

    # loop until broken by exception or valid piece placement
    while True:
        # move to next placement index (or initiate it)
        next(o_plc.idx_gen)

        # if idx blocked, don't bother with placement and move to next iteration
        if _check_idx_blocked(grid_state, o_plc):
            continue

        # otherwise, get placement array.
        placement = _place_o_at_idx(grid_state, o_plc)

        # if placement is valid, update grid state and break loop.
        if _is_valid_grid(placement):
            grid_state[
                o_plc.i : o_plc.i + o_plc.o_dims[0],
                o_plc.j : o_plc.j + o_plc.o_dims[1],
                o_plc.k : o_plc.k + o_plc.o_dims[2],
            ] = placement
            break

    return grid_state


def place_cat(
    grid_state: np.array,
    cat_col: str,
    cat_states: dict = cat_states,
    o_plc: Cat_Placement = None,
) -> tuple[np.array, Cat_Placement]:

    # for some grid state and some cat (selected by colour),
    # place cat in first valid index/orientation found
    # then return updated grid_state and Cat_Placement object

    # if o_plc=None, initiates Cat_Placement object from:
    # ----grid state,
    # ----colour,
    # ----and cat_states.
    # otherwise, uses existing Cat_Placement object
    # (should make initation explict/a wrapper instead of keying off of None)
    if o_plc is None:
        # print("initiating cat object")
        o_plc = Cat_Placement(cat_col, grid_state, cat_states=cat_states)
        # print("cat object initated")

    # try to place o_plc at it's next o,i,j index
    grid_state = place_orientation(grid_state, o_plc)

    return grid_state, o_plc


#####SOLVER#######

# mostly brute force crawling through iterable with a minor heuristic or two.

# dict used for sorting cat placement order by colour label.
# heuristic of biggest to smallest by max height/width, then by area
cat_priority = {
    "green": 1,
    "arctic": 2,
    "red": 3,
    "black": 4,
    "yellow": 5,
    "mint": 6,
    "pink": 7,
    "violet": 8,
    "white": 9,
    "indigo": 10,
    "teal": 11,
    "sky": 12,
}


# def solve_puzzle(cats: list[str], grid_init: np.array, symmetry=True, plot=True):
def solve_puzzle(puz: Puzzle, symmetry=True, plot=True):
    # needs a docstring

    # given a puzzle object, retrieves cat colours and an initial puzzle grid
    # attempts place those pieces onto grid in a valid configuration
    cats = puz.cats
    grid_init = puz.grid

    # sort cats by priority
    # doing this in reverse order so that we can use .pop to
    # get the highest priority piece.
    cats.sort(key=lambda x: -cat_priority[x])

    # initiate dictionary of cat placement objects labled by colours.
    cat_placements = {}
    for col in cats:
        cat_placements[col] = None

    # to_place and placed_cols hold cat colour names, not Cat_Placement objects
    # use colour labels to refer to objects in cat_placements dict
    # initiate to_place with all cat piece colours
    to_place = cats.copy()
    placed_cols = []

    grid_state = grid_init.copy()

    # place first cat. special case not actually coded (for later):
    # if symmetry=True, (m x n) puzzle grid has horizontal/vertical symmetry
    # this means while searching solutions we can hold the inital cat to
    # be placed within the (ceiling(m/2), ceiling(n/2)) top left subgrid.

    # counter for number of iterations
    placement_iterations = 0

    while len(to_place) > 0:

        placement_iterations += 1

        # get highest priority piece
        cat_col = to_place[-1]

        # get current placement of piece (None or Cat_Placement object)
        # print(f"Placing {cat_col}")
        placement = cat_placements[cat_col]
        try:
            # try to place piece
            grid_state, placed_cat = place_cat(grid_state, cat_col, o_plc=placement)
            # really this is catching StopIterator errors
            # but those get turned into runtime errors in generator
            # not sure what best practice is
        except RuntimeError:
            # the gist:
            # all placement/orientation indices exhausted with no success
            # on current grid state and current piece
            # then, remove last successfully placed piece and try again
            # (since place_cat will increment idx_gen, we're guaranteed that
            # it will be placed if possible in new location)
            # ----------------

            # print(f"Placement failed")

            # revert state of current cat in dict to None
            # (any further placements of it re-initiate cat_placement object
            # and start placement routine from beginning again.
            # lots to optimize there.)
            cat_placements[cat_col] = None

            # remove previously placed piece (last member of placed_cols),
            prev_cat = placed_cols.pop()
            # return it back to to_place
            to_place.append(prev_cat)

            # remove piece from grid state
            # get label by capitalizing colour's first letter
            # (should probably be label dict to be more explict)
            # (this should be a function).
            cat_matrix_lbl = prev_cat.upper()[0]
            # print(f"Removing {prev_cat} and placing again")
            grid_state = np.strings.replace(grid_state, cat_matrix_lbl, "")

        else:
            # cat_col was successfully placed
            # print(f"{cat_col} placed")

            # remove cat label from to_place and add to placed_cols
            placed_cols.append(to_place.pop())
            # log placement object to cat_placements
            cat_placements[cat_col] = placed_cat
    print(f"total placement iterations: {placement_iterations}")
    print(grid_state)

    if plot:
        plot_puzzle(grid_state)

    return grid_state, cat_placements
