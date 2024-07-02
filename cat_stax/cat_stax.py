import numpy as np
import pandas as pd

############################################CAT STATE GENERATION###############################################################################
# let's start this all in two dimensions, which is probably gonna be a pain later, but should make this a lot easier.
# also, going to store these in a dictionary for indicing/iteration purposes


####INITAL STATES#####
# define each cat piece as an n x m matrix.
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

########GENERATE ALL POSSIBLE ORIENTATIONS OF EACH PIECE

# this is buildup to a dumb manual function (_remove_dup_arrays) for removing duplicate arrays in a list of m x n numpy arrays because:
#- numpy arrays aren't hashable so the usual python list uniqueness functions don't work
#- np.unique can remove duplicate arrays in a list, but only if they all have the same dimension.
# so we pad arrays to squares, use np.unique, then unpad them.
# guessing there's a better way of doing this but this works for now.


# given m x n numpy array, pads with 0's till it's an m x m or n x n (which ever is bigger) square array.
def _pad_to_square(array: np.array) -> np.array:
    m, n = array.shape
    # padding with "0" by default -- would be more consistent to use empty strings, but only planning to use
    # this within removing dup arays.
    if m == n:
        None
    elif m < n:
        vpad = n - m
        array = np.pad(array, [(0, vpad), (0, 0)], "constant")
    elif m > n:
        hpad = m - n
        array = np.pad(array, [(0, 0), (0, hpad)], "constant")
    return array


# cleanup function for recovering original array from padded square -- removes final column/row if it's '0'
# lazy and hardcoding this to be only for final column/row which is true for this tileset
# padding will only ever add a max of 1 column or row
def _remove_padding(array: np.array) -> np.array:
    # if last column was padded to all '0:
    if (array[:, -1] == "0").all():
        # remove last column
        array = array[:, 0:-1]
    # if last row padded to '0
    elif (array[-1, :] == "0").all():
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
    is_square = cat.shape[0] == cat.shape[1]
    states = [cat]
    for i in range(3):
        # rotate clockwise and add to state list
        cat = np.rot90(cat)
        states.append(cat)
    return states


def generate_states(cat: np.array) -> np.array:
    """Generates up to 8 rotation states of cats -- rotations, flip then rotations, then removes any duplicates.

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
    return states


#####GENERATE CAT_STATES##########
# dictionary of all shape states
for key, val in cat_states.items():
    cat_states[key] = generate_states(val)

#####PUZZLE GENERATION########
# this could be a submodule or something, but keeping it all here to start.

#also might change this to a class so that a puzzle is a set of cat colours
#paired with the intial array, then solving is a method on it.


# missing spaces will be set to value "X"
# blocked indices is list of (m,n) tuples where an X is placed at the index m,n
def init_puzzle_grid(max_height, max_width, blocked_indices=None):
    grid = np.full((max_height, max_width), "", dtype=str)

    if blocked_indices != None:
        for index in blocked_indices:
            m, n = index
            grid[m, n] = "X"

    return grid


# manually defining puzzle grid variables here.
grid1 = init_puzzle_grid(5, 5)
grid5 = init_puzzle_grid(6, 7)
grid14 = init_puzzle_grid(7, 6)
grid16 = init_puzzle_grid(8, 7, blocked_indices=((2, 1), (2, 5), (5, 1), (5, 5)))


#####PIECE PLACEMENT#############


# Cat Placement is a class that holds the state of a given piece attempting to be placed on a puzzle grid
# Manages iterator of placement indices-- a given orientation is attempted to be placed from left to right in the puzzle
# If no valid spot found, orientation index is increased to move to the next piece state and the process repeats.
# Terminates if indices run out.
# An orientation is attempted to be placed with it's 0,0 index in the i,j'th index in the puzzle grid :
# This class tracks:
# ----colour
# ----index of its orientation (relative to cat_states[colour] list)
# ----orientation representation
# ----orientation height/width
# ----i/j (index to attempt to place at)
# ----idx_gen: generator to move through indices and orientations and update class properties as necessary.
# this is probably dumb and definitely ugly.
# not sure cat_states needs to be passed through all of this but losing track of what scope i need to call it in.
# referred to as o_plc in functions for orientation_placement
class Cat_Placement:
    def __init__(
        self, colour, grid_init, i=-1, j=-1, orientation_idx=0, cat_states=cat_states
    ):
        self.colour = colour
        self.orientation_idx = orientation_idx
        self.orientations = cat_states[colour]
        self.orientation = self.orientations[orientation_idx]
        self.cat_size_m = self.orientation.shape[0]  # rows
        self.cat_size_n = self.orientation.shape[1]  # cols
        self.i = i
        self.j = j
        self.idx_gen = self.make_cat_placement_index_gen(grid_init)

    def make_cat_placement_index_gen(self, grid_init):
        grid_height = grid_init.shape[0]
        grid_width = grid_init.shape[1]
        i_max = grid_height - self.cat_size_m
        j_max = grid_width - self.cat_size_n
        o_max = len(self.orientations) - 1

        def idx_gen_func(self, i_max=i_max, j_max=j_max, o_max=o_max):
            i, j, o = 0, 0, 0  # col, row, orientation index
            while True:
                self.i = i
                self.j = j
                self.orientation_idx = o
                yield i, j, o
                # increment column position
                # print("incrementing column index")
                j += 1
                if j > j_max:
                    # if at max columns, return to 0 indexed column and move one row down
                    # print("exceeded column count")
                    j = 0
                    i += 1
                if i > i_max:
                    # if at max rows, return to 0,0 and try next orientation
                    # print("exceeded rowcount")
                    i = 0
                    j = 0
                    # print("updating orientation index")
                    o += 1
                    # update orientation
                    # print("updating self orientation")
                    self.orientation = self.orientations[o]
                    self.cat_size_m = self.orientation.shape[0]  # rows
                    self.cat_size_n = self.orientation.shape[1]  # cols
                # print("self orientation update succeded.")
                if o > o_max:
                    # this exception gets thrown if next(idx_gen) is called at the maximal index.
                    raise Exception("All indices exhausted.")

        return idx_gen_func(self)


##CHECKS AND NONSENSE HEURISTICS


# just checks if all strings in grid are 1 character or less. if anything more, means we've overlapped somewhere.
def _is_valid_grid(grid: np.array) -> np.array:
    return (np.strings.str_len(grid) <= 1).all()


# custom comparison that immediately throws things out if the index to place things at is has an object in it
# and orientation[0,0] is not empty.
# returns true if both indices non-empty (aka, don't bother placeing it), false otherwise (proceed to placement).
# shaves a lil time off jumping into placement and checking for overlaps in all cases
def _check_idx_blocked(grid_state: np.array, o_plc: Cat_Placement) -> bool:
    o_idx = o_plc.orientation[0, 0]
    i = o_plc.i
    j = o_plc.j
    g_idx = grid_state[i, j]

    # if both grid and orientation indices are non-empty
    if (g_idx != "") and (o_idx != ""):
        return True
    else:
        return False


##PLACEMENT OF PIECES


# given a grid state and a Cat_Placement orientation with current index i,j, places orientation and
# returns placement matrix (subset of array with piece slammed on top of it)
# DOES NOT CHECK IF PLACEMENT IS VALID. just generates placement matrix
def _place_o_at_idx(grid_state: np.array, o_plc: Cat_Placement) -> np.array:
    i = o_plc.i
    j = o_plc.j
    cat_size_m = o_plc.cat_size_m
    cat_size_n = o_plc.cat_size_n
    orientation = o_plc.orientation
    placement = grid_state[i : i + cat_size_m, j : j + cat_size_n] + orientation
    return placement


# given an existing grid state and a Cat_Placement object,
# attempts to place piece by iterating through all remaining placement indices
# and orientation indices.
# terminates and returns updated grid state as soon as valid placement is found.
# otherwise, exception is thrown (from index generator in o_plc: 
# this should probably be more explicit, i don't know shit about exception handling)
def place_orientation(grid_state: np.array, o_plc: Cat_Placement) -> np.array:
    # loop until broken by exception or valid placement
    while True:
        # move to next placement index (or initiate it)
        # print("trying next index")
        next(o_plc.idx_gen)

        # print("trying to place")

        # if idx blocked, don't bother with placement and move to next iteration
        if _check_idx_blocked(grid_state, o_plc):
            continue

        # otherwise, get placement array.
        placement = _place_o_at_idx(grid_state, o_plc)

        # if placement is valid, update grid state and break loop.
        if _is_valid_grid(placement):
            # print("placement is valid")
            # print(placement)
            grid_state[
                o_plc.i : o_plc.i + o_plc.cat_size_m,
                o_plc.j : o_plc.j + o_plc.cat_size_n,
            ] = placement
            break

    return grid_state


# for some grid state and some cat (selected by colour),
# place cat in first valid index/orientation found
# then return updated grid_state and Cat_Placement object
# if o_plc=None, initiates Cat_Placement object from grid state, colour, and cat_states definition.
# otherwise continues from existing Cat_Placement object
# (should make initation explict instead of keying off of None)
def place_cat(
    grid_state: np.array,
    cat_col: str,
    cat_states: dict = cat_states,
    o_plc: Cat_Placement = None,
) -> tuple[np.array, Cat_Placement]:
    if o_plc == None:
        # print("initiating cat object")
        o_plc = Cat_Placement(cat_col, grid_state, cat_states=cat_states)
        # print("cat object initated")
    try:
        # try to place o_plc at it's current o,i,j index
        grid_state = place_orientation(grid_state, o_plc)
    except:
        raise Exception(f"Couldn't place {cat_col} token")

    return grid_state, o_plc


#####SOLVER#######

# mostly brute force crawling through iterable with a minor heuristic or two.

# dict used for sorting cat placement order. heuristic of biggest to smallest by max height/width, then by area
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


# given a list of cat colours and an initial puzzle grid
# attempts place those pieces onto grid in a valid configuration
# if all pieces placed with no exceptions, returns grid and dict of Cat Placement objects
def solve_puzzle(cats: list[str], grid_init: np.array, symmetry=True):
    # sort cats by priority
    # doing this in reverse order so that we can use .pop to get the highest priority piece.
    cats.sort(key=lambda x: -cat_priority[x])

    # initiate dictionary of cat placement objects labled by colours.
    cat_placements = {}
    for col in cats:
        cat_placements[col] = None

    # to_place and placed_cols hold cat colour names, not Cat_Placements
    # use these labels to refer to objects in cat_placements dict
    # initiate to_place with all cats
    to_place = cats.copy()
    placed_cols = []

    grid_state = grid_init.copy()

    # place first cat. special case:
    # if symmetry=True, (m x n) puzzle grid has horizontal/vertical symmetry
    # this means while searching solutions we can hold the inital cat to be placed within
    # the (ceiling(m/2), ceiling(n/2)) top left subgrid.

    # iterate until to_place is empty
    # probably needs a runtime limit
    while len(to_place) > 0:
        # get highest priority piece
        cat_col = to_place[-1]

        # get current placement of piece (None or Cat_Placement object)
        # print(f"Placing {cat_col}")
        placement = cat_placements[cat_col]
        try:
            # try to place piece
            grid_state, placed_cat = place_cat(grid_state, cat_col, o_plc=placement)
        except:
            # if place_cat fails, no possible spot in existing grid for that colour piece
            # all possible placements exhausted without success
            # we then remove previously placed piece (last member of placed_cols)
            # return it to to_place and return to loop (we remove it and move it to its next valid spot)

            # print(f"Placement failed")
            prev_cat = placed_cols.pop()
            to_place.append(prev_cat)

            # revert state of current cat in dict to None
            cat_placements[cat_col] = None

            # REMOVE PREVIOUS PIECE FROM GRID STATE.
            # get label by capitalizing colour's first letter.
            cat_matrix_lbl = prev_cat.upper()[0]
            # print(f"Removing {prev_cat} and placing again")
            grid_state = np.strings.replace(grid_state, cat_matrix_lbl, "")

        else:
            # cat_col successfully placed
            # print(f"{cat_col} placed")
            # remove cat label from to_place and add to placed_cols
            placed_cols.append(to_place.pop())
            # log placement object to cat_placements
            cat_placements[cat_col] = placed_cat

    print(grid_state)
    return grid_state, cat_placements
