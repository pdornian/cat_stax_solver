import numpy as np

# everything is two-dimensional right now
# 3D tbd

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
    # given m x n numpy array, pads with 0's so it iss an m x m or n x n
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


def generate_states(cat: np.array) -> np.array:
    """Generates up to 8 rotation states of cats
    - rotations,
    - flip followed by rotations,
    - then removes any duplicates and returns orientation list.

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


# GENERATE CAT_STATES
# cat_states dictionary of all shape statesw
# used as default argument throughout
for key, val in cat_states.items():
    cat_states[key] = generate_states(val)


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
    Initiated at i,j,o = 0,0,0
    Calling 'next' on the generator moves index (i,j) left to right along each row.
    (i, j, o) -> (i, j+1, o)
    If a row is exausted, moves down to the next row and leftmost col.
    (i, j, o) -> (i+1, 0, o)
    If all rows and columns exhausted:
    - move to next orientation index
    - reset i, j to top left (0,0) index
    - updates orientation dependent properties
    (i, j, o) -> (0, 0, o+1)

    Exception generated if next(idx_gen) is called when all of
    i, j, o are at their max values.
    """  # noqa: W293

    def __init__(
        self, colour, grid_init, i=-1, j=-1, orientation_idx=0, cat_states=cat_states
    ):
        self.colour = colour
        self.orientation_idx = orientation_idx
        self.orientations = cat_states[colour]  # all possible orientations
        self.orientation = self.orientations[orientation_idx]
        self.o_dims = self.orientation.shape  # orienation dimensions
        self.g_dims = grid_init.shape  # puzzle grid dimensions
        # self.cat_size_m = self.orientation.shape[0]  #num cat rows
        # self.cat_size_n = self.orientation.shape[1]  #num cat cols
        # self.grid_m= grid_init.shape[0] #num grid rows
        # self.grid_n= grid_init.shape[1] #num grid cols
        self.i = i  # grid row placement idx
        self.j = j  # grid col placement idx
        self.idx_gen = self.make_cat_placement_index_gen()

    # def make_cat_placement_index_gen(self, grid_init):
    def make_cat_placement_index_gen(self):
        # grid_height = grid_init.shape[0]
        # grid_width = grid_init.shape[1]
        # i_max = grid_height - self.cat_size_m
        # j_max = grid_width - self.cat_size_n
        # o_max = len(self.orientations) - 1

        # def idx_gen_func(self, i_max=i_max, j_max=j_max, o_max=o_max):
        def idx_gen_func(self):
            # initiate col, row, orientation indices
            i, j, o = 0, 0, 0

            # initate max indices
            o_max = len(self.orientations) - 1
            i_max = self.g_dims[0] - self.o_dims[0]
            j_max = self.g_dims[1] - self.o_dims[1]

            while True:
                self.i = i
                self.j = j
                self.orientation_idx = o
                yield i, j, o
                # increment column index check cases
                # print("incrementing column index")

                if j < j_max:
                    j += 1

                elif i < i_max:
                    # if at max columns, move one row down and reset col idx
                    # print("exceeded column count")
                    j = 0
                    i += 1
                elif o < o_max:
                    #i and j at max but o not yet maxed
                    # reset i,j to 0,0 and try next orientation
                    # print("exceeded rowcount")
                    i = 0
                    j = 0
                    # print("updating orientation index")
                    o += 1
                    # update orientation and idx maxes.
                    # print("updating self orientation")
                    self.orientation = self.orientations[o]
                    self.o_dims = self.orientation.shape
                    i_max = self.g_dims[0] - self.o_dims[0]
                    j_max = self.g_dims[1] - self.o_dims[1]
                    # print("self orientation update succeded.")
                else:
                    # we attempted to move to an o_idx that doesn't exist.
                    return

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
    o_idx = o_plc.orientation[0, 0]
    i = o_plc.i
    j = o_plc.j
    g_idx = grid_state[i, j]

    # if both grid and orientation indices are non-empty
    if (g_idx != "") and (o_idx != ""):
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
    cat_size_m, cat_size_n = o_plc.o_dims
    placement = grid_state[i : i + cat_size_m, j : j + cat_size_n] + o_plc.orientation
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
                o_plc.i : o_plc.i + o_plc.o_dims[0],
                o_plc.j : o_plc.j + o_plc.o_dims[1],
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
    try:
        # try to place o_plc at it's next o,i,j index
        grid_state = place_orientation(grid_state, o_plc)
        #StopIteration exception thrown if we're out of o,i,j indices
        #might only need handling within solve_puzzle
    except StopIteration:
        raise

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


def solve_puzzle(cats: list[str], grid_init: np.array, symmetry=True):
    #needs a docstring

    # given a list of cat colours and an initial puzzle grid
    # attempts place those pieces onto grid in a valid configuration
    # if all pieces placed with no exceptions, returns grid and
    # dict of Cat Placement objects

    # sort cats by priority
    # doing this in reverse order so that we can use .pop to
    # get the highest priority piece.
    cats.sort(key=lambda x: -cat_priority[x])

    # initiate dictionary of cat placement objects labled by colours.
    cat_placements = {}
    for col in cats:
        cat_placements[col] = None

    # to_place and placed_cols hold cat colour names, not Cat_Placements
    # use colour labels to refer to objects in cat_placements dict
    # initiate to_place with all cat piece colours
    to_place = cats.copy()
    placed_cols = []

    grid_state = grid_init.copy()

    # place first cat. special case not actually coded (for later):
    # if symmetry=True, (m x n) puzzle grid has horizontal/vertical symmetry
    # this means while searching solutions we can hold the inital cat to
    # be placed within the (ceiling(m/2), ceiling(n/2)) top left subgrid.

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
        except StopIteration:
            # the gist:
            # all placement/orientation indices exhausted with no success
            # on current grid state and current piece
            # then, remove last successfully placed piece and try again
            # (since place_cat will increment idx_gen, we're guaranteed that
            # it will be placed if possible in new location)
            #----------------

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

    print(grid_state)
    return grid_state, cat_placements
