from numba import njit
import numpy as np

class Tiles:
    def __init__(self):
        self.tiles = fast_initialise_tiles()
        self.shuffle_tiles()

    def shuffle_tiles(self):
        self.tiles = fast_shuffle_tiles(self.tiles)

    def add_tiles(self, tiles_to_add):
        self.tiles = fast_add_tiles(self.tiles, tiles_to_add)

    def pick_n_tiles_from_bag(self, n):
        self.shuffle_tiles()
        picked_tiles, self.tiles = fast_pick_n_tiles_from_bag(self.tiles, n)
        return picked_tiles

@njit(cache=True)
def flip_tile(tile):
    return tile[::-1]

@njit(cache=True)
def fast_shuffle_tiles(tiles):
    num_tiles = tiles.shape[0]
    idxs = np.arange(num_tiles)
    np.random.shuffle(idxs)
    tiles = tiles[idxs]
    return tiles

@njit(cache=True)
def fast_add_tiles(tiles, tiles_to_add):
    if tiles_to_add.ndim > 1:
        num_tiles = tiles_to_add.shape[0]
    else:
        num_tiles = 1

    tiles_to_add = tiles_to_add.reshape(num_tiles, 2)
    tiles = np.vstack((tiles, tiles_to_add))

    return tiles

@njit(cache=True)
def fast_pick_n_tiles_from_bag(tiles, n):
    picked_tiles = tiles[0:n, :]
    remaining_tiles = tiles[n:, :]
    return picked_tiles, remaining_tiles

@njit(cache=True)
def fast_initialise_tiles():
    return np.array([
            [1, 2], [1, 3], [1, 4], [1, 5], [1, 0], [2, 3], [2, 4], [2, 5], 
            [2, 0], [3, 4], [3, 5], [3, 0], [4, 5], [4, 0], [5, 0], [1, 1], 
            [1, 2], [1, 3], [1, 4], [1, 5], [1, 0], [2, 2], [2, 3], [2, 4], 
            [2, 5], [2, 0], [3, 3], [3, 4], [3, 5], [3, 0], [4, 4], [4, 5], 
            [4, 0], [5, 5], [5, 0], [0, 0], [1, 1], [1, 2], [1, 3], [1, 4], 
            [1, 5], [1, 0], [2, 2], [2, 3], [2, 4], [2, 5], [2, 0], [3, 3], 
            [3, 4], [3, 5], [3, 0], [4, 4], [4, 5], [4, 0], [5, 5], [5, 0], 
            [0, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 0], [2, 2], 
            [2, 3], [2, 4], [2, 5], [2, 0], [3, 3], [3, 4], [3, 5], [3, 0], 
            [4, 4], [4, 5], [4, 0], [5, 5], [5, 0], [0, 0], [1, 1], [1, 2], 
            [1, 3], [1, 4], [1, 5], [1, 0], [2, 2], [2, 3], [2, 4], [2, 5], 
            [2, 0], [3, 3], [3, 4], [3, 5], [3, 0], [4, 4], [4, 5], [4, 0], 
            [5, 5], [5, 0], [0, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], 
            [1, 0], [2, 2], [2, 3], [2, 4], [2, 5], [2, 0], [3, 3], [3, 4], 
            [3, 5], [3, 0], [4, 4], [4, 5], [4, 0], [5, 5], [5, 0], [0, 0]], dtype=np.uint8)