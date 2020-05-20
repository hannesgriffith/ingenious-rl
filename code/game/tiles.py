from numba import njit, jitclass, uint8
import numpy as np

@njit
def flip_tile(tile):
    return tile[::-1]

tiles_spec = [
    ('tiles', uint8[:, :]),
]

@jitclass(tiles_spec)
class Tiles:
    def __init__(self):
        self.tiles = np.zeros((120, 2), dtype=np.uint8)
        self.initialise_tiles()

    def initialise_tiles(self):
        self.tiles = np.array([
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
        self.shuffle_tiles()

    def shuffle_tiles(self):
        num_tiles = self.tiles.shape[0]
        idxs = np.arange(num_tiles)
        np.random.shuffle(idxs)
        self.tiles = self.tiles[idxs]

    def add_tiles(self, tiles_to_add):
        if tiles_to_add.ndim > 1:
            num_tiles = tiles_to_add.shape[0]
        else:
            num_tiles = 1

        tiles_to_add = tiles_to_add.reshape(num_tiles, 2)
        self.tiles = np.vstack((self.tiles, tiles_to_add))

    def pick_n_tiles_from_bag(self, n):
        self.shuffle_tiles()
        tiles = self.tiles[0:n, :]
        self.tiles = self.tiles[n:, :]
        return tiles
