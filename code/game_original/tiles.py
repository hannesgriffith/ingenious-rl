import numpy as np

def flip_tile(tile):
    return tile[::-1]

class Tiles:
    def __init__(self):
        self.tiles = np.array([
                (1, 2), (1, 3), (1, 4), (1, 5), (1, 0), (2, 3), (2, 4), (2, 5), 
                (2, 0), (3, 4), (3, 5), (3, 0), (4, 5), (4, 0), (5, 0), (1, 1), 
                (1, 2), (1, 3), (1, 4), (1, 5), (1, 0), (2, 2), (2, 3), (2, 4), 
                (2, 5), (2, 0), (3, 3), (3, 4), (3, 5), (3, 0), (4, 4), (4, 5), 
                (4, 0), (5, 5), (5, 0), (0, 0), (1, 1), (1, 2), (1, 3), (1, 4), 
                (1, 5), (1, 0), (2, 2), (2, 3), (2, 4), (2, 5), (2, 0), (3, 3), 
                (3, 4), (3, 5), (3, 0), (4, 4), (4, 5), (4, 0), (5, 5), (5, 0), 
                (0, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 0), (2, 2), 
                (2, 3), (2, 4), (2, 5), (2, 0), (3, 3), (3, 4), (3, 5), (3, 0), 
                (4, 4), (4, 5), (4, 0), (5, 5), (5, 0), (0, 0), (1, 1), (1, 2), 
                (1, 3), (1, 4), (1, 5), (1, 0), (2, 2), (2, 3), (2, 4), (2, 5), 
                (2, 0), (3, 3), (3, 4), (3, 5), (3, 0), (4, 4), (4, 5), (4, 0), 
                (5, 5), (5, 0), (0, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), 
                (1, 0), (2, 2), (2, 3), (2, 4), (2, 5), (2, 0), (3, 3), (3, 4), 
                (3, 5), (3, 0), (4, 4), (4, 5), (4, 0), (5, 5), (5, 0), (0, 0)
        ], dtype=np.uint8)

        num_start_tiles = self.tiles.shape[0]
        self.tiles = self.tiles.reshape(num_start_tiles, 2)
        self.shuffle_tiles()

    def shuffle_tiles(self):
        num_tiles = self.tiles.shape[0]
        idxs = np.arange(num_tiles)
        np.random.shuffle(idxs)
        self.tiles = self.tiles[idxs]

    def add_tiles(self, tiles_to_add):
        num_tiles = tiles_to_add.shape[0] if tiles_to_add.ndim > 1 else 1
        tiles_to_add = tiles_to_add.reshape(num_tiles, 2)
        self.tiles = np.vstack([self.tiles, tiles_to_add])

    # def pick_tile_from_bag(self):
    #     idx = np.random.randint(0, high=self.tiles.shape[0], size=1)
    #     tile = self.tiles[idx]
    #     self.tiles = np.delete(self.tiles, idx[0], axis=0)
    #     return tile

    def pick_n_tiles_from_bag(self, n):
        self.shuffle_tiles()
        tiles = self.tiles[0:n, :]
        self.tiles = self.tiles[n:, :]
        return tiles
