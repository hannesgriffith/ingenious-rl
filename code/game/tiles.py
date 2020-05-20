import random as rn

class Tiles:
    def __init__(self):
        self.doubles = [(i, i) for i in range(1, 7)]
        self.tiles = [(i, j) for i in range(1, 7) for j in range(i, 7)] * 6
        for tile in self.doubles:
            self.tiles.remove(tile)

    def __str__(self):
        return str(self.tiles)

    def add_tiles(self, tiles_to_add):
        self.tiles.extend(tiles_to_add)

    def get_random_tile_idx(self):
        return rn.randint(0, len(self.tiles) - 1)

    def sample_random_tile(self):
        return self.tiles[self.get_random_tile_idx()]

    def pick_tile_from_bag(self):
        return self.tiles.pop(self.get_random_tile_idx())
