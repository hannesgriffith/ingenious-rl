import numpy as np
from matplotlib import pyplot as plt
from numba import njit

@njit
def get_playable_coords(offset=True):
    coords = np.array(((0, 4), (0, 5), (0, 6), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
                    (1, 7), (1, 8), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5),
                    (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (3, 0), (3, 1), (3, 2),
                    (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10),
                    (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7),
                    (4, 8), (4, 9), (4, 10), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4),
                    (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (6, 0), (6, 1),
                    (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9),
                    (6, 10), (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6),
                    (7, 7), (7, 8), (7, 9), (7, 10), (8, 1), (8, 2), (8, 3), (8, 4),
                    (8, 5), (8, 6), (8, 7), (8, 8), (8, 9), (9, 3), (9, 4), (9, 5),
                    (9, 6), (9, 7), (10, 5)), dtype=np.float32)
    
    if offset:
        for i in range(coords.shape[0]):
            if coords[i, 1] % 2 == 0:
                coords[i, 0] += 0.5

    return coords

@njit
def grid_to_coords(grid):
    idxs = np.where(grid == 1)
    num = idxs[0].shape[0]

    xs, ys = [], []
    for i in range(num):
        x = idxs[0][i]
        y = idxs[1][i]

        if y % 2 == 0:
            x += 0.5
        xs.append(x)
        ys.append(y)

    return (xs, ys)

@njit
def grid_to_coords_playable(grid):
    idxs = np.where(grid == 1)
    playable = get_playable_coords(offset=False).astype(np.uint8)
    num = idxs[0].shape[0]

    xs, ys = [], []
    for i in range(num):
        x = idxs[0][i]
        y = idxs[1][i]

        for j in range(playable.shape[0]):
            p1, p2 = playable[j]
            if int(x) == int(p1) and int(y) == int(p2):
                if y % 2 == 0:
                    x += 0.5
                xs.append(x)
                ys.append(y)
                break

    return (xs, ys)

def draw_coloured_numbers(numbers, start_x, start_y):
    colours = ["red", "darkorange", "gold", "forestgreen", "royalblue", "darkviolet"]
    assert numbers.shape[0] == 6
    for i in range(6):
        font = {'family': 'serif', 'color':  colours[i], 'weight': 'normal', 'size': 12,}
        plt.text(start_x + 0.1 * i, start_y, f'{numbers[i]}', fontdict=font)

def generate_debug_visualisation(vis_inputs):
    inputs, labels, preds = vis_inputs # (4,), b, b
    board_repr = inputs[0]      # b x 11 x 11 x 8
    deck_repr = inputs[1]       # b x 2 x 6
    scores_repr = inputs[2]     # b x 2 x 6
    general_repr = inputs[3]    # b x 5

    num = 4
    playable = get_playable_coords()

    fig = plt.figure(figsize=(12, 4 * num))
    for i in range(num):
        red = grid_to_coords(board_repr[i, :, :, 0])
        orange = grid_to_coords(board_repr[i, :, :, 1])
        yellow = grid_to_coords(board_repr[i, :, :, 2])
        green = grid_to_coords(board_repr[i, :, :, 3])
        blue = grid_to_coords(board_repr[i, :, :, 4])
        purple = grid_to_coords(board_repr[i, :, :, 5])
        occupied = grid_to_coords_playable(board_repr[i, :, :, 6])
        available = grid_to_coords(board_repr[i, :, :, 7])

        your_score = scores_repr[i, 0, :]
        other_score = scores_repr[i, 1, :]
        tiles_single = deck_repr[i, 0, :]
        tiles_double = deck_repr[i, 1, :]

        plt.subplot(num, 3, i * 3 + 1)
        plt.scatter(playable[:, 0], playable[:, 1], c='lightgrey')
        plt.scatter(occupied[0], occupied[1], s=50, c='grey')
        plt.axis('off')

        plt.subplot(num, 3, i * 3 + 2)
        plt.scatter(playable[:, 0], playable[:, 1], c='lightgrey')
        plt.scatter(red[0], red[1], s=80, c='red')
        plt.scatter(orange[0], orange[1], s=80, c='darkorange')
        plt.scatter(yellow[0], yellow[1], s=80, c='gold')
        plt.scatter(green[0], green[1], s=80, c='forestgreen')
        plt.scatter(blue[0], blue[1], s=80, c='royalblue')
        plt.scatter(purple[0], purple[1], s=80, c='darkviolet')
        plt.scatter(available[0], available[1], facecolors='none', edgecolors='dimgrey')
        plt.axis('off')

        plt.subplot(num, 3, i * 3 + 3)
        font = {'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 12,}
        plt.text(0, 0.95, f'Your score:', fontdict=font)
        plt.text(0, 0.85, f'Other score:', fontdict=font)
        plt.text(0, 0.7, f'Single tiles:', fontdict=font)
        plt.text(0, 0.6, f'Double tiles:', fontdict=font)
        plt.text(0, 0.45, f'Ingenious / num ingenious: {general_repr[i, 0]} / {general_repr[i, 1]}', fontdict=font)
        plt.text(0, 0.35, f'Can / should exchange:       {general_repr[i, 2]} / {general_repr[i, 3]}', fontdict=font)
        plt.text(0, 0.25, f'Move number:                     {general_repr[i, 4]}', fontdict=font)
        plt.text(0, 0.1, 'Win label: {:.2f}'.format(labels[i]), fontdict=font)
        plt.text(0, 0.0, 'Win pred: {:.2f}'.format(preds[i]), fontdict=font)
        draw_coloured_numbers(your_score, 0.5, 0.95)
        draw_coloured_numbers(other_score, 0.5, 0.85)
        draw_coloured_numbers(tiles_single, 0.5, 0.7)
        draw_coloured_numbers(tiles_double, 0.5, 0.6)
        plt.axis('off')

    return fig
