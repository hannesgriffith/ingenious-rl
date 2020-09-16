import numpy as np
from matplotlib import pyplot as plt
from numba import njit

@njit(cache=True)
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

def split_grid_inputs(i, n):
    colour_states = i[:, :, :, :6].astype(np.int32)
    playable = i[:, :, :, 9].astype(np.int32)
    available = i[:, :, :, 10].astype(np.int32)

    scores = i[:, :, :, 10:10+24].reshape(n, 11 + 2, 21 + 4, 4, 6)
    scores *= np.array((5, 5, 5, 9)).reshape(1, 1, 1, 4, 1).astype(np.float32)
    max_total_score = np.max(scores[:, :, :, 3, :], axis=3).astype(np.int32)

    return (colour_states, playable, available, max_total_score)

def split_vector_inputs(i, n):
    num_playable = i[:, 0]
    num_playable = (num_playable * 85.).astype(np.int32)

    num_available = i[:, 1]
    num_available = (num_available * 21.).astype(np.int32)

    idx = 2
    colour_counts = i[:, idx:idx+6]
    colour_counts = (colour_counts * 21.).astype(np.int32)

    idx += 6
    score_counts = i[:, idx:idx+((3+8)*6)].reshape(n, 3 + 8, 6)
    factors = np.array((21, 45, 1, 9, 9, 9, 9, 9, 9, 9, 9), dtype=np.float32)
    score_counts *= factors.reshape(1, 3 + 8, 1)
    score_counts = score_counts.astype(np.int32)

    idx += (3+8)*6
    deck_repr = i[:, idx:idx+(2*6)].reshape(n, 2, 6)
    deck_repr = (deck_repr * 4.).astype(np.int32)

    idx += 2*6
    scores_repr = i[:, idx:idx+(3*6)].reshape(n, 3, 6)
    scores_repr = (scores_repr * 18.).astype(np.int32)

    idx += 3*6
    general_repr = i[:, idx:]
    general_repr *= np.array(((1, 2, 1, 1, 40))).astype(np.float32)
    general_repr = general_repr.astype(np.int32)

    return (num_playable, num_available, colour_counts, score_counts, deck_repr, scores_repr, general_repr)

def split_inputs(inputs, n):
    (grid_inputs, _), vector_inputs = inputs
    grid_inputs = np.transpose(grid_inputs, (0, 2, 3, 1)) # NCHW -> NHWC
    vis_grid_inputs = split_grid_inputs(grid_inputs[:n], n)
    vis_vector_inputs = split_vector_inputs(vector_inputs[:n], n)
    return vis_grid_inputs, vis_vector_inputs

@njit(cache=True)
def grid_to_coords(grid):
    idxs = np.where(grid.T.astype(np.uint8) == 1)
    num = idxs[0].shape[0]

    xs, ys = [], []
    for i in range(num):
        x = idxs[0][i]
        y = idxs[1][i]

        xs.append(x / 2)
        ys.append(y)

    return (xs, ys)

def draw_coloured_numbers(numbers, start_x, start_y):
    colours = ["red", "darkorange", "gold", "forestgreen", "royalblue", "darkviolet"]
    assert numbers.shape[0] == 6
    for i in range(6):
        font = {'family': 'serif', 'color':  colours[i], 'weight': 'normal', 'size': 12,}
        plt.text(start_x + 0.1 * i, start_y, f'{numbers[i]}', fontdict=font)

def generate_debug_visualisation(vis_inputs, num=4):
    inputs, labels, preds = vis_inputs
    vis_grid_inputs, vis_vector_inputs = split_inputs(inputs, num)

    colour_states, _, available_grid, _ = vis_grid_inputs
    num_playable, num_available, colour_counts, score_counts, deck_repr, scores_repr, general_repr = vis_vector_inputs

    colour_states = colour_states[:, 1:-1, 2:-2, :]
    available_grid = available_grid[:, 1:-1, 2:-2]

    labels = labels.astype(np.float)
    labels[labels == 0] = -1

    playable = get_playable_coords()

    fig = plt.figure(figsize=(16, 4 * num))
    for i in range(num):
        red = grid_to_coords(colour_states[i, :, :, 0])
        orange = grid_to_coords(colour_states[i, :, :, 1])
        yellow = grid_to_coords(colour_states[i, :, :, 2])
        green = grid_to_coords(colour_states[i, :, :, 3])
        blue = grid_to_coords(colour_states[i, :, :, 4])
        purple = grid_to_coords(colour_states[i, :, :, 5])
        available = grid_to_coords(available_grid[i])

        your_score = scores_repr[i, 0, :].astype(np.int32)
        other_score = scores_repr[i, 1, :].astype(np.int32)
        scores_diff = scores_repr[i, 2, :].astype(np.int32)
        tiles_single = deck_repr[i, 0, :].astype(np.int32)
        tiles_double = deck_repr[i, 1, :].astype(np.int32)

        plt.subplot(num, 4, i * 4 + 1)
        plt.scatter(playable[:, 0], playable[:, 1], c='lightgrey')
        plt.scatter(red[0], red[1], s=80, c='red')
        plt.scatter(orange[0], orange[1], s=80, c='darkorange')
        plt.scatter(yellow[0], yellow[1], s=80, c='gold')
        plt.scatter(green[0], green[1], s=80, c='forestgreen')
        plt.scatter(blue[0], blue[1], s=80, c='royalblue')
        plt.scatter(purple[0], purple[1], s=80, c='darkviolet')
        plt.scatter(available[0], available[1], facecolors='none', edgecolors='dimgrey')
        plt.axis('off')

        plt.subplot(num, 4, i * 4 + 2)
        font = {'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 12}
        plt.text(0, 1.0, f'Your score:', fontdict=font)
        plt.text(0, 0.9, f'Other score:', fontdict=font)
        plt.text(0, 0.8, 'Score diff:', fontdict=font)
        plt.text(0, 0.6, f'Single tiles:', fontdict=font)
        plt.text(0, 0.5, f'Double tiles:', fontdict=font)
        plt.text(0, 0.3, 'Colour counts:', fontdict=font)
        plt.text(0, 0.2, 'Scoring counts:', fontdict=font)
        draw_coloured_numbers(your_score, 0.55, 1.0)
        draw_coloured_numbers(other_score, 0.55, 0.9)
        draw_coloured_numbers(scores_diff, 0.55, 0.8)
        draw_coloured_numbers(tiles_single, 0.55, 0.6)
        draw_coloured_numbers(tiles_double, 0.55, 0.5)
        draw_coloured_numbers(colour_counts[i], 0.55, 0.3)
        draw_coloured_numbers(score_counts[i, 0, :], 0.55, 0.2)
        plt.axis('off')

        plt.subplot(num, 4, i * 4 + 3)
        font = {'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 12}
        plt.text(0, 1.0, 'Total scores:', fontdict=font)
        plt.text(0, 0.9, 'Total scores = 0:', fontdict=font)
        plt.text(0, 0.7, 'Max score 1:', fontdict=font)
        plt.text(0, 0.6, 'Max score 2:', fontdict=font)
        plt.text(0, 0.5, 'Max score 3:', fontdict=font)
        plt.text(0, 0.4, 'Max score 4:', fontdict=font)
        plt.text(0, 0.3, 'Max score 5:', fontdict=font)
        plt.text(0, 0.2, 'Max score 6:', fontdict=font)
        plt.text(0, 0.1, 'Max score 7:', fontdict=font)
        plt.text(0, 0.0, 'Max score 8:', fontdict=font)
        draw_coloured_numbers(score_counts[i, 1, :], 0.55, 1.0)
        draw_coloured_numbers(score_counts[i, 2, :], 0.55, 0.9)
        draw_coloured_numbers(score_counts[i, 3, :], 0.55, 0.7)
        draw_coloured_numbers(score_counts[i, 4, :], 0.55, 0.6)
        draw_coloured_numbers(score_counts[i, 5, :], 0.55, 0.5)
        draw_coloured_numbers(score_counts[i, 6, :], 0.55, 0.4)
        draw_coloured_numbers(score_counts[i, 7, :], 0.55, 0.3)
        draw_coloured_numbers(score_counts[i, 8, :], 0.55, 0.2)
        draw_coloured_numbers(score_counts[i, 9, :], 0.55, 0.1)
        draw_coloured_numbers(score_counts[i, 10, :], 0.55, 0.0)
        plt.axis('off')

        plt.subplot(num, 4, i * 4 + 4)
        font = {'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 12}
        plt.text(0, 1.0, f'Num playable:  {num_playable[i]}', fontdict=font)
        plt.text(0, 0.9, f'Num available: {num_available[i]}', fontdict=font)
        plt.text(0, 0.8, f'Ingenious:   {general_repr[i, 0]}', fontdict=font)
        plt.text(0, 0.7, f'# ingenious: {general_repr[i, 1]}', fontdict=font)
        plt.text(0, 0.6, f'Can exchange:     {general_repr[i, 2]}', fontdict=font)
        plt.text(0, 0.5, f'Should exchange:  {general_repr[i, 3]}', fontdict=font)
        plt.text(0, 0.4, f'Move #:      {general_repr[i, 4]}', fontdict=font)
        plt.text(0, 0.2, 'Win pred: {:.2f}'.format(preds[i]), fontdict=font)
        plt.text(0, 0.1, 'Win label: {:.2f}'.format(labels[i]), fontdict=font)
        plt.axis('off')

    return fig