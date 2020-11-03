# ingenious-rl

Code to train and play against a reinforcement learning (RL) agent at the game of [Ingenious](https://en.wikipedia.org/wiki/Ingenious_(board_game)).

![alt text](docs/ingenious_UI.png "Gameplay UI")

## Usage

To play or train install the dependencies given in `requirements.txt`.

To play the agent simply run `python main.py`. Game settings are in `settings_play.json`.

To train a model, set your hyperparameters in `learn/configs/<your-training-config>.json`. Then run `python -m learn.train <path/to/your/trainin/config/json>`.

Note that the first time you play or train there will be a couple delays that may last some minutes where numba is compiling. However, this is cached so won't happen on subsequent runs.

## Background

The RL approach was inspired by both Tesauro's TD-Gammon and DeepMind's AlphaGo algorithms. Design details are given in [project-details](docs/project-details.md). A summary of learnings from the project are given in [experimental-learnings](docs/experimental-learnings.md).

## To-Do's

- Train an updated CNN model
- add docs (project-details & experimental-learnings)
- improve UI code
