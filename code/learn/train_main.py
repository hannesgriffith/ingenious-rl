import numpy as np

from game.gameplay import TrainingGameplay
from utils.io import load_json

settings_path = "train_settings.json"
weights_dir = "weights"
models_dir = "models"

def train():
    pass

def main():
    params = load_json(settings_path)
    game = TrainingGameplay(params)
    net = Network(params)

if __name__ == "__main__":
    main()
