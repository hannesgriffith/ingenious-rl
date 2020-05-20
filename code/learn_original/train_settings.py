# PARAMS = {
#     "game_type": "training",
#     "network_type": "network1",
#     "representation": "features_1_one_hot",
#     "exploration": "exploration1",
#     "value_type": "basic_1",
#     "lambda": None,                 # 0.8
#     "discount_rate": None,          # None
#     "log_every_n": 10,              # 10
#     "test_every_n": 250,            # 1000
#     "n_test_games": 50,             # 50
#     "n_other_games": 10,            # 10
#     "replay_buffer_size": 10000,    # 10,000
#     "batch_size": 64,               # 64
#     "initial_learning_rate": 1e-3,  # 1e-3
#     "total_training_steps": 1e5,    # 1e7
#     # "training_device": "/cpu:0"
#     "training_device": "/device:GPU:0"
# }

# PARAMS = {
#     "game_type": "training",
#     "network_type": "network1",
#     "representation": "features_1_one_hot",
#     "exploration": "exploration1",
#     "value_type": "basic_1",
#     "lambda": None,                 # 0.8
#     "discount_rate": None,          # None
#     "log_every_n": 50,              # 10
#     "test_every_n": 400,            # 1000
#     "n_test_games": 30,             # 50
#     "n_other_games": 30,            # 10
#     "replay_buffer_size": 200000,   # 100,000
#     "batch_size": 128,              # 64
#     "steps_per_episode": 10,        # 2
#     "initial_learning_rate": 1e-2,  # 1e-3
#     "total_training_steps": 40000,  # 1e7
#     "restore_session": "logs/2019-12-11_17-57",
#     # "start_ckpt": "logs/2019-11-29_09-22/weights-29750.ckpt",
#     # "best_model": "logs/2019-11-29_09-22/best_weights.ckpt",
#     # "start_replay_buffer": "logs/2019-12-08_19-59/buffer",
# }

PARAMS = {
    "game_type": "training",
    "network_type": "network1",
    "representation": "features_1_one_hot",
    "exploration": "exploration1",
    "value_type": "basic_1",
    "lambda": None,                 # 0.8
    "discount_rate": None,          # None
    "log_every_n": 20,              # 10
    "test_every_n": 200,            # 1000
    "n_test_games": 30,             # 50
    "n_other_games": 30,            # 10
    "replay_buffer_size": 1000,   # 100,000
    "batch_size": 128,              # 64
    "steps_per_episode": 10,        # 2
    "initial_learning_rate": 1e-2,  # 1e-3
    "total_training_steps": 50000,  # 1e7
}

# PARAMS = {
#     "game_type": "training",
#     "network_type": "network1",
#     "representation": "features_1_one_hot",
#     "exploration": "exploration1",
#     "value_type": "basic_1",
#     "lambda": None,                 # 0.8
#     "discount_rate": None,          # None
#     "log_every_n": 1,               # 10
#     "test_every_n": 1,              # 1000
#     "n_test_games": 1,              # 50
#     "n_other_games": 1,             # 10
#     "replay_buffer_size": 100,      # 10,000
#     "batch_size": 32,               # 64
#     "steps_per_episode": 8,         # 2
#     "initial_learning_rate": 1e-2,  # 1e-3
#     "total_training_steps": 1e6,    # 1e7
#     # "restore_session": "logs/2019-12-08_19-59",
#     "start_ckpt": "logs/2019-11-29_09-22/weights-29750.ckpt",
#     "best_model": "logs/2019-11-29_09-22/best_weights.ckpt",
#     # "start_replay_buffer": "logs/2019-12-08_19-59/buffer",
# }
