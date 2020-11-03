from dataclasses import dataclass
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class TrainingConfig:
    restore_ckpt_dir: str
    game_type: str
    network_type: str
    log_every_n_steps: int
    vis_every_n_steps: int
    test_every_n_steps: int
    n_test_games: int
    n_other_games: int
    effective_batch_size: int
    max_train_batch_size: int
    max_eval_batch_size: int
    episodes_per_step: int
    updates_per_step: int
    improvement_threshold: float
    initial_learning_rate: float
    lowest_learning_rate: float
    reduce_lr_every_n: int
    replay_buffer_size: int
    weight_decay: float