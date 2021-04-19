import json
from dataclasses import dataclass

'''
Load experiment configuration in a nested case class from statically configured directory below. 
'''

CONFIG_DIRECTORY = "MarioWM/configs/"


# TODO: actually load the config
def get_config_from_json(experiment_name):
    return json.loads(CONFIG_DIRECTORY.join(experiment_name).join(".json"), object_hook=lambda d: Experiment(**d))


@dataclass(frozen=True)
class Param:
    use_linear_lr_decay: bool
    learning_rate: float
    use_clipped_gradient: bool
    value_loss_coeff: float
    entropy: float
    num_steps: int
    num_mini_batch: int
    epsiode_length: int


@dataclass(frozen=True)
class System:
    number_processes: int
    timeout: int


@dataclass(frozen=True)
class Logging:
    trace_size: int
    trace_length: int
    log_interval: int


@dataclass(frozen=True)
class Environment:
    def __init__(self):
        name: str
        use_weird: bool
        use_autoshroom: bool
        level: int


@dataclass(frozen=True)
class Checkpoint:
    def __init__(self):
        save_interval: int
        save_dir: str


@dataclass(frozen=True)
class Experiment:
    reward: str
    algo: str
    use_recurrent_policy: bool
    use_skip: bool
    use_gae: bool
    param: Param
    system: System
    logging: Logging
    environment: Environment
    checkpoint: Checkpoint
