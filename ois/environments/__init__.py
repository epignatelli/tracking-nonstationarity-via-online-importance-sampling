from .actions import *
from .base import *
from .reward import *
from .windy import *


# register random reward env to gym
register_sassy_env = lambda size: register(
    "MiniGrid-Empty-Sassy-{}x{}-v0".format(size),
    entry_point="ois.environments:EmptySassy{}x{}".format(size, size),
)
register_sassy_env(4)
register_sassy_env(8)
register_sassy_env(16)


# register windy env to gym
register_windy_env = lambda size: register(
    "MiniGrid-Empty-Windy-{}x{}-v0".format(size),
    entry_point="ois.environments:EmptyWindy{}x{}".format(size, size),
)
register_windy_env(4)
register_windy_env(8)
register_windy_env(16)


# register possessed env to gym
register_possessed_env = lambda size: register(
    "MiniGrid-Empty-Possessed-{}x{}-v0".format(size),
    entry_point="ois.environments:EmptyPossessed{}x{}".format(size, size),
)
register_possessed_env(4)
register_possessed_env(8)
register_possessed_env(16)