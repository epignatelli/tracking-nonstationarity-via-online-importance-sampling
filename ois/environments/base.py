from gym_minigrid.wrappers import *
from bsuite.utils.gym_wrapper import DMEnvFromGym


def make(name):
    env = gym.make(name)
    env = RGBImgPartialObsWrapper(env)  # Get pixel observations
    env = ImgObsWrapper(env)  # Get rid of the 'mission' field
    env = DMEnvFromGym(env)  #  Convert to dm_env.Environment
    return env
