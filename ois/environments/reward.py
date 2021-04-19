from gym_minigrid.register import register
from gym_minigrid.minigrid import MiniGridEnv, Grid, Goal


class SassyEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse and non-stationary reward
    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        super().__init__(
            grid_size=size,
            max_steps=4 * size * size,
            # Set this to True for maximum speed
            see_through_walls=True,
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.place_obj(Goal(), top=(1, 1), size=(width - 2, height - 2))

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"


class EmptySassyEnv4x4(SassyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=4, **kwargs)


class EmptySassyEnv8x8(SassyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=8, **kwargs)


class EmptySassyEnv16x16(SassyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=16, **kwargs)
