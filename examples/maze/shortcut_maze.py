from examples.maze.maze import MazeEnvironment


class ShortcutMazeEnvironment(MazeEnvironment):
    change_at_n = 0
    timesteps = 0

    def __init__(self, maze_dim, obstacles, start_location, final_location, change_at_n):
        super().__init__(maze_dim, obstacles, start_location, final_location)
        self.change_at_n = change_at_n

    def reset(self):
        super().reset()
        self.timesteps = 0

    def step(self):
        self.timesteps += 1
        if self.timesteps == self.change_at_n:
            self.obstacles = self.obstacles[:-1]

        super().step()
