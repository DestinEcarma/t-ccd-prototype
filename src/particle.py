import matplotlib.pyplot as plt
import numpy as np


class Particle:
    def __init__(
        self, position: np.ndarray, velocity: np.ndarray, radius=1.0, mass=1.0
    ):
        self.position = position
        self.velocity = velocity
        self.radius = radius
        self.mass = mass

    def set_color(self, color: str):
        self.color = color

    def radius_velocity(self, t: float) -> np.ndarray:
        return self.position + self.velocity * t

    def world_to_grid(
        self, cell_size: np.ndarray, origin: np.ndarray
    ) -> tuple[np.ndarray]:
        return np.floor((self.position - origin) / cell_size).astype(np.int64)

    def clipped_position(
        self, boundary: tuple[np.ndarray], t: float
    ) -> tuple[np.ndarray]:
        dir = self.radius_velocity(t) - self.position
        t_min, t_max = 0.0, 1.0

        boundary_min = boundary[0]
        boundary_max = boundary[1]

        for i in range(2):
            if dir[i] != 0:
                t1 = (boundary_min[i] - self.position[i]) / dir[i]
                t2 = (boundary_max[i] - self.position[i]) / dir[i]
                t_enter = min(t1, t2)
                t_exit = max(t1, t2)

                t_min = max(t_min, t_enter)
                t_max = min(t_max, t_exit)
            elif not (boundary_min[i] <= self.position[i] <= boundary_max[i]):
                return None, None

        if t_min > t_max:
            return None, None

        clipped_start = self.position + dir * t_min
        clipped_end = self.position + dir * t_max
        return clipped_start, clipped_end

    def get_cells_along_direction(
        self, cell_size: np.ndarray, boundary: tuple[np.ndarray], t: float
    ) -> list[tuple[np.int64]]:
        cells = []

        boundary_min = boundary[0]

        start, end = self.clipped_position(boundary, t)

        delta = end - start

        step = np.sign(delta).astype(np.int64)

        start_cell = np.floor((start - boundary_min) / cell_size).astype(
            np.int64
        )
        end_cell = np.floor((end - boundary_min) / cell_size).astype(np.int64)

        t_max = np.zeros(2)
        t_delta = np.zeros(2)

        for i in range(2):
            if delta[i] != 0:
                next_boundary = (start_cell[i] + (step[i] > 0)) * cell_size[
                    i
                ] + boundary_min[i]
                t_max[i] = (next_boundary - start[i]) / delta[i]
                t_delta[i] = cell_size[i] / abs(delta[i])
            else:
                t_max[i] = np.inf
                t_delta[i] = np.inf

        curr = start_cell.copy()

        while not np.array_equal(curr, end_cell):
            cells.append(tuple(curr))
            axis = 0 if t_max[0] < t_max[1] else 1
            curr[axis] += np.int64(step[axis])
            t_max[axis] += t_delta[axis]

            if all(
                (step[i] > 0 and curr[i] > end_cell[i])
                or (step[i] < 0 and curr[i] < end_cell[i])
                or (step[i] == 0 and curr[i] == end_cell[i])
                for i in range(2)
            ):
                break

        cells.append(tuple(end_cell))

        return cells

    def get_circle_cells(
        self, cell_size: np.ndarray, boundary: tuple[np.ndarray]
    ) -> list[tuple[np.int64]]:
        cells = []
        min_bound = boundary[0]
        grid_shape = ((boundary[1] - boundary[0]) / cell_size).astype(np.int64)

        circle_min = self.position - self.radius
        circle_max = self.position + self.radius

        cell_min = np.floor((circle_min - min_bound) / cell_size).astype(
            np.int64
        )
        cell_max = np.floor((circle_max - min_bound) / cell_size).astype(
            np.int64
        )

        cell_min = np.clip(cell_min, [0, 0], grid_shape - 1)
        cell_max = np.clip(cell_max, [0, 0], grid_shape - 1)

        for i in range(cell_min[0], cell_max[0] + 1):
            for j in range(cell_min[1], cell_max[1] + 1):
                cells.append((i, j))

        return cells

    def plt_add(self, **kwargs):
        kwargs = {"color": self.color, **kwargs}

        plt.gca().add_patch(plt.Circle(self.position, self.radius, **kwargs))

    def plt_annotate(self, text: str, **kwargs):
        kwargs = {"color": "white", "fontsize": 12, **kwargs}

        plt.annotate(text, self.position, **kwargs)

    def plt_velocity(self, t: float, **kwargs):
        kwargs = {
            "arrowprops": dict(
                arrowstyle="->",
                color=kwargs.pop("color", self.color),
                lw=1,
                shrinkA=0,
                shrinkB=0,
            ),
            "zorder": 0,
            **kwargs,
        }

        plt.annotate(
            "", xytext=self.position, xy=self.radius_velocity(t), **kwargs
        )

    def plt_collision_point(self, position: np.ndarray, **kwargs):
        kwargs = {"color": self.color, "alpha": 0.5, **kwargs}

        plt.gca().add_patch(plt.Circle(position, self.radius, **kwargs))

    def plt_velocity_from_position(
        self, position: np.ndarray, t: float, **kwargs
    ):
        kwargs = {
            "arrowprops": dict(
                arrowstyle="->",
                color=kwargs.pop("color", self.color),
                lw=1,
                shrinkA=0,
                shrinkB=0,
            ),
            "zorder": 0,
            **kwargs,
        }

        plt.annotate(
            "", xytext=position, xy=position + self.velocity * t, **kwargs
        )
