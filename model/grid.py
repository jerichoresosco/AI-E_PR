import numpy as np

class Grid:
    def __init__(self, cell_lenght, cell_number, initial_position=None):
        """
        Parameters
        ----------
        cell_lenght : int
            Side length of each square cell.
        cell_number : int
            Number of cells per side.
        initial_position : tuple of float, optional
            (x, y) Cartesian coordinates of the user's starting position.
            The cell containing this position is pre-marked as visited (count = 1).
            When None (default) the central cell is used, corresponding to (0, 0).
        """
        self.cell_lenght = cell_lenght
        self.cell_number = cell_number
        self.matrix = np.zeros((cell_number, cell_number), dtype=int)
        self.spatial_center = (cell_lenght * cell_number) / 2

        # Mark the initial position's cell as already visited
        if initial_position is None:
            center_idx = cell_number // 2
            self.matrix[center_idx, center_idx] = 1
        else:
            x, y = initial_position
            cell_x = int((x + self.spatial_center) // self.cell_lenght)
            cell_y = int((y + self.spatial_center) // self.cell_lenght)
            if 0 <= cell_x < self.cell_number and 0 <= cell_y < self.cell_number:
                self.matrix[cell_x, cell_y] = 1

    def get_cell_value_given_position(self, position):
        x, y = position
        cell_x = int((x + self.spatial_center) // self.cell_lenght)
        cell_y = int((y + self.spatial_center) // self.cell_lenght)
        if 0 <= cell_x < self.cell_number and 0 <= cell_y < self.cell_number:
            return self.matrix[cell_x, cell_y]
        else:
            return False  # Out of bounds
        
    def update_cell_value_given_position(self, position):
        x, y = position
        cell_x = int((x + self.spatial_center) // self.cell_lenght)
        cell_y = int((y + self.spatial_center) // self.cell_lenght)
        if 0 <= cell_x < self.cell_number and 0 <= cell_y < self.cell_number:
            self.matrix[cell_x, cell_y] += 1