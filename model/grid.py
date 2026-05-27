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
            cell_x, cell_y = self._position_to_cell_coords(initial_position)
            if 0 <= cell_x < self.cell_number and 0 <= cell_y < self.cell_number:
                self.matrix[cell_x, cell_y] = 1

    def _position_to_cell_coords(self, position):
        x, y = position
        cell_x = int((x + self.spatial_center) // self.cell_lenght)
        cell_y = int((y + self.spatial_center) // self.cell_lenght)
        return cell_x, cell_y
    
    def position_to_cell_name(self, position):
        cell_x, cell_y = self._position_to_cell_coords(position)
        return f"Cell_({cell_x},{cell_y})"

    def get_cell_value_given_position(self, position):
        cell_x, cell_y = self._position_to_cell_coords(position)
        if 0 <= cell_x < self.cell_number and 0 <= cell_y < self.cell_number:
            return self.matrix[cell_x, cell_y]
        else:
            return False  # Out of bounds
        
    def update_cell_value_given_position(self, position):
        cell_x, cell_y = self._position_to_cell_coords(position)
        if 0 <= cell_x < self.cell_number and 0 <= cell_y < self.cell_number:
            self.matrix[cell_x, cell_y] += 1

    def cell_coords_to_centroid(self, cell_x, cell_y):
        cx = (cell_x + 0.5) * self.cell_lenght - self.spatial_center
        cy = (cell_y + 0.5) * self.cell_lenght - self.spatial_center
        return (cx, cy)

    @staticmethod
    def snap_to_centroid_static(position, cell_lenght, cell_number):
        spatial_center = (cell_lenght * cell_number) / 2
        cell_x = int((position[0] + spatial_center) // cell_lenght)
        cell_y = int((position[1] + spatial_center) // cell_lenght)
        cell_x = max(0, min(cell_x, cell_number - 1))
        cell_y = max(0, min(cell_y, cell_number - 1))
        cx = (cell_x + 0.5) * cell_lenght - spatial_center
        cy = (cell_y + 0.5) * cell_lenght - spatial_center
        return (cx, cy)