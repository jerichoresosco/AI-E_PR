import numpy as np

class Grid:
    def __init__(self, cell_lenght, cell_number):
        self.cell_lenght = cell_lenght
        self.cell_number = cell_number
        self.matrix = np.zeros((cell_number, cell_number), dtype=int)
        self.spatial_center = (cell_lenght * cell_number) / 2
        center_idx = cell_number // 2

        # Central initial position counted as already visited
        self.matrix[center_idx, center_idx] = 1

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