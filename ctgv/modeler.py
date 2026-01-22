"""
CTGV System - Geometric Data Modeler
"""
import numpy as np
from typing import List, Tuple, Dict
from .gebit import Gebit
from .shapes import Shape

class GeometricDataModeler:
    """Converts conventional data into CTGV structures"""

    @staticmethod
    def encode_pattern(pattern: np.ndarray,
                      shape_mapping: Dict[float, Shape] = None) -> List[Gebit]:
        """
        Encodes 2D pattern into gebit network

        Args:
            pattern: 2D array with values [0,1]
            shape_mapping: Intensity → shape map

        Returns:
            List of gebits in grid order
        """
        if shape_mapping is None:
            shape_mapping = {
                0.0: Shape.INHIBITOR,
                0.3: Shape.FLOW,
                0.6: Shape.AMPLIFIER,
                0.9: Shape.ORIGIN
            }

        height, width = pattern.shape
        grid = []

        # Create gebits
        for y in range(height):
            row = []
            for x in range(width):
                intensity = float(pattern[y, x])

                # Find closest shape
                closest_val = min(shape_mapping.keys(),
                                key=lambda k: abs(k - intensity))
                shape = shape_mapping[closest_val]

                # Create gebit
                gebit = Gebit(
                    shape=shape,
                    intensity=intensity,
                    label=f"G_{y}_{x}",
                    dimensions=2
                )
                row.append(gebit)
            grid.append(row)

        # Connect in grid (4-connectivity)
        connections_count = 0
        for y in range(height):
            for x in range(width):
                current = grid[y][x]

                # Right
                if x + 1 < width:
                    right = grid[y][x + 1]
                    weight = 1.0 - abs(current.intensity - right.intensity)
                    if current.connect_to(right, weight):
                        connections_count += 1

                # Down
                if y + 1 < height:
                    below = grid[y + 1][x]
                    weight = 1.0 - abs(current.intensity - below.intensity)
                    if current.connect_to(below, weight):
                        connections_count += 1

        print(f"[Modeler] Network {height}×{width}: {height*width} gebits, "
              f"{connections_count} connections")

        # Return flat list
        return [gebit for row in grid for gebit in row]

    @staticmethod
    def decode_pattern(network: List[Gebit],
                      shape: Tuple[int, int]) -> np.ndarray:
        """Decodes gebit network to 2D pattern"""
        height, width = shape
        pattern = np.zeros((height, width), dtype=np.float32)

        for i, gebit in enumerate(network[:height * width]):
            y, x = divmod(i, width)
            if y < height and x < width:
                pattern[y, x] = gebit.state

        return pattern