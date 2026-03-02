"""Spatial graph propagation module for multi-node congestion forecasting.

Models congestion spillover across connected roads using adjacency matrices
and iterative graph convolution.
"""

import numpy as np
from typing import List, Dict, Any, Tuple


class GraphEngine:
    """Propagates congestion across a network of roads.

    Uses graph diffusion: c_new = alpha * A @ c + (1-alpha) * c
    where A is the adjacency matrix, c is congestion, and alpha controls spillover.
    """

    def __init__(self, adjacency_matrix: np.ndarray, alpha: float = 0.3, iterations: int = 1):
        """Initialize graph engine with road network topology.

        Args:
            adjacency_matrix: K x K matrix where A[i,j] > 0 means road j influences road i.
                             Usually row-normalized so each row sums to 1.
            alpha: Propagation factor in [0,1]. Higher = more spillover from neighbors.
            iterations: Number of diffusion steps per propagation call.
        """
        self.adjacency_matrix = np.asarray(adjacency_matrix, dtype=np.float32)
        self.num_roads = self.adjacency_matrix.shape[0]
        self.alpha = alpha
        self.iterations = iterations

        # Validate square matrix
        if self.adjacency_matrix.shape != (self.num_roads, self.num_roads):
            raise ValueError(f"adjacency_matrix must be square; got {self.adjacency_matrix.shape}")

        if not (0 <= alpha <= 1):
            raise ValueError(f"alpha must be in [0,1]; got {alpha}")

    def propagate(self, congestion: np.ndarray) -> np.ndarray:
        """Apply graph-based congestion propagation.

        Args:
            congestion: 1D array of shape (num_roads,) representing congestion per road [0,1].

        Returns:
            1D array of propagated congestion values.
        """
        congestion = np.asarray(congestion, dtype=np.float32).flatten()

        if len(congestion) != self.num_roads:
            raise ValueError(
                f"congestion length {len(congestion)} != num_roads {self.num_roads}"
            )

        # Iterative diffusion: c_{t+1} = alpha * A @ c_t + (1-alpha) * c_t
        result = congestion.copy()
        for _ in range(self.iterations):
            neighbor_influence = self.adjacency_matrix @ result
            result = self.alpha * neighbor_influence + (1 - self.alpha) * result

        # Clip to [0,1] to represent valid congestion range
        result = np.clip(result, 0.0, 1.0)

        return result

    def propagate_batch(self, congestion_batch: np.ndarray) -> np.ndarray:
        """Propagate congestion for multiple time horizons or samples.

        Args:
            congestion_batch: 2D array of shape (horizon, num_roads) or (num_samples, num_roads).

        Returns:
            2D array with propagated values for each sample/horizon.
        """
        congestion_batch = np.asarray(congestion_batch, dtype=np.float32)

        if congestion_batch.ndim != 2:
            raise ValueError(f"congestion_batch must be 2D; got shape {congestion_batch.shape}")

        if congestion_batch.shape[1] != self.num_roads:
            raise ValueError(
                f"congestion_batch columns {congestion_batch.shape[1]} != num_roads {self.num_roads}"
            )

        # Propagate each row independently
        propagated = np.array([self.propagate(row) for row in congestion_batch])
        return propagated

    def build_adjacency_from_edges(edges: List[Tuple[int, int]], num_roads: int, 
                                   normalize: bool = True) -> np.ndarray:
        """Construct adjacency matrix from edge list.

        Args:
            edges: List of (source, target) tuples, where influence flows from target to source.
            num_roads: Total number of roads in the network.
            normalize: If True, row-normalize so each row sums to 1.

        Returns:
            Adjacency matrix A of shape (num_roads, num_roads).
        """
        A = np.zeros((num_roads, num_roads), dtype=np.float32)

        for src, tgt in edges:
            if not (0 <= src < num_roads and 0 <= tgt < num_roads):
                raise ValueError(f"edge ({src}, {tgt}) out of range [0, {num_roads})")
            A[src, tgt] = 1.0

        if normalize:
            row_sum = A.sum(axis=1, keepdims=True)
            # avoid division by zero for isolated nodes
            row_sum[row_sum == 0] = 1.0
            A = A / row_sum

        return A

    def set_alpha(self, alpha: float):
        """Update propagation factor."""
        if not (0 <= alpha <= 1):
            raise ValueError(f"alpha must be in [0,1]; got {alpha}")
        self.alpha = alpha

    def set_iterations(self, iterations: int):
        """Update number of diffusion iterations."""
        if iterations < 1:
            raise ValueError(f"iterations must be >= 1; got {iterations}")
        self.iterations = iterations
