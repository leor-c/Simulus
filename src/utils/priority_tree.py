from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


class DynamicSoftmaxPriorityTree:
    def __init__(
        self,
        initial_capacity: int = 10_000,
        temperature: float = 1.0,
        *,
        dtype: np.dtype = np.float64,
        growth_factor: float = 2.0,
        recenter_threshold: float = 60.0,
        seed: int | None = None,
    ) -> None:
        if initial_capacity <= 0:
            raise ValueError("initial_capacity must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if growth_factor <= 1:
            raise ValueError("growth_factor must be greater than 1")

        self.temperature = float(temperature)
        self.dtype = np.dtype(dtype)
        self.growth_factor = float(growth_factor)
        self.recenter_threshold = float(recenter_threshold)
        self.rng = np.random.default_rng(seed)

        self.capacity = self._next_power_of_two(initial_capacity)
        self.tree_capacity = self.capacity

        # Index 0 is unused. Leaves begin at tree_capacity.
        self.tree = np.zeros(
            2 * self.tree_capacity,
            dtype=self.dtype,
        )

        self.values = np.full(
            self.capacity,
            -np.inf,
            dtype=self.dtype,
        )

        self.size = 0
        self.offset = 0.0

    @staticmethod
    def _next_power_of_two(value: int) -> int:
        return 1 << (value - 1).bit_length()

    def __len__(self) -> int:
        return self.size

    @property
    def total_weight(self) -> float:
        return float(self.tree[1])

    def _weights_from_values(
        self,
        values: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        return np.exp(
            (values - self.offset) / self.temperature
        )

    def _ensure_capacity(self, required_size: int) -> None:
        if required_size <= self.capacity:
            return

        requested = max(
            required_size,
            int(np.ceil(self.capacity * self.growth_factor)),
        )

        new_capacity = self._next_power_of_two(requested)

        old_values = self.values

        self.capacity = new_capacity
        self.tree_capacity = new_capacity

        self.values = np.full(
            new_capacity,
            -np.inf,
            dtype=self.dtype,
        )
        self.values[:self.size] = old_values[:self.size]

        self.tree = np.zeros(
            2 * new_capacity,
            dtype=self.dtype,
        )

        # Leaf indices changed, so the complete tree must be rebuilt.
        self._rebuild()

    def _rebuild(self, offset: float | None = None) -> None:
        self.tree.fill(0)

        if self.size == 0:
            if offset is not None:
                self.offset = float(offset)
            return

        active_values = self.values[:self.size]

        if offset is None:
            offset = float(active_values.max())

        self.offset = float(offset)

        leaves = (
            self.tree_capacity
            + np.arange(self.size, dtype=np.int64)
        )

        self.tree[leaves] = self._weights_from_values(
            active_values
        )

        # Build each level from the leaves upward.
        width = self.tree_capacity

        while width > 1:
            parents = np.arange(
                width // 2,
                width,
                dtype=np.int64,
            )

            self.tree[parents] = (
                self.tree[2 * parents]
                + self.tree[2 * parents + 1]
            )

            width //= 2

    def _maybe_recenter(self) -> bool:
        if self.size == 0:
            return False

        active_values = self.values[:self.size]

        max_scaled = (
            float(active_values.max()) - self.offset
        ) / self.temperature

        min_scaled = (
            float(active_values.min()) - self.offset
        ) / self.temperature

        if (
            max_scaled > self.recenter_threshold
            or min_scaled < -self.recenter_threshold
        ):
            self._rebuild(
                offset=float(active_values.max())
            )
            return True

        return False

    def append(
        self,
        values: ArrayLike,
    ) -> NDArray[np.int64]:
        """
        Append new entries and return their assigned indices.

        Unlike a circular buffer, existing entries are never overwritten.
        """
        values = np.asarray(
            values,
            dtype=self.dtype,
        ).reshape(-1)

        if values.size == 0:
            return np.empty(0, dtype=np.int64)

        if not np.all(np.isfinite(values)):
            raise ValueError("priority values must be finite")

        start = self.size
        end = start + values.size

        self._ensure_capacity(end)

        indices = np.arange(
            start,
            end,
            dtype=np.int64,
        )

        self.values[indices] = values
        self.size = end

        if start == 0:
            self.offset = float(values.max())

        if self._maybe_recenter():
            return indices

        leaves = self.tree_capacity + indices
        self.tree[leaves] = self._weights_from_values(values)

        self._update_ancestors(leaves)

        return indices

    def update(
        self,
        indices: ArrayLike,
        values: ArrayLike,
    ) -> None:
        indices = np.asarray(
            indices,
            dtype=np.int64,
        ).reshape(-1)

        values = np.asarray(
            values,
            dtype=self.dtype,
        ).reshape(-1)

        if indices.size != values.size:
            raise ValueError(
                "indices and values must have equal lengths"
            )

        if indices.size == 0:
            return

        if np.any(indices < 0) or np.any(indices >= self.size):
            raise IndexError("index does not refer to an active entry")

        if not np.all(np.isfinite(values)):
            raise ValueError("priority values must be finite")

        # Final occurrence wins for duplicate indices.
        reverse_indices = indices[::-1]
        _, reverse_positions = np.unique(
            reverse_indices,
            return_index=True,
        )

        positions = indices.size - 1 - reverse_positions
        positions.sort()

        indices = indices[positions]
        values = values[positions]

        self.values[indices] = values

        if self._maybe_recenter():
            return

        leaves = self.tree_capacity + indices
        self.tree[leaves] = self._weights_from_values(values)

        self._update_ancestors(leaves)

    def _update_ancestors(
        self,
        leaves: NDArray[np.int64],
    ) -> None:
        parents = np.unique(leaves // 2)

        while parents.size > 0:
            self.tree[parents] = (
                self.tree[2 * parents]
                + self.tree[2 * parents + 1]
            )

            if parents.size == 1 and parents[0] == 1:
                break

            parents = np.unique(parents // 2)
            parents = parents[parents > 0]

    def sample_prioritized(
        self,
        batch_size: int,
    ) -> tuple[
        NDArray[np.int64],
        NDArray[np.float64],
    ]:
        if batch_size < 0:
            raise ValueError("batch_size must be nonnegative")
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty tree")

        if batch_size == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
            )

        total = self.tree[1]

        masses = self.rng.random(batch_size) * total
        nodes = np.ones(batch_size, dtype=np.int64)

        while nodes[0] < self.tree_capacity:
            left_nodes = 2 * nodes
            left_weights = self.tree[left_nodes]

            go_right = masses >= left_weights

            masses -= go_right * left_weights
            nodes = left_nodes + go_right

        indices = nodes - self.tree_capacity

        if np.any(indices >= self.size):
            raise RuntimeError("Sampled an inactive leaf")

        probabilities = np.asarray(
            self.tree[nodes] / total,
            dtype=np.float64,
        )

        return indices, probabilities

    def sample_uniform(
        self,
        batch_size: int,
    ) -> tuple[
        NDArray[np.int64],
        NDArray[np.float64],
    ]:
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty tree")

        indices = self.rng.integers(
            0,
            self.size,
            size=batch_size,
            dtype=np.int64,
        )

        probabilities = np.full(
            batch_size,
            1.0 / self.size,
            dtype=np.float64,
        )

        return indices, probabilities
