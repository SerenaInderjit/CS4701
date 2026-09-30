from collections import deque
from typing import Deque, Tuple

import numpy as np


def _area_resize_axis(x: np.ndarray, out_size: int, axis: int) -> np.ndarray:
    """Average-pool `x` along one axis to `out_size` bins.

    Uses the same bin edges as adaptive average pooling, so every source pixel
    contributes and thin sprites are not lost the way they are with
    nearest-neighbour sampling.
    """
    n = x.shape[axis]
    idx = np.arange(out_size)
    starts = (idx * n) // out_size
    ends = -((-(idx + 1) * n) // out_size)  # ceil((i + 1) * n / out_size)

    cumulative = np.cumsum(x, axis=axis, dtype=np.float64)
    zero_shape = list(x.shape)
    zero_shape[axis] = 1
    cumulative = np.concatenate([np.zeros(zero_shape), cumulative], axis=axis)

    sums = np.take(cumulative, ends, axis=axis) - np.take(cumulative, starts, axis=axis)
    counts_shape = [1] * x.ndim
    counts_shape[axis] = out_size
    return sums / (ends - starts).reshape(counts_shape)


class ObservationPreprocessor:
    """Turns raw NES frames (240x256x3, uint8) into stacked network inputs.

    Output is channel-first uint8 with shape (num_stack * channels, H, W),
    ready to feed to a CNN after dividing by 255.
    """

    GRAY_WEIGHTS = np.array([0.299, 0.587, 0.114])

    def __init__(
        self,
        frame_size: Tuple[int, int] = (84, 84),
        num_stack: int = 4,
        grayscale: bool = True,
    ):
        if num_stack < 1:
            raise ValueError("num_stack must be >= 1")
        self.frame_size = frame_size
        self.num_stack = num_stack
        self.grayscale = grayscale
        self._frames: Deque[np.ndarray] = deque(maxlen=num_stack)

    @property
    def channels_per_frame(self) -> int:
        return 1 if self.grayscale else 3

    @property
    def observation_shape(self) -> Tuple[int, int, int]:
        height, width = self.frame_size
        return (self.num_stack * self.channels_per_frame, height, width)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Convert one raw HxWx3 frame to a (C, H, W) uint8 array."""
        x = np.asarray(frame, dtype=np.float64)
        if self.grayscale:
            x = (x @ self.GRAY_WEIGHTS)[..., np.newaxis]  # H, W, 1
        height, width = self.frame_size
        x = _area_resize_axis(x, height, axis=0)
        x = _area_resize_axis(x, width, axis=1)
        x = np.clip(np.rint(x), 0, 255).astype(np.uint8)
        return np.transpose(x, (2, 0, 1))

    def _stacked(self) -> np.ndarray:
        return np.concatenate(list(self._frames), axis=0)

    def reset(self, frame: np.ndarray) -> np.ndarray:
        """Start a new episode: fill the whole stack with the first frame."""
        processed = self.process_frame(frame)
        self._frames.clear()
        for _ in range(self.num_stack):
            self._frames.append(processed)
        return self._stacked()

    def step(self, frame: np.ndarray) -> np.ndarray:
        """Push a new frame and return the updated stack (oldest frame first)."""
        if not self._frames:
            return self.reset(frame)
        self._frames.append(self.process_frame(frame))
        return self._stacked()
