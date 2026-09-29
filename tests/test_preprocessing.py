import numpy as np

from src.preprocessing import ObservationPreprocessor


def raw_frame(value=None):
    if value is None:
        return np.random.randint(0, 256, size=(240, 256, 3), dtype=np.uint8)
    return np.full((240, 256, 3), value, dtype=np.uint8)


def test_process_frame_shape_and_dtype():
    pre = ObservationPreprocessor()
    out = pre.process_frame(raw_frame())
    assert out.shape == (1, 84, 84)
    assert out.dtype == np.uint8


def test_constant_frame_keeps_its_value():
    pre = ObservationPreprocessor()
    assert (pre.process_frame(raw_frame(100)) == 100).all()
    assert (pre.process_frame(raw_frame(255)) == 255).all()


def test_resize_averages_pixels_instead_of_dropping_them():
    pre = ObservationPreprocessor(frame_size=(1, 1))
    frame = raw_frame(0)
    frame[:120] = 200  # top half bright
    assert abs(int(pre.process_frame(frame)[0, 0, 0]) - 100) <= 1


def test_color_mode_keeps_three_channels():
    pre = ObservationPreprocessor(grayscale=False, num_stack=2)
    assert pre.process_frame(raw_frame()).shape == (3, 84, 84)
    assert pre.reset(raw_frame()).shape == (6, 84, 84)
    assert pre.observation_shape == (6, 84, 84)


def test_reset_fills_stack_with_first_frame():
    pre = ObservationPreprocessor(num_stack=4)
    stacked = pre.reset(raw_frame(50))
    assert stacked.shape == (4, 84, 84)
    assert (stacked == 50).all()


def test_step_shifts_stack_oldest_first():
    pre = ObservationPreprocessor(num_stack=3)
    pre.reset(raw_frame(10))
    pre.step(raw_frame(20))
    stacked = pre.step(raw_frame(30))
    assert [int(stacked[i, 0, 0]) for i in range(3)] == [10, 20, 30]
    stacked = pre.step(raw_frame(40))
    assert [int(stacked[i, 0, 0]) for i in range(3)] == [20, 30, 40]


def test_reset_clears_previous_episode():
    pre = ObservationPreprocessor(num_stack=2)
    pre.reset(raw_frame(10))
    pre.step(raw_frame(20))
    stacked = pre.reset(raw_frame(99))
    assert (stacked == 99).all()
