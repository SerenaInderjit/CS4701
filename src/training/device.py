import torch


def resolve_device(requested: str = "auto") -> str:
    """Pick the best available device.

    Priority: requested (if not "auto") > MPS (Apple Silicon) > CUDA > CPU.
    This lets the same config run on any machine without changes.
    """
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
