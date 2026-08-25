"""ResidualBrick components integrated into the MLBricks package."""

from .controller import ResController
from .native import is_available as native_backend_available
from .native import backend_name as native_backend_name

__version__ = "0.2.1"

__all__ = [
    "ResController",
    "native_backend_available",
    "native_backend_name",
]
