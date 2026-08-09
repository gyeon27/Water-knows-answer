"""Debug teacher trajectory generation for Phase 2 pipeline validation."""

from .trajectory_writer import DebugTeacherWriter, TrajectoryConfig
from .wcsph_writer import WCSPHTeacherWriter

__all__ = ["DebugTeacherWriter", "TrajectoryConfig", "WCSPHTeacherWriter"]
