from dataclasses import dataclass, field


@dataclass
class Position:
    x: int
    y: int
    radius: int


@dataclass
class Contour:
    x: int
    y: int
    radius: int
    ss: float


@dataclass
class FrameData:
    frame_number: int
    analyzed: bool
    frame_read_time: float = None
    mask_creation_time: float = None
    contour_find_time: float = None
    num_contours: int = None
    num_evaluated_contours: int = None
    contour_evaluated_time: float = None
    contour_evaluated_time_avg: float = None
    ball_position: Position = None


@dataclass
class Point:
    x: int
    y: int