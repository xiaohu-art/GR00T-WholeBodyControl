from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegionSpec:
    key: str
    title: str
    cols: int
    rows: int
    indices: list[int]


# Values are one-based positions in the 256-byte raw sensor array.
REGIONS = [
    RegionSpec(
        key="front_chest",
        title="前胸",
        cols=8,
        rows=6,
        indices=[
            51, 35, 19, 3, 243, 227, 211, 195,
            52, 36, 20, 4, 244, 228, 212, 196,
            53, 37, 21, 5, 245, 229, 213, 197,
            54, 38, 22, 6, 246, 230, 214, 198,
            55, 39, 23, 7, 247, 231, 215, 199,
            56, 40, 24, 8, 248, 232, 216, 200,
        ],
    ),

    RegionSpec(
        key="back",
        title="后背",
        cols=8,
        rows=5,
        indices=[
            58, 42, 26, 10, 250, 234, 218, 202,
            59, 43, 27, 11, 251, 235, 219, 203,
            60, 44, 28, 12, 252, 236, 220, 204,
            61, 45, 29, 13, 253, 237, 221, 205,
            62, 46, 30, 14, 254, 238, 222, 206,
        ],
    ),

    RegionSpec(
        key="left_arm",
        title="左臂",
        cols=4,
        rows=2,
        indices=[
            79, 95, 111, 127,
            80, 96, 112, 128,
        ],
    ),

    RegionSpec(
        key="left_shoulder",
        title="左肩",
        cols=4,
        rows=1,
        indices=[
            9, 25, 41, 57,
        ],
    ),

    RegionSpec(
        key="right_arm",
        title="右臂",
        cols=4,
        rows=2,
        indices=[
            177, 161, 145, 129,
            178, 162, 146, 130,
        ],
    ),

    RegionSpec(
        key="right_shoulder",
        title="右肩",
        cols=4,
        rows=1,
        indices=[
            249, 233, 217, 201,
        ],
    ),
]
