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
            195, 211, 227, 243, 3, 19, 35, 51,
            196, 212, 228, 244, 4, 20, 36, 52,
            197, 213, 229, 245, 5, 21, 37, 53,
            198, 214, 230, 246, 6, 22, 38, 54,
            199, 215, 231, 247, 7, 23, 39, 55,
            200, 216, 232, 248, 8, 24, 40, 56,
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
        indices=[79, 95, 111, 127, 80, 96, 112, 128],
    ),
    RegionSpec(
        key="left_shoulder",
        title="左肩",
        cols=4,
        rows=1,
        indices=[9, 25, 41, 57],
    ),
    RegionSpec(
        key="right_arm",
        title="右臂",
        cols=4,
        rows=2,
        indices=[177, 162, 146, 130, 178, 161, 145, 129],
    ),
    RegionSpec(
        key="right_shoulder",
        title="右肩",
        cols=4,
        rows=1,
        indices=[249, 233, 217, 201],
    ),
]


# Arm sleeve (缠绕式手臂, JQGY-YL-132): the full 256-channel array is used as a
# single 16x16 grid. Per the spec's "手臂分区1传感点：从左到右" table, the raw
# channel order fills the grid as 129..256 then 1..128. Left and right arms are
# the same model / physical layout, so they share this mapping — only their
# source device stream differs.
ARM_REGION = RegionSpec(
    key="arm",
    title="手臂",
    cols=16,
    rows=16,
    indices=list(range(129, 257)) + list(range(1, 129)),
)


# Regions grouped by device, for viewers / offline analysis. ``vest`` reuses the
# short-sleeve body regions above; each arm is one 16x16 grid.
REGIONS_BY_DEVICE = {
    "vest": REGIONS,
    "left_arm": [ARM_REGION],
    "right_arm": [ARM_REGION],
}
