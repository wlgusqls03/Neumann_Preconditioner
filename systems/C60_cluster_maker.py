# File: tools/build_c60_cluster_compact.py
"""
Compact non-periodic C60 clusters for real-space solvers.
- Rectangular cell to minimize volume.
- User controls center spacing and wall gap.

Cheatsheet: parameters controlling C60 dimer clusters for real-space solvers.

Key formulas (n=2, line):
  Lx = spacing + 7.1 + 2*wall_gap
  Ly = Lz = 7.1 + 2*wall_gap
Memory ~ (Lx*Ly*Lz) / (Δx^3)   [Δx = runtime --spacing, e.g., 0.4 Å]

Recommended presets:
  # Tight but safe (메모리 최소)
  spacing = 8.5   # Å
  wall_gap = 4.0  # Å
  rectangular = True
  random_orientation = False
  => cell ≈ [23.6, 15.1, 15.1] Å

  # vdW 접촉 연구
  spacing = 10.5
  wall_gap = 4.0
  => cell ≈ [27.6, 15.1, 15.1] Å

  # 거의 비상호작용
  spacing = 12.0
  wall_gap = 5.0
  => cell ≈ [29.1, 17.1, 17.1] Å

When to change what:
  - 줄여야 할 때: 먼저 cell(box 또는 [Lx,Ly,Lz])를 줄이고, 다음에 Δx(--spacing)를 줄여 정확도 올린다.
  - 상호작용을 세게: spacing ↓ (단, 원자간 최소거리 < ~3 Å는 피함).
  - 경계 효과가 보이면: wall_gap ↑ (3→5 Å).
  - 무작위 회전이 꼭 필요 없으면 random_orientation=False로 외접구 축소.

"""

from __future__ import annotations
import math
from typing import Iterable, Tuple, List, Optional
from ase import Atoms
from ase.io import read, write

C60_DIAMETER = 7.1    # 떨어진 C60 분자 지름
C60_RADIUS = C60_DIAMETER / 2.0  # C60 반지


def _random_rotate(mol: Atoms, enable: bool) -> Atoms:
    mol = mol.copy()
    if enable:
        for ang, ax in zip(
            (0.0, 0.0, 0.0), ("x", "y", "z")
        ):  # keep fixed unless enabled out름side
            pass
    return mol


def _place_on_positions(
    mol: Atoms, offsets: Iterable[Tuple[float, float, float]]
) -> Atoms:
    out = None
    for off in offsets:
        m = mol.copy()
        m.translate(off)
        out = m if out is None else out + m
    return out


def _line_offsets(n: int, spacing: float) -> List[Tuple[float, float, float]]:
    """Centers along +x from 0."""
    return [(i * spacing, 0.0, 0.0) for i in range(n)]


def _compact_cell_for_line(
    n: int, spacing: float, wall_gap: float, rectangular: bool
) -> Tuple[float, float, float]:
    span_x = (n - 1) * spacing + C60_DIAMETER
    if rectangular:
        Lx = span_x + 2 * wall_gap
        Ly = C60_DIAMETER + 2 * wall_gap
        Lz = Ly
    else:
        Lcube = max(span_x, C60_DIAMETER) + 2 * wall_gap
        Lx = Ly = Lz = Lcube
    return (Lx, Ly, Lz)


def build_c60_cluster(
    input_xyz: str = "C60.xyz",  # 기본 C60 파일
    n_molecules: int = 2,   # C60 입자수 개수
    spacing: float = 8.5,  # C60 입자 사이 거리
    wall_gap: float = 4.0,  # atom-to-wall margin
    rectangular: bool = True,  # minimize volume
    random_orientation: bool = False,  # keep orientation fixed for compactness
    output_xyz: Optional[str] = None,
) -> str:
    base: Atoms = read(input_xyz)
    # keep as non-periodic molecule
    base.set_pbc([False, False, False])

    # positions for N=2 on x-axis
    offsets = _line_offsets(n_molecules, spacing=spacing)

    # center first molecule near origin to keep coords small/positive later
    base.translate(-base.get_center_of_mass())

    cluster = _place_on_positions(base, offsets)

    # move so min >= gap
    mins = cluster.get_positions().min(axis=0)
    cluster.translate(
        [
            max(0.0, wall_gap - mins[0]),
            max(0.0, wall_gap - mins[1]),
            max(0.0, wall_gap - mins[2]),
        ]
    )

    # compact cell
    Lx, Ly, Lz = _compact_cell_for_line(n_molecules, spacing, wall_gap, rectangular)
    cluster.set_cell([Lx, Ly, Lz])
    cluster.set_pbc([False, False, False])

    if output_xyz is None:
        tag = "rect" if rectangular else "cube"
        output_xyz = f"C60_dimer_{tag}_s{spacing:.1f}_gap{wall_gap:.1f}.xyz"
    write(output_xyz, cluster, format="extxyz")
    return output_xyz


if __name__ == "__main__":
    # Example: tight but safe box for 2×C60
    out1 = build_c60_cluster(
        input_xyz="C60.xyz",  # 단일 C60 분자 파일
        n_molecules=2,  # C60 분자 수 
        spacing=10,  # 8.5~10.5 정도 적정, 분자끼리 떨어진 거리
        wall_gap=4.0,  # 3~5 is typical
        rectangular=True,  # minimize volume  부피 최소로 설정 --> 메모리 이슈로 진행
        random_orientation=False,
    )
    print(f"Wrote: {out1}")
