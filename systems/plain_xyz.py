# File: tools/rewrite_xyz_for_gospel.py
"""
Rewrite an extxyz to (a) plain XYZ and (b) strict extxyz with atomic_numbers.
Why: some codes ignore 'species' and need Z to compute electron count.
"""

from ase.io import read, write
from ase import Atoms
from pathlib import Path

def rewrite_for_compat(src: str, dst_plain: str, dst_extxyz: str) -> None:
    at: Atoms = read(src)         # extxyz OK
    at.center(vacuum=0.0)         # keep coords; do not add cell header
    # (a) Plain XYZ (no Lattice/pbc header)
    write(dst_plain, at, format="xyz")  # just symbols + positions

    # (b) ExtXYZ with atomic_numbers (forces Z field)
    # ASE writes numbers automatically if you pass 'numbers' array in arrays
    at2 = at.copy()
    at2.set_array("numbers", at2.get_atomic_numbers())  # ensure Z exists
    write(dst_extxyz, at2, format="extxyz")

if __name__ == "__main__":
    src = "C60_dimer.xyz"
    #Path("data/systems").mkdir(parents=True, exist_ok=True)
    rewrite_for_compat(src, "C60_dimer_plain.xyz",
                       "C60_dimer_numbers.xyz")
    print("Wrote: C60_dimer_plain.xyz and C60_dimer_numbers.xyz")


