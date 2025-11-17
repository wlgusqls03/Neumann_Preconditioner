# file: repeat_test.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Set

"""
'계산 전용' 스크립트.
- SCF를 모든 조합에 대해 항상 실행
- FIXED는 옵션에 따라 SCF 밀도 필요
- SCF/FIXED 로그/요약 기록
- diag_iter: SCF/FIXED 분리 전달 유지
- diag_tol: phase별 선택 전달 (None이면 옵션 자체 미전달 → 테스트 스크립트 내부 디폴트 사용)

사용 예시
nohub python repeat_test.py --mode scf-then-fixed > log 2>&1 &
"""

# ========== 파일 경로 및 시스템 파일경로 설정 ==========

RESULTS_ROOT = Path("result_scail_up")  # 계산 결과를 저장할 파일
DENSITY_KEY_MODE = os.environ.get("DENSITY_KEY_MODE", "geom4").lower()

SYSTEM_ROOTS = ("data/systems",)  # 시스템 파일이 들어있는 경로
INCLUDE_EXTS = ("*.cif", "*.sdf", "*.xyz")  # 물질들 확장자

# ========== 계산에 사용할 시스템 설정 ==========
# systems 내부에 있는 xyz, cif, sdf 파일 중  계산에 사용할 파일 이름 입력

SELECTED_SYSTEMS: List[str] = [
    # ------------결정--------------
    "CsPbI3.cif",
    "MAPbI3.cif",
    "Si_diamond.cif",
    "MgO.cif",
    # -----------생분자-------------
    #   "aspirin.sdf",             # 필요시 주석 제거 (시간 스케일 너무 작음)
    "beta_carotene.sdf",
    "B12.sdf",
    "Maltododecaose.sdf",
    # ---------일반 분자------------
    "water_cluster_64.xyz",  # 이전 계산 결과
    "water_cluster_80.xyz",
    "water_cluster_90.xyz",
    "water_cluster_108.xyz",
    "water_cluster_128.xyz",
    "C60_1.xyz",  # monomer
    "C60_2.xyz",  # dimeer
    "C60_3.xyz",  # trimer
    "C60_4.xyz",  # tetramer
]

DEFAULT_SYSTEM_PARAMS = dict(
    nbands=None, supercell=(1, 1, 1), pbc=None, spacing=0.2
)  # 기본 supercell 과 spacing 설정

OVERRIDE_BY_NAME: Dict[
    str, Dict
] = {  # default 설정(supercell = [1, 1, 1], pbc = (0, 0, 0), spacing  = 0.2)이 아니라면, 아래에서 물질별로 수정 가능
    # ------------결정--------------
    "CsPbI3.cif": {
        "supercell": [
            (4, 3, 3),  # supercell 을 리스트 형태로  주어  반복  계산  가능
            (3, 3, 3),
            (3, 3, 2),  # 이전 결과 superceell
        ],
        "pbc": (1, 1, 1),
    },  
    "MAPbI3.cif": {
        "supercell": [
            (3, 2, 2),
            (2, 2, 2),
            (2, 2, 1),  # 이전  결과 supercell
        ],
        "pbc": (1, 1, 1),
    },  
    "Si_diamond.cif": {
        "supercell": [
            (4, 4, 4), 
            (4, 4, 3),
            (4, 3, 3),
            (3, 3, 3),  # 이전 결과 superceell
        ],
        "pbc": (1, 1, 1),
    },
    "MgO.cif": {
        "supercell": [
            (4, 4, 4),
            (4, 4, 3),
            (4, 3, 3),  # 이전 결과 supercell
        ],
        "pbc": (1, 1, 1),
    },
    #     -----------생분자-------------  #  생분자, 일반분자는 따로 설정해주지 않아도  supercell = [1, 1, 1] 과  pbc = (0,  0, 0) 로 설정
    #  spacing 을 다르게 하거나, 특이한 경우에 아래와 같이 설정
    #    "aspirin.sdf": {
    #        "supercell": (1, 1, 1),
    #        "pbc": (0, 0, 0),
    #    }, # (MW = 180.16 g/mol)   특이사항 : 스케일 키우기 불가능 --> 제외
    #    "beta_carotene.sdf": {
    #        "supercell": (1, 1, 1),
    #        "pbc": (0, 0, 0),
    #    },  # (MW = 536.888 g/mol)
    #    "B12.sdf": {
    #        "supercell": (1, 1, 1),
    #        "pbc": (0, 0, 0),
    #    },  # (MW = 1335.4 g/mol) 특이사항 : Co와 같은 금속이 1개 있음
    #    "Maltododecaose.sdf": {
    #        "supercell": (1, 1, 1),
    #        "pbc": (0, 0, 0),
    #    },  # (MW = 1639.42 g/mol) 특이사항 : C, H, O 로만 이루어진 선형 올리고당
    #
    #    # ---------일반 분자------------
    #
    #    "water_cluster_64.xyz": {
    #        "supercell": (1, 1, 1),
    #        "pbc": (0, 0, 0),
    #    },  # 스케일 업 하여 계산 진행 물분자 64개 --> 128개(2배)
    #    "C60.xyz": {
    #        "supercell": (1, 1, 1),
    #        "pbc": (0, 0, 0),
    #    },  # tetramer 계산으로 제외 --> 필요하면 주석 제거
    #    "C60_2.xyz": {
    #        "supercell":(1, 1, 1),
    #        "pbc": (0, 0, 0),
    #     },  # tetramer 계산으로 제외 --> 필요하면 주석 제거
    #    "C60_4.xyz": {
    #        "supercell": (1, 1, 1),
    #        "pbc": (0, 0, 0),
    #     },  # 스케일 업 하여 계산 진행 C60 2개 --> 4개 (2차원으로 2개씩)
}

# ========== 반복 계산 값 추가(값을 리스트로 주면 여러번 반복 계산) ==========

USER_SWEEP = dict(
    preconds=[],  # 예: ["neu_ISI"], 공백이면 neu, neu_ISI, ISI 전부 실행
    threads=[1],
    outerorder=[0, 2, 4, 6, 8, 10, "res"],  # neumann preconditioner 의 order
    innerorder=[0],  # ISI preconditioner 의 innerprecond neumann 의 order
    pcg_neumann=[5],  # ISI preconditioner 의 pcg iter
    error_cutoff=[-0.4],  # Error cutoff --> neumann order - dynamic 일때
    spacing=[0.2],  # 그리드 점 간격
    nbands=[],  # 직접 계산할 밴드 설정
    virtual_factor=[1.2],  # virtual factor 설정 --> 1.2 사용
    merge_iter=[3, 5, 7, 9],  # neu_ISI 에서 초반 neumann precond 횟수 설정
)

# ========== 고정값 추가 ==========

GLOBAL_FIXED = dict(
    mode="scf-then-fixed",  # "scf" | "fixed" | "scf-then-fixed"  ---> scf 결과 이후 바로 fixed hamiltonian diagonalization 수행
    phase="fixed",  # fixed 랑 scf 를 따로 돌릴때 설정, scf-then-fixed 이면 반영 X
    temperature=0.00,  # 물질 온도 설정
    scf_energy_tol=1e-6,  # SCF 에너지 tolerence
    pp_type="TM",  # pseudopotential 종류
    use_cuda=False,  # GPU, CPU 여부
    warmup_when_cuda=1,  # warmup 시간 --> GPU 면 있어야한다.
    diag_iter=1000,  # fixed hamiltonian diagonalization
    diag_tol=None,  # None ⇒ 미전달
    diag_iter_scf=11,  # 1회 SCF 에 수행하는 대각화 횟수 --> diag_iter_scf - 1 이 preconditioning 횟수 (첫번째는 X)
    diag_iter_fixed=1000,  # fixed hamiltonian diagonalization 에서 주는 반복 횟수
    diag_tol_scf=None,  # SCF는 미전달(내부 디폴트) --> density_diff * 0.1
    diag_tol_fixed=1e-6,  # fixed hamiltonian diagonalization에서 대각화 tolerence
    nblock=2,
    locking=False,  # locking, fill block 모두 False
    fill_block=False,
    runs_per_combo=3,  # 동일한 계산을 3번 반복 수행 --> 중앙값을 summary 파일에 저장
    resume=True,
    dry_run=False,
    require_density_for_fixed=True,  # SCF 계산 수행 후 만들어진 전자밀도로 fixed hamiltonian diagonalization 수행
    verbosity=1,
    seed=0,
)

# ========== 타이머 요약 필드 ==========
CALC_SUMMARY_FIELDS = {  # 로그에서 특정 라벨을 찾기 위한 후보 문자열
    "davidson_total": {
        "candidates": [
            "davidson",
            "Davidson.diagonalize",
            "Davidson diagonalize",
            "Davidson",
        ],
        "attr": "total",
    },
    "diag_iter_count": {
        "candidates": [
            "Diag. Iter.",
            "Diag Iter.",
            "Diag Iter",
            "SCF iter.",
            "SCF iter",
            "SCF iteration",
        ],
        "attr": "count",
    },
    "preconditioning_total": {"candidates": ["Preconditioning"], "attr": "total"},
    "preconditioning_count": {"candidates": ["Preconditioning"], "attr": "count"},
}

VARY_TOKENS: Set[str] = set()

# ========== 유틸 ==========

_slug_re = re.compile(r"[^A-Za-z0-9_.-]+")


def slugify(s: str) -> str:  # 영문, 숫
    return _slug_re.sub("-", s).strip("-")


def pair_to_str(xyz: Tuple[int, int, int]) -> str:  # (a, b, c) --> axbxc
    return "x".join(map(str, xyz))


def ensure_dir(p: Path) -> Path:  # 설정한 디랙토리가 없으면 생성
    p.mkdir(parents=True, exist_ok=True)
    return p


def tail_print(path: Path, n: int = 40) -> None:  # 로그 파일 마지막 n 줄 출력, 에러 확인용
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()[-n:]
        print("[TAIL]", path)
        for ln in lines:
            sys.stdout.write(ln)
        if lines and not lines[-1].endswith("\n"):
            sys.stdout.write("\n")
    except Exception as e:
        print(f"[TAIL][ERR] {path}: {e}")


# ========== 시스템 스캔 ==========


def _as_optional_int_seq(v):  # 입력값을 튜플로 정규화
    if isinstance(v, (list, tuple)):
        return tuple(None if x is None else int(x) for x in v)
    return (None if v is None else int(v),)


def _as_tuple3_seq(
    v, default=(1, 1, 1)
):  # supercell, pbc 처럼 3개의 정수를 가진 tuple들의 tuple 로 변화 (if v = (a, b, c) --> ((a, b, c), )
    if v is None:
        v = default
    if (
        isinstance(v, (list, tuple))
        and len(v) == 3
        and not isinstance(v[0], (list, tuple))
    ):
        return (tuple(int(a) for a in v),)
    return tuple(tuple(int(a) for a in t) for t in v)


def _as_number_seq(v, default: float):  # 숫자 혹은 리스트를 tuple 로 정규화
    if v is None:
        v = default
    if isinstance(v, (list, tuple)):
        return tuple(float(x) for x in v)
    return (float(v),)


def _get_override_by_name(name: str) -> Dict:  # OVERRIDE_BY_NAME 에서 사용한 설정으로 덮어쓰기
    if name in OVERRIDE_BY_NAME:
        return OVERRIDE_BY_NAME[name]
    name_l = name.lower()
    for k, v in OVERRIDE_BY_NAME.items():
        if k.lower() == name_l:
            return v
    stem = Path(name).stem.lower()
    for k, v in OVERRIDE_BY_NAME.items():
        if Path(k).stem.lower() == stem:
            return v
    return {}


def _mk_system_entry(
    p: Path,
) -> Dict[
    str, Sequence
]:  # 하나의 시스템 파일에서 suffix 로 pbc 를 결정하고, 최종 cfg 생성 {"nbands" = ..., "supercell" = ...}
    suffix = p.suffix.lower()
    default_pbc = (
        (0, 0, 0) if suffix in (".sdf", ".xyz") else (1, 1, 1)
    )  # OVERRIDING 에 적지 않은 경우엔 pbc가 xyz,  sdf인 경우 (0, 0, 0), cif 파일이면 (1, 1, 1)로 설정
    cfg = {**DEFAULT_SYSTEM_PARAMS, **_get_override_by_name(p.name)}
    return {
        "nbands": _as_optional_int_seq(cfg.get("nbands")),
        "supercell": _as_tuple3_seq(
            cfg.get("supercell", (1, 1, 1))
        ),  # supercell 은 [1, 1, 1] 로   고정
        "pbc": _as_tuple3_seq(cfg.get("pbc", default_pbc), default_pbc),
        "spacing": _as_number_seq(cfg.get("spacing", 0.2), 0.2),  # spacing 은 0.2 로  고정
    }


def scan_systems() -> Dict[str, Dict[str, Sequence]]:  # 실제 디스크에서 시스템을 찾는 함수
    names = (
        set(SELECTED_SYSTEMS) if SELECTED_SYSTEMS else None
    )  # SYSTEM_ROOTS 아래에 INCLUDE_EXTS 패턴으로 탐색
    out: Dict[str, Dict[str, Sequence]] = {}
    for root in SYSTEM_ROOTS:
        rp = Path(root)
        if not rp.exists():
            continue
        for pat in INCLUDE_EXTS:
            for p in rp.rglob(pat):
                if names and p.name not in names:
                    continue
                out[str(p)] = _mk_system_entry(p)  # {"파일 경로" : {옵션들}} 딕셔너리로 리턴
    return out


# =============================================================
# 설정 컨테이너
@dataclass
class FixedConfig:
    python_exe: str = sys.executable  # 사용할 파이썬 실행기
    test_script: str = str(Path(__file__).with_name("test.py"))  # 실제 계산을 수행하는 스크립트

    DENSITY_ROOT: Path = RESULTS_ROOT / "density"
    HISTORY_ROOT: Path = RESULTS_ROOT / "history"
    LOG_ROOT: Path = RESULTS_ROOT / "logs"  # 결과 저장 루트들

    mode: str = GLOBAL_FIXED.get("mode", "scf-then-fixed")
    phase: str = GLOBAL_FIXED.get("phase", "fixed")
    temperature: float = GLOBAL_FIXED.get("temperature", 0.01)
    scf_energy_tol: float = GLOBAL_FIXED.get("scf_energy_tol", 1e-4)
    pp_type: str = GLOBAL_FIXED.get("pp_type", "TM")
    use_cuda: bool = GLOBAL_FIXED.get("use_cuda", True)
    warmup_when_cuda: int = GLOBAL_FIXED.get("warmup_when_cuda", 1)
    virtual_factor: float = GLOBAL_FIXED.get("virtual_factor", 1.2)  # 물리, 계산 옵션

    # diag_iter: 분리 유지
    diag_iter_scf: int = GLOBAL_FIXED.get(
        "diag_iter_scf", GLOBAL_FIXED.get("diag_iter", 1000)
    )
    diag_iter_fixed: int = GLOBAL_FIXED.get(
        "diag_iter_fixed", GLOBAL_FIXED.get("diag_iter", 1000)
    )

    # diag_tol: None이면 미전달 (phase별/글로벌 + 명시 여부)
    diag_tol_global: Optional[float] = GLOBAL_FIXED.get("diag_tol", None)
    diag_tol_scf: Optional[float] = GLOBAL_FIXED.get("diag_tol_scf", None)
    diag_tol_fixed: Optional[float] = GLOBAL_FIXED.get("diag_tol_fixed", None)
    diag_tol_global_is_set: bool = False
    diag_tol_scf_is_set: bool = False
    diag_tol_fixed_is_set: bool = False

    nblock: int = GLOBAL_FIXED.get("nblock", 2)
    locking: bool = GLOBAL_FIXED.get("locking", False)
    fill_block: bool = GLOBAL_FIXED.get("fill_block", False)
    verbosity: int = GLOBAL_FIXED.get("verbosity", 1)
    seed: int = GLOBAL_FIXED.get("seed", 0)

    merge_neu_steps: int = 5

    threads_list: Sequence[int] = (1,)
    preconds: Sequence[str] = ("neumann", "shift-and-invert", "neu_ISI")
    inner_for_isi: Sequence[str] = ("neumann",)
    outerorder_list: Sequence[str] = ("dynamic",)
    innerorder_list: Sequence[str] = ("0", "1", "2")
    pcg_iter_by_inner: Dict[str, Sequence[int]] = field(
        default_factory=lambda: {"neumann": (2,)}
    )
    error_cutoff_list: Sequence[float] = tuple(round(-0.1 * k, 1) for k in range(1, 8))

    virtual_factor_list: Sequence[float] = field(
        default_factory=lambda: (GLOBAL_FIXED.get("virtual_factor", 1.2),)
    )
    merge_neu_steps_list: Sequence[int] = field(default_factory=lambda: (5,))

    systems: Dict[str, Dict[str, Sequence]] = field(
        default_factory=lambda: {
            "data/systems/Si_diamond.cif": {
                "nbands": (None,),
                "supercell": ((1, 1, 1),),
                "pbc": ((1, 1, 1),),
                "spacing": (0.2,),
            },
        }
    )

    runs_per_combo: int = GLOBAL_FIXED.get("runs_per_combo", 3)
    resume: bool = GLOBAL_FIXED.get("resume", True)
    dry_run: bool = GLOBAL_FIXED.get("dry_run", False)
    require_density_for_fixed: bool = GLOBAL_FIXED.get(
        "require_density_for_fixed", True
    )


CFG = FixedConfig()  # 전역 설정 인스턴트 하나 만들어서 코드 전체에 대해서 사용


def apply_user_sweep_to_cfg():  # USER_SWEEP 에 사용자가 적어둔 값을 CFG 에 반영하는 함수
    if USER_SWEEP.get("preconds"):
        CFG.preconds = tuple(USER_SWEEP["preconds"])
    if USER_SWEEP.get("threads"):
        CFG.threads_list = tuple(int(x) for x in USER_SWEEP["threads"])
    if USER_SWEEP.get("outerorder"):
        CFG.outerorder_list = tuple(USER_SWEEP["outerorder"])
    if USER_SWEEP.get("innerorder"):
        CFG.innerorder_list = tuple(USER_SWEEP["innerorder"])
    if USER_SWEEP.get("pcg_neumann"):
        CFG.pcg_iter_by_inner["neumann"] = tuple(
            int(x) for x in USER_SWEEP["pcg_neumann"]
        )
    if USER_SWEEP.get("error_cutoff"):
        vals = USER_SWEEP["error_cutoff"]
        CFG.error_cutoff_list = tuple(
            float(x)
            for x in (
                vals
                if not isinstance(vals, str)
                else [float(v) for v in vals.split(",")]
            )
        )
    if USER_SWEEP.get("spacing"):
        vals = tuple(float(x) for x in USER_SWEEP["spacing"])
        for k in list(CFG.systems.keys()):
            CFG.systems[k]["spacing"] = vals
    if USER_SWEEP.get("nbands"):
        vals = tuple(
            None if (x is None or str(x).lower() == "none") else int(x)
            for x in USER_SWEEP["nbands"]
        )
        for k in list(CFG.systems.keys()):
            CFG.systems[k]["nbands"] = vals
    if USER_SWEEP.get("virtual_factor"):
        CFG.virtual_factor_list = tuple(float(x) for x in USER_SWEEP["virtual_factor"])
    if USER_SWEEP.get("merge_iter"):
        CFG.merge_neu_steps_list = tuple(int(x) for x in USER_SWEEP["merge_iter"])


# ========= 결과 로그에서 원하는 값 추출 ==========

_davidson_re = re.compile(r"^\s*davidson\s*\|\s*([0-9]*\.?[0-9]+)\s*\|", re.M)
_timer_row_re = re.compile(
    r"^(?P<label>[A-Za-z0-9 .()_@&\\/\\-]+?)\s*\|\s*(?P<total>[0-9]*\.?[0-9]+)\s*\|\s*(?P<count>\d+)\s*$",
    re.M,
)


def parse_davidson_seconds(
    log_path: Path,
) -> Optional[float]:  # davidson_re 로 한 줄 찾아 시간으로 기록
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    m = _davidson_re.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def parse_timer_metrics(
    log_path: Path,
) -> Dict[str, Dict[str, float]]:  # 타이머 표 전체를 읽어 딕셔너리로 값들 생성
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for m in _timer_row_re.finditer(text):
        label = m.group("label").strip()
        total = float(m.group("total"))
        count = int(m.group("count"))
        out[label] = {"total": total, "count": count}
    return out


def pick_metric(  # 첫번째로 존재하는 label 의 값을 꺼내줌
    metrics: Dict[str, Dict[str, float]], candidates: List[str], attr: str
) -> Optional[float]:
    for key in candidates:
        if key in metrics:
            return metrics[key].get(attr)
    return None


# =============================================================
# 조합 & density 키
@dataclass
class Combo:
    sys_path: str
    spacing: float
    nbands: Optional[int]
    supercell: Tuple[int, int, int]
    pbc: Tuple[int, int, int]
    threads: int
    precond: str
    inner: Optional[str]
    outerorder: Optional[str]
    innerorder: Optional[str]
    pcg_iter: Optional[int]
    error_cutoff: Optional[float]
    virtual_factor: Optional[float]
    merge_neu_steps: Optional[int]


def generate_combos(cfg: FixedConfig) -> Iterator[Combo]:  # 실제 계산 모든 조건을 만들어내는 코드
    for sys_path, opts in cfg.systems.items():
        for spacing, nbands, scell, pbc in itertools.product(
            opts.get("spacing", (0.2,)),
            opts.get("nbands", (None,)),
            opts.get("supercell", ((1, 1, 1),)),
            opts.get("pbc", ((1, 1, 1),)),
        ):
            for threads in cfg.threads_list:
                vf_list = cfg.virtual_factor_list if nbands is None else (None,)
                for vf in vf_list:
                    for precond in cfg.preconds:
                        if precond == "neumann":
                            for outer in cfg.outerorder_list:
                                for ec in cfg.error_cutoff_list:
                                    yield Combo(
                                        sys_path,
                                        spacing,
                                        nbands,
                                        scell,
                                        pbc,
                                        threads,
                                        precond,
                                        None,
                                        outer,
                                        None,
                                        None,
                                        ec,
                                        vf,
                                        None,
                                    )
                        elif precond == "shift-and-invert":
                            for order in cfg.innerorder_list:
                                for pcg in cfg.pcg_iter_by_inner.get("neumann", ()):
                                    yield Combo(
                                        sys_path,
                                        spacing,
                                        nbands,
                                        scell,
                                        pbc,
                                        threads,
                                        precond,
                                        "neumann",
                                        None,
                                        order,
                                        pcg,
                                        None,
                                        vf,
                                        None,
                                    )
                        elif precond == "neu_ISI":
                            for outer in cfg.outerorder_list:
                                for order in cfg.innerorder_list:
                                    for pcg in cfg.pcg_iter_by_inner.get("neumann", ()):
                                        for ec in cfg.error_cutoff_list:
                                            for miter in cfg.merge_neu_steps_list:
                                                yield Combo(
                                                    sys_path,
                                                    spacing,
                                                    nbands,
                                                    scell,
                                                    pbc,
                                                    threads,
                                                    precond,
                                                    "neumann",
                                                    outer,
                                                    order,
                                                    pcg,
                                                    ec,
                                                    vf,
                                                    int(miter),
                                                )
                        else:
                            raise ValueError(f"Unknown precond {precond}")


def build_density_subpath(  # SCF의 결과 density 파일을 저장할 경로를 생성
    *,
    sys_path: str,
    spacing: float,
    supercell: Tuple[int, int, int],
    pbc: Tuple[int, int, int],
    nbands: Optional[int],
    virtual_factor: Optional[float],
) -> Path:
    name = Path(sys_path).stem
    parts: List[str] = [slugify(name), "phase=scf", f"pp={CFG.pp_type}"]
    eff_nbands = (
        (nbands * supercell[0] * supercell[1] * supercell[2])
        if nbands is not None
        else None
    )
    parts.append(f"scell={pair_to_str(supercell)}")
    parts.append(f"nbands={eff_nbands if eff_nbands is not None else 'auto'}")
    if nbands is None:
        vf = virtual_factor if virtual_factor is not None else CFG.virtual_factor
        parts.append(f"vf={vf}")
    parts.append(f"spacing={spacing}")
    return Path(parts[0]).joinpath(*parts[1:])


@dataclass
class RunResult:  # 고정 계싼 한 번 수행한 여러 값
    run_idx: int
    ret_history: Path
    log_path: Path
    davidson_s: Optional[float]


@dataclass
class RunPaths:  # 하나에 Combo 에 대해 필요한 주요 디랙토리 및 파일 위치들
    base_subpath_scf: Path
    base_subpath_fixed: Path
    density_dir: Path
    history_scf_dir: Path
    history_dir: Path
    logs_scf_dir: Path
    logs_fixed_dir: Path

    def run_dir(self, run_idx: int) -> Path:
        return ensure_dir(self.logs_fixed_dir / f"run-{run_idx}")

    def run_history_path(self, run_idx: int) -> Path:
        return self.history_dir / f"run-{run_idx}" / "history.pt"

    def run_log_path(self, run_idx: int) -> Path:
        return self.logs_fixed_dir / f"run-{run_idx}" / "stdout.log"

    def hist_run_log_path(self, run_idx: int) -> Path:
        return self.history_dir / f"run-{run_idx}" / "stdout.log"

    def density_file(self) -> Path:
        return self.density_dir / "density.pt"

    def class_dir(self, label: str) -> Path:
        return ensure_dir(self.history_dir / label)  # 디랙토리, histroy 파일 생성


# ========== diag_tol 전달/미전달 해석==========


def _parse_optional_float_arg(s: Optional[str]) -> Tuple[bool, Optional[float]]:
    if s is None:
        return (False, None)  # 사용자 미지정
    if str(s).strip().lower() in ("none", ""):
        return (True, None)  # 명시적 미전달
    return (True, float(s))


def _resolve_diag_tol_for_phase(
    phase: str,
) -> Optional[float]:  # SCF 및 fixed 에서 사용할 diag_tol 을 각각 결정
    # why: 특정 phase에서 'none'을 명시하면 글로벌보다 우선해 완전 생략해야 함
    if phase == "scf":
        if CFG.diag_tol_scf_is_set:
            return CFG.diag_tol_scf
    else:
        if CFG.diag_tol_fixed_is_set:
            return CFG.diag_tol_fixed
    if CFG.diag_tol_global_is_set:
        return CFG.diag_tol_global
    return None


def build_combo_subpath(  # SCF, fixed 각각에 대해 이 조합이 어떤 파라미터로 돌았는지 전부 폴더 이름으로 생성
    *,
    sys_path: str,
    threads: int,
    precond: str,
    inner: Optional[str],
    outerorder: Optional[str],
    innerorder: Optional[str],
    pcg_iter: Optional[int],
    error_cutoff: Optional[float],
    nbands: Optional[int],
    supercell: Tuple[int, int, int],
    pbc: Tuple[int, int, int],
    spacing: float,
    phase_token: str,
    virtual_factor: Optional[float],
    merge_neu_steps: Optional[int],
) -> Path:
    name = Path(sys_path).stem
    parts: List[str] = [slugify(name)]

    def add(text: str, cond: bool = True):
        if cond:
            parts.append(text)

    eff_nbands = (
        (nbands * supercell[0] * supercell[1] * supercell[2])
        if nbands is not None
        else None
    )
    add(f"phase={phase_token}")
    add(f"pp={CFG.pp_type}")
    add(f"cuda={int(CFG.use_cuda)}")
    add(f"thr={threads}")
    add(f"prec={precond}")
    add(f"inner={inner}", inner is not None)
    add(f"outerorder={outerorder}", outerorder is not None)
    add(f"innerorder={innerorder}", innerorder is not None)
    add(f"pcg={pcg_iter}", pcg_iter is not None)
    add(
        f"ec={error_cutoff}",
        (precond in ("neumann", "neu_ISI") and error_cutoff is not None),
    )
    add(f"scell={pair_to_str(supercell)}")
    add(f"pbc={pair_to_str(pbc)}")
    add(f"nbands={eff_nbands if eff_nbands is not None else 'auto'}")
    add(f"spacing={spacing}")
    add(f"vf={virtual_factor}", (nbands is None and virtual_factor is not None))
    add(
        f"merge_iter={merge_neu_steps}",
        (precond == "neu_ISI" and merge_neu_steps is not None),
    )
    add(
        f"diag_iter={CFG.diag_iter_scf if phase_token == 'scf' else CFG.diag_iter_fixed}"
    )
    tol_for_phase = _resolve_diag_tol_for_phase(phase_token)
    add(f"diag_tol={tol_for_phase}", tol_for_phase is not None)
    add(f"nblock={CFG.nblock}")
    add(f"lock={int(CFG.locking)}")
    add(f"fill={int(CFG.fill_block)}")
    return Path(parts[0]).joinpath(*parts[1:])


def prepare_paths(
    cfg: FixedConfig, combo: Combo
) -> RunPaths:  # 하나의 Combo 를 받아 파일 경로 설정 및 디랙토리 설정
    dens_sub = build_density_subpath(
        sys_path=combo.sys_path,
        spacing=combo.spacing,
        supercell=combo.supercell,
        pbc=combo.pbc,
        nbands=combo.nbands,
        virtual_factor=combo.virtual_factor,
    )
    sub_scf = build_combo_subpath(
        sys_path=combo.sys_path,
        threads=combo.threads,
        precond=combo.precond,
        inner=combo.inner,
        outerorder=combo.outerorder,
        innerorder=combo.innerorder,
        pcg_iter=combo.pcg_iter,
        error_cutoff=combo.error_cutoff,
        nbands=combo.nbands,
        supercell=combo.supercell,
        pbc=combo.pbc,
        spacing=combo.spacing,
        phase_token="scf",
        virtual_factor=combo.virtual_factor,
        merge_neu_steps=combo.merge_neu_steps,
    )
    sub_fixed = build_combo_subpath(
        sys_path=combo.sys_path,
        threads=combo.threads,
        precond=combo.precond,
        inner=combo.inner,
        outerorder=combo.outerorder,
        innerorder=combo.innerorder,
        pcg_iter=combo.pcg_iter,
        error_cutoff=combo.error_cutoff,
        nbands=combo.nbands,
        supercell=combo.supercell,
        pbc=combo.pbc,
        spacing=combo.spacing,
        phase_token="fixed",
        virtual_factor=combo.virtual_factor,
        merge_neu_steps=combo.merge_neu_steps,
    )
    return RunPaths(
        base_subpath_scf=sub_scf,
        base_subpath_fixed=sub_fixed,
        density_dir=ensure_dir(cfg.DENSITY_ROOT / dens_sub),
        history_scf_dir=ensure_dir(cfg.HISTORY_ROOT / sub_scf),
        history_dir=ensure_dir(cfg.HISTORY_ROOT / sub_fixed),
        logs_scf_dir=ensure_dir(cfg.LOG_ROOT / sub_scf),
        logs_fixed_dir=ensure_dir(cfg.LOG_ROOT / sub_fixed),
    )


# ========== 실제 실행 커맨드 및 실행 ==========


def build_cmd(  # 실제 test.py 를 호출할 때  쓸 커맨드 라인 인자 리스트들 생성
    cfg: FixedConfig,
    combo: Combo,
    paths: RunPaths,
    run_idx: int,
    *,
    phase: str,
    include_ret_history: bool = True,
) -> List[str]:
    warmup = cfg.warmup_when_cuda if cfg.use_cuda else 0
    diag_iter_for_phase = cfg.diag_iter_scf if phase == "scf" else cfg.diag_iter_fixed
    diag_tol_for_phase = _resolve_diag_tol_for_phase(phase)

    cmd: List[str] = [
        cfg.python_exe,
        "-u",
        cfg.test_script,
        "--filepath",
        combo.sys_path,
        "--spacing",
        str(combo.spacing),
        "--supercell",
        *map(str, combo.supercell),
        "--pbc",
        *map(str, combo.pbc),
        "--phase",
        phase,
        "--pp_type",
        cfg.pp_type,
        "--threads",
        str(combo.threads),
        "--warmup",
        str(warmup),
        "--diag_iter",
        str(diag_iter_for_phase),
        "--nblock",
        str(cfg.nblock),
        "--verbosity",
        str(cfg.verbosity),
        "--seed",
        str(CFG.seed + run_idx),
        "--temperature",
        str(cfg.temperature),
        "--scf_energy_tol",
        str(cfg.scf_energy_tol),
        "--density_filename",
        str(paths.density_file()),
    ]
    if diag_tol_for_phase is not None:
        # why: None이면 내부 디폴트를 사용하게 옵션 자체를 생략
        cmd.extend(["--diag_tol", str(diag_tol_for_phase)])
    if cfg.use_cuda:
        cmd.append("--use_cuda")
    if combo.nbands is not None:
        eff = int(
            combo.nbands * combo.supercell[0] * combo.supercell[1] * combo.supercell[2]
        )
        cmd.extend(["--nbands", str(eff)])
    else:
        vf = (
            combo.virtual_factor
            if combo.virtual_factor is not None
            else cfg.virtual_factor
        )
        cmd.extend(["--virtual_factor", str(vf)])
    if cfg.locking:
        cmd.append("--locking")
    if cfg.fill_block:
        cmd.append("--fill_block")

    if combo.precond == "shift-and-invert":
        cmd.extend(["--precond", "shift-and-invert"])
        cmd.extend(["--inner", "neumann"])
        if combo.innerorder is not None:
            cmd.extend(["--innerorder", str(combo.innerorder)])
        if combo.pcg_iter is not None:
            cmd.extend(["--pcg_iter", str(combo.pcg_iter)])
    elif combo.precond == "neumann":
        cmd.extend(["--precond", "neumann"])
        if combo.outerorder is not None:
            cmd.extend(["--outerorder", str(combo.outerorder)])
        if combo.error_cutoff is not None:
            cmd.extend(["--error_cutoff", str(combo.error_cutoff)])
    elif combo.precond == "neu_ISI":
        cmd.extend(["--precond", "merge"])
        miter = (
            combo.merge_neu_steps
            if combo.merge_neu_steps is not None
            else CFG.merge_neu_steps
        )
        cmd.extend(["--merge_iter", str(miter)])
        if combo.outerorder is not None:
            cmd.extend(["--outerorder", str(combo.outerorder)])
        if combo.error_cutoff is not None:
            cmd.extend(["--error_cutoff", str(combo.error_cutoff)])
        cmd.extend(["--inner", "neumann"])
        if combo.innerorder is not None:
            cmd.extend(["--innerorder", str(combo.innerorder)])
        if combo.pcg_iter is not None:
            cmd.extend(["--pcg_iter", str(combo.pcg_iter)])
    else:
        raise ValueError(f"Unknown precond {combo.precond}")

    if phase == "fixed" and include_ret_history:
        cmd.extend(["--retHistory", str(paths.run_history_path(run_idx))])
    return [x for x in cmd if x]


def run_once(cmd: List[str], log_path: Path, threads: int) -> int:  # 실제 프로세스 실행하는 함수
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
            "NUMEXPR_NUM_THREADS": str(threads),
        }
    )
    ensure_dir(log_path.parent)
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        return proc.wait()


def classify_runs_by_time(
    times: List[Tuple[int, Optional[float]]]
) -> Dict[str, int]:  # 동일 계산을 3번 돌릴 때, 시간 기준 정렬(fast, median, slow 결정)
    def key(t: Tuple[int, Optional[float]]):
        _, sec = t
        return float("inf") if sec is None else sec

    ordered = sorted(times, key=key)
    idxs = [t[0] for t in ordered]
    if not idxs:
        return {"fast": 1, "median": 1, "slow": 1}
    if len(idxs) == 1:
        return {"fast": idxs[0], "median": idxs[0], "slow": idxs[0]}
    if len(idxs) == 2:
        return {"fast": idxs[0], "median": idxs[1], "slow": idxs[1]}
    return {"fast": idxs[0], "median": idxs[1], "slow": idxs[2]}


def write_setting_summary(  # 전체 설정 요약을 summary로 저장
    results_root: Path, combos: Sequence[Combo], systems: Dict[str, Dict[str, Sequence]]
):
    ensure_dir(results_root)
    payload = {
        "targets": list(systems.keys()),
        "vary_args": sorted(list(VARY_TOKENS)) if VARY_TOKENS else [],
        "fixed_args": {
            "phase_mode": CFG.mode,
            "pp_type": CFG.pp_type,
            "use_cuda": CFG.use_cuda,
            "virtual_factor_default": CFG.virtual_factor,
            "diag_iter_scf": CFG.diag_iter_scf,
            "diag_iter_fixed": CFG.diag_iter_fixed,
            "diag_tol_effective": {
                "global": CFG.diag_tol_global if CFG.diag_tol_global_is_set else None,
                "scf": _resolve_diag_tol_for_phase("scf"),
                "fixed": _resolve_diag_tol_for_phase("fixed"),
            },
            "nblock": CFG.nblock,
            "locking": CFG.locking,
            "fill_block": CFG.fill_block,
            "verbosity": CFG.verbosity,
        },
        "runs_per_combo": CFG.runs_per_combo,
    }
    (results_root / "setting_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _fmt_val(v):
    if isinstance(v, str):
        return f'"{v}"'
    return json.dumps(v, ensure_ascii=False)


def write_pretty_summary(dirpath: Path, row: Dict[str, object], filename: str) -> None:
    ensure_dir(dirpath)
    ordered_keys = [
        "material",
        "spacing",
        "preconditioner",
        "nbands_input",
        "nbands_eff",
        "virtual_factor",
        "supercell",
        "threads",
        "solver_type",
        "order",
        "innerorder",
        "innerprecond",
        "error_cutoff",
        "pcg_number",
        "merge_iter",
        "scf_iterations",
    ] + [
        k
        for k in row.keys()
        if k
        not in {
            "material",
            "spacing",
            "preconditioner",
            "nbands_input",
            "nbands_eff",
            "virtual_factor",
            "supercell",
            "threads",
            "solver_type",
            "order",
            "innerorder",
            "innerprecond",
            "error_cutoff",
            "pcg_number",
            "merge_iter",
            "scf_iterations",
        }
    ]
    line = ", ".join(
        [f"{k} = { _fmt_val(row.get(k)) }" for k in ordered_keys if k in row]
    )
    out_path = dirpath / filename
    if out_path.exists():
        out_path.write_text(
            out_path.read_text(encoding="utf-8") + line + "\n", encoding="utf-8"
        )
    else:
        out_path.write_text(line + "\n", encoding="utf-8")


def write_scf_only_summary(
    combo: Combo, scf_log: Path
) -> None:  # SCF 로그를 보고 summary 파일 생성
    metrics = parse_timer_metrics(scf_log)
    scf_iter_cnt = pick_metric(
        metrics, ["SCF iter.", "SCF iter", "SCF iteration"], "count"
    )
    dav_total = pick_metric(
        metrics,
        ["davidson", "Davidson.diagonalize", "Davidson diagonalize", "Davidson"],
        "total",
    )

    material = Path(combo.sys_path).stem
    base_row: Dict[str, object] = {
        "material": material,
        "spacing": combo.spacing,
        "preconditioner": combo.precond,
        "nbands_input": combo.nbands if combo.nbands is not None else None,
        "nbands_eff": (
            (
                combo.nbands
                * combo.supercell[0]
                * combo.supercell[1]
                * combo.supercell[2]
            )
            if combo.nbands is not None
            else "auto"
        ),
        "virtual_factor": (combo.virtual_factor if combo.nbands is None else None),
        "supercell": list(combo.supercell),
        "threads": combo.threads,
        "solver_type": "SCF-only",
    }

    if combo.precond == "neumann":
        base_row.update(
            {
                "order": combo.outerorder,
                "innerorder": None,
                "innerprecond": None,
                "error_cutoff": combo.error_cutoff,
                "pcg_number": None,
                "merge_iter": None,
            }
        )
    elif combo.precond == "shift-and-invert":
        base_row.update(
            {
                "order": None,
                "innerorder": combo.innerorder,
                "innerprecond": "neumann",
                "error_cutoff": None,
                "pcg_number": combo.pcg_iter,
                "merge_iter": None,
            }
        )
    elif combo.precond == "neu_ISI":
        base_row.update(
            {
                "order": combo.outerorder,
                "innerorder": combo.innerorder,
                "innerprecond": "neumann",
                "error_cutoff": combo.error_cutoff,
                "pcg_number": combo.pcg_iter,
                "merge_iter": combo.merge_neu_steps,
            }
        )
    else:
        base_row.update(
            {
                "order": None,
                "innerorder": None,
                "innerprecond": None,
                "error_cutoff": None,
                "pcg_number": None,
                "merge_iter": None,
            }
        )

    row = {
        **base_row,
        "scf_iterations": scf_iter_cnt,
        "davidson_total": dav_total,
        "scf_iter_count": scf_iter_cnt,
    }
    write_pretty_summary(
        RESULTS_ROOT, row, filename="calculation_summary_scf.txt"
    )  # 저장되는 결과 파일


def find_label_log(  # fast, slow, median 의 존재 여부 확인
    runpaths: RunPaths, label: str, idx: Optional[int]
) -> Optional[Path]:
    cand: List[Path] = [
        runpaths.history_dir / label / "stdout.log",
        runpaths.logs_fixed_dir / label / "stdout.log",
    ]
    if idx is not None:
        cand += [
            runpaths.logs_fixed_dir / f"run-{idx}" / "stdout.log",
            runpaths.history_dir / f"run-{idx}" / "stdout.log",
        ]
    for p in cand:
        if p.exists():
            return p
    return None


def write_fixed_summary(  # fixed hamiltonian diagonalization 에서 사용
    runpaths: RunPaths, combo: Combo, labels: Dict[str, int]
) -> None:
    median_idx = labels.get("median")
    log_path = find_label_log(runpaths, "median", median_idx)
    metrics: Dict[str, Dict[str, float]] = {}
    if log_path is not None:
        metrics = parse_timer_metrics(log_path)
    material = Path(combo.sys_path).stem
    base_row: Dict[str, object] = {
        "material": material,
        "spacing": combo.spacing,
        "preconditioner": combo.precond,
        "nbands_input": combo.nbands if combo.nbands is not None else None,
        "nbands_eff": (
            (
                combo.nbands
                * combo.supercell[0]
                * combo.supercell[1]
                * combo.supercell[2]
            )
            if combo.nbands is not None
            else "auto"
        ),
        "virtual_factor": (combo.virtual_factor if combo.nbands is None else None),
        "supercell": list(combo.supercell),
        "threads": combo.threads,
        "solver_type": (
            "ISI"
            if combo.precond == "shift-and-invert"
            else ("merge" if combo.precond == "neu_ISI" else combo.precond)
        ),
        "order": (
            combo.outerorder if combo.precond in ("neumann", "neu_ISI") else None
        ),
        "innerorder": (
            combo.innerorder
            if combo.precond in ("shift-and-invert", "neu_ISI")
            else None
        ),
        "innerprecond": (
            "neumann"
            if combo.precond == "neu_ISI"
            else (combo.inner if combo.precond == "shift-and-invert" else None)
        ),
        "error_cutoff": (
            combo.error_cutoff if combo.precond in ("neumann", "neu_ISI") else None
        ),
        "pcg_number": (
            combo.pcg_iter if combo.precond in ("shift-and-invert", "neu_ISI") else None
        ),
        "merge_iter": (combo.merge_neu_steps if combo.precond == "neu_ISI" else None),
    }
    row = dict(base_row)
    row["davidson_total"] = pick_metric(
        metrics, CALC_SUMMARY_FIELDS["davidson_total"]["candidates"], "total"
    )
    row["diag_iter_count"] = pick_metric(
        metrics, CALC_SUMMARY_FIELDS["diag_iter_count"]["candidates"], "count"
    )
    row["preconditioning_total"] = pick_metric(
        metrics, CALC_SUMMARY_FIELDS["preconditioning_total"]["candidates"], "total"
    )

    if row["davidson_total"] is None:
        try:
            ranking = json.loads(
                (runpaths.history_dir / "run_ranking.json").read_text(encoding="utf-8")
            )
            m_idx = labels.get("median")
            if isinstance(ranking, dict) and "order" in ranking and m_idx is not None:
                for idx, sec in ranking["order"]:
                    if idx == m_idx and isinstance(sec, (int, float)):
                        row["davidson_total"] = float(sec)
                        break
        except Exception:
            pass
    write_pretty_summary(RESULTS_ROOT, row, filename="calculation_summary_fixed.txt")


# ========== 메인 ==========
def main():
    global VARY_TOKENS
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["scf", "fixed", "scf-then-fixed"],
        default=GLOBAL_FIXED.get("mode", "scf-then-fixed"),
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--runs_per_combo", type=int, default=GLOBAL_FIXED.get("runs_per_combo", 3)
    )

    # diag_tol 계열(문자열로 받아 'none' 허용)
    parser.add_argument(
        "--diag_tol",
        type=str,
        default=None,
        help="(legacy) 공통 diag_tol; 'none'이면 옵션 미전달",
    )
    parser.add_argument(
        "--diag_tol_scf",
        type=str,
        default=None,
        help="SCF 전용 diag_tol; 'none'이면 옵션 미전달",
    )
    parser.add_argument(
        "--diag_tol_fixed",
        type=str,
        default=None,
        help="FIXED 전용 diag_tol; 'none'이면 옵션 미전달",
    )

    # diag_iter 분리 유지
    parser.add_argument(
        "--diag_iter", type=int, default=None, help="(legacy) 공통 diag_iter"
    )
    parser.add_argument(
        "--diag_iter_scf", type=int, default=None, help="SCF 전용 diag_iter"
    )
    parser.add_argument(
        "--diag_iter_fixed", type=int, default=None, help="FIXED 전용 diag_iter"
    )

    args, _ = parser.parse_known_args()
    CFG.mode = args.mode
    CFG.dry_run = args.dry_run or CFG.dry_run
    CFG.runs_per_combo = int(args.runs_per_combo)

    # diag_tol 적용: phase별 > 글로벌 > 미전달
    g_set, g_val = _parse_optional_float_arg(args.diag_tol)
    s_set, s_val = _parse_optional_float_arg(args.diag_tol_scf)
    f_set, f_val = _parse_optional_float_arg(args.diag_tol_fixed)
    if g_set:
        CFG.diag_tol_global_is_set = True
        CFG.diag_tol_global = g_val
    if s_set:
        CFG.diag_tol_scf_is_set = True
        CFG.diag_tol_scf = s_val
    if f_set:
        CFG.diag_tol_fixed_is_set = True
        CFG.diag_tol_fixed = f_val
    # 기본값이 있으면 플래그도 켠다(사용자가 CLI로 안 넘겨도 전달되게)
    if not g_set and CFG.diag_tol_global is not None:
        CFG.diag_tol_global_is_set = True
    if not s_set and CFG.diag_tol_scf is not None:
        CFG.diag_tol_scf_is_set = True
    if not f_set and CFG.diag_tol_fixed is not None:
        CFG.diag_tol_fixed_is_set = True

    # diag_iter 적용: specific > legacy > default
    if args.diag_iter is not None:
        CFG.diag_iter_scf = int(args.diag_iter)
        CFG.diag_iter_fixed = int(args.diag_iter)
    if args.diag_iter_scf is not None:
        CFG.diag_iter_scf = int(args.diag_iter_scf)
    if args.diag_iter_fixed is not None:
        CFG.diag_iter_fixed = int(args.diag_iter_fixed)

    # 시스템 로드
    systems = scan_systems()
    if not systems:
        example = Path(next(iter(SELECTED_SYSTEMS), "ATP.sdf"))
        systems[str(Path("data/systems") / example)] = _mk_system_entry(
            Path(str(Path("data/systems") / example))
        )
    # USER_SWEEP 일부 즉시 반영(파일 스캔 이후)
    if USER_SWEEP.get("spacing"):
        vals = tuple(float(x) for x in USER_SWEEP["spacing"])
        for k in list(systems.keys()):
            systems[k]["spacing"] = vals
    if USER_SWEEP.get("nbands"):
        vals = tuple(
            None if (x is None or str(x).lower() == "none") else int(x)
            for x in USER_SWEEP["nbands"]
        )
        for k in list(systems.keys()):
            systems[k]["nbands"] = vals
    CFG.systems = systems
    apply_user_sweep_to_cfg()

    combos = list(generate_combos(CFG))
    if not combos:
        print("No combos to run – check USER_SWEEP/preconds.")
        return

    # vary 토큰(정보성)
    keys = {
        "phase",
        "pp",
        "cuda",
        "thr",
        "prec",
        "inner",
        "outerorder",
        "innerorder",
        "pcg",
        "ec",
        "scell",
        "pbc",
        "nbands",
        "spacing",
        "vf",
        "merge_iter",
        "diag_iter",
        "diag_tol_scf",
        "diag_tol_fixed",
        "nblock",
        "lock",
        "fill",
    }
    values: Dict[str, set] = {k: set() for k in keys}
    for c in combos:

        def put(k: str, v):
            if v is None:
                return
            values[k].add(v)

        put("phase", CFG.mode)
        put("pp", CFG.pp_type)
        put("cuda", int(CFG.use_cuda))
        put("thr", c.threads)
        put("prec", c.precond)
        put("inner", c.inner)
        put("outerorder", c.outerorder)
        put("innerorder", c.innerorder)
        put("pcg", c.pcg_iter)
        put("ec", c.error_cutoff if c.precond in ("neumann", "neu_ISI") else None)
        put("scell", c.supercell)
        put("pbc", c.pbc)
        put("nbands", c.nbands if c.nbands is not None else "auto")
        put("spacing", c.spacing)
        put("vf", c.virtual_factor if c.nbands is None else None)
        put("merge_iter", c.merge_neu_steps if c.precond == "neu_ISI" else None)
        put("diag_iter", CFG.diag_iter_scf)  # 정보성
        put("diag_tol_scf", _resolve_diag_tol_for_phase("scf"))
        put("diag_tol_fixed", _resolve_diag_tol_for_phase("fixed"))
        put("nblock", CFG.nblock)
        put("lock", int(CFG.locking))
        put("fill", int(CFG.fill_block))
    VARY_TOKENS = {k for k, s in values.items() if len(s) > 1}
    write_setting_summary(RESULTS_ROOT, combos, systems)

    # === 실행 ===
    for c_idx, combo in enumerate(combos, 1):
        paths = prepare_paths(CFG, combo)
        dens_file = paths.density_file()

        # --- SCF: 항상 실행 ---
        scf_rc = 0
        if CFG.mode in ("scf", "scf-then-fixed"):
            scf_cmd = build_cmd(
                CFG, combo, paths, run_idx=0, phase="scf", include_ret_history=False
            )
            scf_log = paths.logs_scf_dir / "scf.log"
            print(f"[SCF] ({c_idx}/{len(combos)}) combo={paths.base_subpath_scf}")
            print("CMD:", " ".join(scf_cmd))
            if not CFG.dry_run:
                scf_rc = run_once(scf_cmd, scf_log, combo.threads)
                if scf_rc != 0:
                    print(f"[ERR][SCF] Return code {scf_rc} — see: {scf_log}")
                    tail_print(scf_log, 60)
            else:
                ensure_dir(scf_log.parent)
                scf_log.write_text("[dry_run] scf\n", encoding="utf-8")
            write_scf_only_summary(combo, scf_log)

        if CFG.mode == "scf":
            continue

        # FIXED 는 필요 시 SCF 산 밀도를 사용
        if CFG.require_density_for_fixed and (scf_rc != 0 or not dens_file.exists()):
            print(
                f"[SKIP][FIXED] Missing/failed SCF density for combo={paths.base_subpath_fixed} — skip fixed runs."
            )
            continue

        # --- FIXED (multi-run) ---
        results: List[RunResult] = []
        for run_idx in range(1, CFG.runs_per_combo + 1):
            cmd = build_cmd(
                CFG, combo, paths, run_idx, phase="fixed", include_ret_history=True
            )
            print(
                f"[RUN] ({c_idx}/{len(combos)}) combo={paths.base_subpath_fixed} run={run_idx}"
            )
            print("CMD:", " ".join(cmd))
            if CFG.dry_run:
                results.append(
                    RunResult(
                        run_idx,
                        paths.run_history_path(run_idx),
                        paths.run_log_path(run_idx),
                        None,
                    )
                )
                continue
            ensure_dir(paths.run_history_path(run_idx).parent)
            rc = run_once(cmd, paths.run_log_path(run_idx), combo.threads)
            if rc != 0:
                print(f"[ERR] Return code {rc} — see: {paths.run_log_path(run_idx)}")
                tail_print(paths.run_log_path(run_idx), 60)
            dtime = parse_davidson_seconds(paths.run_log_path(run_idx))
            print(f"  → davidson(s) = {dtime}")
            results.append(
                RunResult(
                    run_idx,
                    paths.run_history_path(run_idx),
                    paths.run_log_path(run_idx),
                    dtime,
                )
            )

        labels = classify_runs_by_time([(r.run_idx, r.davidson_s) for r in results])
        order = sorted(
            [(r.run_idx, r.davidson_s) for r in results],
            key=lambda t: float("inf") if t[1] is None else t[1],
        )
        ranking = {"order": order, "labels": labels}
        (paths.history_dir / "run_ranking.json").write_text(
            json.dumps(ranking, indent=2), encoding="utf-8"
        )
        with open(paths.history_dir / "run_ranking.txt", "w", encoding="utf-8") as f:
            for rank, (idx, sec) in enumerate(order, 1):
                tag = [k for k, v in labels.items() if v == idx]
                f.write(
                    f"{rank}) run-{idx}: davidson={sec}  label={tag[0] if tag else '-'}\n"
                )

        # 라벨별 이동 및 정리
        for label, idx in labels.items():
            dst_dir = paths.class_dir(label)
            src_h = paths.run_history_path(idx)
            src_l = paths.run_log_path(idx)
            if src_h.exists():
                ensure_dir(dst_dir)
                shutil.move(str(src_h), str(dst_dir / "history.pt"))
            else:
                print(f"[WARN] history not found for label={label}: {src_h}")
            if src_l.exists():
                ensure_dir(dst_dir)
                shutil.move(str(src_l), str(dst_dir / "stdout.log"))
            else:
                print(f"[WARN] log not found for label={label}: {src_l}")
        for d in paths.logs_fixed_dir.glob("run-*"):
            shutil.rmtree(d, ignore_errors=True)
        for d in paths.history_dir.glob("run-*"):
            shutil.rmtree(d, ignore_errors=True)

        (paths.history_dir / "summary.json").write_text(
            json.dumps(
                {
                    "path": str(paths.history_dir),
                    "vary_tokens": sorted(list(VARY_TOKENS)) if VARY_TOKENS else [],
                    "runs": [
                        {"run": r.run_idx, "davidson_seconds": r.davidson_s}
                        for r in results
                    ],
                    "labels": labels,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        write_fixed_summary(paths, combo, labels)

    print("\nAll done (compute-only).")


if __name__ == "__main__":
    main()
