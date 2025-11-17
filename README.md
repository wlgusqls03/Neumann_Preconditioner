# Neumann preconditioner를 이용한 전자 구조 계산 및 성능 비교
 이 레포는 Block davidson method 를 이용한 양자화학 계산에서 neumann expansion 을 이용한 preconditioner 를 이용하는 실험을 담고있닫.
	
	

##환경 설정

		# 가상환경 생성
		conda create -n neumann_precond python=3.12
  		conda activate neumann_precond

  		# torch 설치
		pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

		# 라이브러리 설치
		pip install numpy==1.26.4     # Torch compatibility (numpy version pinned)
		pip install ase               # Atomic Simulation Environment
		pip install gitpython         # Git interface for Python
		pip install "spglib>=1.16.1"  # Symmetry analysis library

		# Gospel 설치
		# Install GOSPEL (local development mode)
		git clone https://gitlab.com/jhwoo15/gospel.git
		cd gospel
		python setup.py develop

		# pylibxc 설치 (XC functional)
		git clone https://gitlab.com/libxc/libxc.git
		cd libxc
		git checkout 6.0.0  # Switch to 6.0.0 tag
		conda install -c conda-forge cmake  # Run this if cmake is not installed
		python setup.py develop  # or: pip install -e .

		# If pylibxc import fails:
		# You may need to add libxc.so* to your library path.
		# Example: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/libxc

		# Install neumann_precond (this repo)
		cd neumann_precond
		python setup.py develop


Git state 가 아래와 같은지 확인

* Branch: multi_gpu

* Commit: 7947754a5d1e6b5743f976c2fe46aba8b97c227a

##사용 예시

		# 1.  Neumann preconditioner fixed diagonalization 계산
		python test.py \
			--filepath data/systems/C60.xyz \
			--phase fixed  --spacing 0.2 --supercell 1 1 1  --pbc 0 0 0 \
			--temperature 0.00    --pp_type TM  \ 
			--virtual_factor 1.2 --diag_tol 1e-6 --diag_iter 11 \
			--precond neumann  --outerorder 10 \
			--retHistory History.neumann.pt

		# 2.  ISI preconditioner 계산
		python test.py \
			--filepath data/systems/C60.xyz \
			--phase scf  --spacing 0.2 --supercell 1 1 1  --pbc 0 0 0 \
			--temperature 0.00    --pp_type TM  \ 
			--virtual_factor 1.2 --diag_tol 1e-6 --diag_iter 11 \
			--precond shift-and-invert  --innerorder 0 --pcg_iter 5 \
			--retHistory History.neumann.pt

		# 3.  Merge preconditioner (Neumann + ISI) 계산
		python test.py \
			--filepath data/systems/MAPbI3.cif \
			--phase scf  --spacing 0.2 --supercell 2 2 2  --pbc 0 0 0 \
			--temperature 0.00    --pp_type TM  \ 
			--virtual_factor 1.2 --diag_tol 1e-4 --diag_iter 11 \
			--precond merge  --outerorder 4 --mergee_iter 5 --innerorder 0 --pcg_iter 5 \
			--retHistory History.neumann.pt

		# 4.  시스템 반복 계산
		nohub python repeat_test.py --mode scf-then-fixed > log 2>&1 &

## repeat_test.py 사용법

이 부분은 data/systems에 .cif, .sdf, .xyz 파일에 대하여 test.py 를 반복 계산을 수행하는 repeat_test.py 의 사용법을 정리하였다.


    1. RESULTS_ROOT 를 수정해 반복 계산 결과를 저장할 디랙토리 경로를 지정한다.
		ex) RESULTS_ROOT = Path("result_debug2")
		
	2. SELECTED_SYSTEMS: List[str] = {...} 부분에 systems에 존재하는 파일 중, 반복 계산을 수행할 물질을 선택한다. 

	3. OVERRIDE_BY_NAME: Dict[str, Dict] = {...} 부분은 2번에서 지정한 물질들의 
	spacing 및 supercell 에 대한 반복 계산을 위해 설정하는 부분이다. 이 부분을 따로 작성하지 않았지만,
	SELECTED_SYSTEMS: List[str] = {...} 구분에 들어있는 계산 시스템은 DEFAULT_SYSTEM_PARAMS = {...} 의 계산 설정을 받게 된다.
	ex)
		SELECTED_SYSTEMS: List[str] = [
    						"MAPbI3.cif",
    						"Maltododecaose.sdf",
    						"C60_tetramer.xyz",
							]
		DEFAULT_SYSTEM_PARAMS = dict(nbands=None, supercell=(1, 1, 1), pbc=(0, 0, 0), spacing=0.2)
		OVERRIDE_BY_NAME: Dict[str, Dict] = { 
		"MAPbI3.cif": {
        "supercell": [
				(2, 2, 2),
				(2, 2, 3),
				]
        "pbc": (1, 1, 1),
			},
		}		

		이렇게 설정하면 MAPbI3 는 supercell 에 대해 반복 계산하고 supercell (2, 2, 2), (2, 2, 3), 을 사용하나,
		설정 하지 않은 "Maltododecaose.sdf", "C60_tetramer.xyz" 는 기본 설정값인 supercell = (1, 1, 1) 을 받게 된다.
		
	4. USER_SWEEP 과 GLOBAL_FIXED 부분을 수정하여 반복 계산, 고정 계산에 쓸 물리량을 조절한다.
	USER_SWEEP 에 리스트에 원소 개수 별로 반복 계산을 수행한다. 그리고, GLOBAL_FIXED 는 고정되어 전달하는 인자값을 조절한다. 
	이때, 비어있는 리스트에 경우엔 하단의 class FixedConfig: 부분에 설정된 기본 값이 자동으로 들어가게 된다.
	ex) 
		USER_SWEEP = dict(
    		preconds=[], # 공석으로 neumann , ISI, merge preconditioner 계산을 모두 수행하도록 설정  
    		threads=[1],
    		outerorder=[2, "res"],  # neumann preconditioner 의 order
    		innerorder=[0],  # ISI preconditioner 의 innerprecond neumann 의 order
    		pcg_neumann=[5],  # ISI preconditioner 의 pcg iter
    		error_cutoff=[-0.4],  # Error cutoff --> neumann order - dynamic 일때
    		spacing=[0.2],  # 그리드 점 간격
    		nbands=[],  # 삭제 예정, virtual factor 로만 설정
    		virtual_factor=[1.2],  # virtual 의 비율 설정 --> 20 % 사용
			merge_iter=[3, 5, 7, 9],  # neu_ISI 에서 초반 neumann precond 횟수 설정
			)
		
		GLOBAL_FIXED = dict(
    		mode="scf-then-fixed",  # "scf" | "fixed" | "scf-then-fixed" 
    		phase="fixed",          # fixed 랑 scf 를 따로 돌릴때 설정, scf-then-fixed 이면 반영 X
    		temperature=0.00,       # 물질 온도 설정
    		scf_energy_tol=1e-6,    # SCF 에너지 tolerence
    		pp_type="TM",           # pseudopotential 종류
    		use_cuda=True,          # True = GPU 계산, False = CPU 계산
    		warmup_when_cuda=1,     # GPU 계산시 warmup
    		diag_iter=1000,         # fixed hamiltonian diagonalization
    		diag_tol=None,          # None ⇒ 미전달
    		diag_iter_scf=11,       # 1회 SCF 에 수행하는 대각화 횟수 --> diag_iter_scf - 1 이 preconditioning 횟수
    		diag_iter_fixed=1000,   # fixed hamiltonian diagonalization 에서 주는 반복 횟수
    		diag_tol_scf=None,      # SCF는 미전달(내부 디폴트) --> density_diff * 0.1
    		diag_tol_fixed=1e-6,    # fixed hamiltonian diagonalization에서 대각화 tolerence
    		nblock=2,
    		locking=False,          # locking, fill block 모두 False
    		fill_block=False,
    		runs_per_combo=3,       # 동일한 계산을 3번 반복 수행 --> maximum은 3까지
    		resume=True,
    		dry_run=False,
    		require_density_for_fixed=True,  # SCF 계산 수행 후 만들어진 전자밀도로 fixed hamiltonian diagonalization 수행
    		verbosity=1,
    		seed=0,                                                                                		)
		
	5. 결과 수행 이후 디랙토리에 로그가 나타나 계산 진척 정도와 저장되는 density files, stdout.log 의 경로를 확인할 수 있다.
	runs_per_combo = ... 옵션으로 인해 동일한 계산 설정을 여러번 수행하는데, 이 중 중앙값을 
	calculation_summary_fixed.txt ,calculation_summary_scf.txt 로 저장한다.

	6. 결과 폴더 내부에 density 폴더는 SCF 계산에서 수렴한 density file 을 저장하며,
	이를 이용해 fixed hamiltonian diagonalization 의 hamiltonian 을 로드한다.

	7. 결과 폴더 내부에 log 폴더는 SCF 계산에서 나온 log를 저장한다.

	8. 결과 폴더 내부에 history 폴더는 fixed hamiltonian diagonalization의 
	residual, eigenvalue history 파일과 출력 로그를 확인할 수 있다.
