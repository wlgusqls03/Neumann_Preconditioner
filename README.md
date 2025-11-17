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
	이 코드는 
	
		
		
