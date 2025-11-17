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

Branch: multi_gpu

Commit: 7947754a5d1e6b5743f976c2fe46aba8b97c227a

##사용 예시

		# 1.  
		
		


- 결과 파일이 있는 폴더의 경로와 그래프를 생성할 코드를 각각 --root , --out 에 줘야한다.

  		ex) python line_plot.py --root ./results_diractory --out ./graph_diractory
