# Neumann preconditioner를 이용한 전자 구조 계산 

## run_files.py


- 이 코드는 cif 랑 sdf 파일이 존재하는 경로와 계산 material 을 넣으면 일치하는 파일을 찾아 시스템 구축 후 계산을 진행하거나, ase 에서 지원하는 구조에 경우 코드를 수정하여 구현 가능하다
  
- cif, sdf 파일의 이름은 {material}_sdf, {material}_cif 로 존재하여야 한다.
  
- 파일 내부에 존재하는 여러 인자들을 주면, 그에 맞는 시스템을 구축하여 계산을 수행한다.
  
- density file 인자는  scf 인 경우 density matrix 를 저장하는 이름을, fixed 인 경우 이미 만들어진 density matrix로  fixed Hamiltonian diagonalization 을 진행한다.
  
- upf_files 는 직접 pseudopotential 지정이 필요한 경우 제공한다.
  
- outerorderr 가 dynamic 인 경우는 Neumann order 가 각 반복마다 낮은 오차를 가지도록 변한다.
  
  		ex) python run_files.py --material CaTiO3 --dir ./ --precond Neumann --spacing 0.2 --nbands 540 --supercell 3 3 3 --outerorder dynamic --error_cutoff -0.3 --phase scf --pp_type TM --density_filename "density_diamond.pt"
        


## calculation_files.py
- 이 코드는 특정 시스템 별로 조건을 달리하여 계산하고 싶을 때 조건을 설정하고 반복적으로 run_files.py 를 수행한다.
  
- 코드 내부에 존재하는 조건을 수정하고 코드를 실행하면 총 동일한 계산 조건을 3번씩 반복하여 total diagonalization time 이 중앙값에 있는 결과를 설정한 결과 폴더에 저장하고, 여러 중요한 결과값을 summary 파일에 저장한다.
  
- 따로 args 로 주는 인자는 없으며, 코드 내부에서 조건을 수정한다.


	    ex) python calculation_files.py  처럼 args 는 없고 파일 내부에서 설정하도록 코드 작성


## line_plot.py
- 이 코드는 total diagonalization time 과 iteration count, Neumann preconditioner 의 outerorder 가 dynamic 인 경우에 매 iteration 마다 사용하는 order 에 대하여 그래프를 제작하는 코드이다.
  
- x 축은 preconditioner 의 종류와 Neumann preconditioner 계산에서 사용한 cutoff_error 를 나열한 precond type 을 의미한다.

- y 축은 total diagonalization time 과 iteration 의 의미를 갖는다.

- cutoff 에 따라 Neumann preconditioner 의 성능을 비교하고 싶을때 사용하는 코드이다.

- 결과 파일이 있는 폴더의 경로와 그래프를 생성할 코드를 각각 --root , --out 에 줘야한다.

  		ex) python line_plot.py --root ./results_diractory --out ./graph_diractory
