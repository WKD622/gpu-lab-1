build:
	mpicc -o test_c test.c 	
	mpiexec -machinefile ./allnodes	-np 2 ./test_c 3
