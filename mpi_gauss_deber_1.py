from mpi4py import MPI
import time
import numpy as np

def lectura_datos(file1):
    with open(file1) as f: 
        result = f.readlines()
    result  = [x.strip() for x in result] 
    return result

def generate_random_diagonal_dominant_square_matrix( n, max_random_int,rank, size, comm):
    if rank == 0:
        A = max_random_int * np.random.rand(n*n) 
        for i in range(0,n):
            A[i*n+i]=np.sum(A[i*n:(i+1)*n-1])
    else:
        A = None
    A_local = np.empty((m / size, n), dtype='i')
    comm.Scatter(A, A_local, root=0)
    return A_local

# Programa principal
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print('Inicia proceso {} de {}\n'.format(rank, size))

total_num_tests=5
map=[]
b=[]
c=[]
x=[]
A=[]
A_temp=[]
b_temp=[]
result_tests=np.array([0]*total_num_tests)
max_random_int=10
file1='datosGauss.txt'
datos=lectura_datos(file1)
datos = filter(None, datos) 
for ns in datos:
  n=int(ns)
  map=np.array([0.0]*n)
  c=np.array([0.0]*n)
  x=np.array([0.0]*n)
  b=np.array([0.0]*n)
  A=np.array([0.0]*n*n)

  for num_test in range(0,total_num_tests):
    if rank==0:
      A = max_random_int * np.random.rand(n*n) 
      for i in range(0,n):
            A[i*n+i]=np.sum(A[i*n:(i+1)*n-1])

    comm.barrier()
    comm.Bcast(A, root=0)
    comm.Bcast(b, root=0)
   
    for i in range(n):
      map[i]= i % size
 
    start= time.time()
    for k in range(0,n):
      comm.Bcast(A[k*n + k:], root=map[k])
      comm.Bcast(b[k:], root=map[k])
      for i in range(k+1,n):
         if map[i]== rank:
            c[i]=A[i*n+k]/A[k*n+k]
            A[i*n+k:i*n+n-1]=A[i*n+k:i*n+n-1]-( c[i]*A[k*n+k:k*n+n-1])
            b[i]=b[i]-( c[i]*b[k] );


    if rank==0:
        x[n-1]=b[n-1]/A[(n-1)*n + n-1]
        for i in list(reversed(range(0,n-1))):
            sum=0
            sum=np.sum(A[i*n+i+1:i*n+n-1]*x[i+1:n-1])
            x[i]=(b[i]-sum)/A[i*n + i]

    end = time.time()
    tiempo = end - start

    tiempo = comm.allreduce(tiempo, op=MPI.MAX)
    result_tests[num_test]=tiempo


  if rank==0:
        print('tamano de la matrix  =: {:.0f}\n'.format(n))
        print('num procesos =: {:.0f}\n'.format(size))
        print('Tiempo transcurrido promedio: {:.8f}\n'.format(np.average(result_tests)))
        temp=A.reshape(n,n)
        print('suma componentes de residuos: {:.8f}\n'.format(np.linalg.norm(np.dot(temp, x)-b)))


  
