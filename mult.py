import numpy as np
from cudapy import compiler, gpuarray, tools
import pycuda.driver as drv
import pycuda.autoinit

MATRIX_SIZES = [128, 256, 512, 1024,2048]
BLOCK_SIZE = 16
kernel = """
__global__ void matrix_mult(int matrixsize,float *a, float *b, float *c)
{
    // 2D Thread ID 
    int tx = blockDim.x*blockIdx.x + threadIdx.x; // Compute column index
    int ty = blockDim.y*blockIdx.y + threadIdx.y; // Compute row index
    // Each thread loads one row of M and one column of N, 
    //   to produce one element of P.
    if((ty <matrixsize) && (tx < matrixsize))
    {
    // P-value is used to store the element of the matrix
    // that is computed by the thread
    float Pvalue = 0;
    for(int k=0; k<matrixsize; ++k)
    {
    float Aelement = a[ty*matrixsize +k];
    float Belement = b[k*matrixsize +tx];
    Pvalue += Aelement * Belement;
    }
    c[ty * matrixsize + tx] = Pvalue;
    }
}
"""

mod = compiler.SourceModule(kernel)
matrix_mult = mod.get_function("matrix_mult")


def mult_cpu(a, b):
  return a.dot(b)


def mult_gpu(a, b, MATRIX_SIZE):
  # transfer host (CPU) memory to device (GPU) memory
  a_gpu = gpuarray.to_gpu(a)
  b_gpu = gpuarray.to_gpu(b)

  # create empty gpu array for the result (C = A * B)
  c_gpu = gpuarray.empty((MATRIX_SIZE, MATRIX_SIZE), np.float32)
  grid=(MATRIX_SIZE//BLOCK_SIZE,MATRIX_SIZE//BLOCK_SIZE,1)

  # call the kernel on the card
  matrix_mult(np.uint32(MATRIX_SIZE),
              # inputs
              a_gpu, b_gpu,
              # output
              c_gpu,
              grid=grid,
              block = (BLOCK_SIZE, BLOCK_SIZE, 1),
              )
  return c_gpu


def calculate(a, b, MATRIX_SIZE):
    start_cpu = timer()
    c_cpu = mult_cpu(a, b)
    cpu_multiply_time = timer() - start_cpu

    start_gpu = timer()
    c_gpu = mult_gpu(a, b, MATRIX_SIZE)
    gpu_multiply_time = timer() - start_gpu

    return cpu_multiply_time * 1000, gpu_multiply_time * 1000, np.allclose(c_cpu, c_gpu.get())


count = 15

print(" N \t CPU time \t GPU time \t Speedup")

for size in MATRIX_SIZES:
  cpu_time = 0
  gpu_time = 0

  for i in range(count):
    A = np.random.rand(size, size).astype(np.float32)
    B = np.random.rand(size, size).astype(np.float32)

    current_cpu_time, current_gpu_time, err = calculate(A, B, size)
    cpu_time += current_cpu_time
    gpu_time += current_gpu_time

    if err is False:
      print("N = {:d}: results not equals".format(size))

  print("{:4d} \t {:7.3f} \t {:7.3f} \t {:7.2f}".format(size, cpu_time / count, gpu_time / count, cpu_time / gpu_time))


'''

N 	 CPU time 	 GPU time 	 Speedup
 128 	   0.112 	   0.416 	    0.27
 256 	   0.604 	   0.607 	    1.00
 512 	   4.458 	   2.122 	    2.10
1024 	  33.154 	   9.679 	    3.43
2048 	 256.891 	  61.235 	    4.20

'''