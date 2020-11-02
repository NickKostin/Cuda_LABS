# Cuda_LABS
Matrix multiplication using Python3.8, CUDA, PyCUDA
- Implemented the fastest way to multiply matrix ( using numpy.dot() );
- Compared a time running of two ways on gpu: 1) with numpy.dot(); 2) with C++ kernel in PyCUDA 
- Mean results:
  |  N  | CPU time, ms | GPU time, ms | Speedup|
  |:---:|:------------:|:------------:|:------:|
  | 128 |        0.112 |        0.416 |    0.27|
  | 256 |        0.604 |        0.607 |    1.00|
  | 512 |        4.458 |        2.122 |    2.10|
  |1024 |       33.154 |        9.679 |    3.43|
  |2048 |      256.891 |       61.235 |    4.20|
  
- Conclusion:
  - On small matrix sizes CPU calculate better than GPU, but, with the matrix size growing, GPU shows better results, in compare with CPU. 
  - Most reasonable for multiplying high-sizes matrix is to use GPU, instead of CPU.
