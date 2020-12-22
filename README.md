# Cuda_LABS

# mult - Matrix Multiplication
Matrix multiplication using Python3.8, CUDA, PyCUDA
- Implemented the fastest way to multiply matrix ( using numpy.dot() );
- Compared a time running of two ways on gpu: 1) with numpy.dot(); 2) with C++ kernel in PyCUDA 
- Results:

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

# harris - Harris Corner Detector
Harris Corner Detector using Python3.8, CUDA, PyCUDA

| Original Image: | Processed Image: |
|:---------------:|:----------------:|
| ![](https://github.com/NickKostin/Cuda_LABS/blob/main/Harris%20images/checkerboard.png?raw=true) | ![](https://github.com/NickKostin/Cuda_LABS/blob/main/Harris%20images/finalimage.png?raw=true) |

# salt and pepper - Salt and Pepper noise filtering
Salt and Pepper noise filtering using Python3.8, CUDA, PyCUDA

- To obtain an array characterizing the color of pixels, the `Pillow` library was used
- Each element of the output image was calculated (on the GPU) by a separate thread
- To speed up calculations on the GPU inside each image block, copying of elements from global memory to shared memory was implemented, which reduces the number of calls to global memory

- Results for filter 3х3 (median)

  |   File   | CPU time, ms | GPU time, ms | Speedup |
  |:--------:|:------------:|:------------:|:-------:|
  | 256.bmp  |      524.320 |        0.289 | 1813.15 |
  | 512.bmp  |     2273.571 |        0.492 | 4619.92 |
  | 1024.bmp |     9105.398 |        1.484 | 6135.44 |

- Results for filter 5х5 (median)

  |   File   | CPU time, ms | GPU time, ms | Speedup |
  |:--------:|:------------:|:------------:|:-------:|
  | 256.bmp  |     1210.220 |        0.925 | 1308.35 |
  | 512.bmp  |     5102.574 |        2.189 | 2331.01 |
  | 1024.bmp |    19996.483 |        6.873 | 2909.43 |
  
  | Original Image: | CPU Processed Image: | GPU Processed Image: |
  |:---------------:|:--------------------:|:--------------------:|
  | ![](https://github.com/NickKostin/Cuda_LABS/blob/main/SaltAndPepper%20images/original_img.bmp?raw=true) | ![](https://github.com/NickKostin/Cuda_LABS/blob/main/SaltAndPepper%20images/cpu_img.bmp?raw=true) | ![](https://github.com/NickKostin/Cuda_LABS/blob/main/SaltAndPepper%20images/gpu_img.bmp?raw=true) |


- Conclusion:
    -  Using the GPU in the framework of the median filtering task gives great increase in speed.
---
