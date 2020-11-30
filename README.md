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

# Salt And Pepper
- Лабораторная написана на языке Python 3 с использованием библиотеки `pycuda`
    - CUDA-ядро написано на С++ и обернуто в python-функцию
- Для получения массива чисел, характеризующих цвет пикселя использовалась библиотека `Pillow`  
- Каждый элемент выходного изображения вычислялся (на GPU) отдельной нитью (потоком)
- Для ускорения вычислений на GPU внутри каждого блока изображения было реализовано копирование элементов из глобальной памяти в разделяемую, что позволило уменьшить число обращений к глобальной памяти


|Исходное изображение 256x256| Обработанное на CPU | Обработанное на GPU |
|:--------------------------:|:-------------------:|:-------------------:|
| ![](SaltAndPepperImages/256.bmp) | ![](SaltAndPepperImages/cpu256.bmp) | ![](SaltAndPepperImages/gpu256.bmp) |

- Результаты для фильтра 3х3 (усредненные по нескольким запускам)

|   File   | CPU time, ms | GPU time, ms | Speedup |
|:--------:|:------------:|:------------:|:-------:|
| 256.bmp  |      556.000 |        0.351 | 1585.50 |
| 512.bmp  |     2334.820 |        0.509 | 4591.42 |
| 1024.bmp |     9366.697 |        1.511 | 6200.49 |

- Результаты для фильтра 5х5 (усредненные по нескольким запускам)

|   File   | CPU time, ms | GPU time, ms | Speedup |
|:--------:|:------------:|:------------:|:-------:|
| 256.bmp  |     1232.150 |        0.967 | 1274.56 |
| 512.bmp  |     5238.691 |        2.454 | 2135.13 |
| 1024.bmp |    20812.174 |        7.231 | 2878.32 |

- Выводы
    -  Использование GPU в рамках задачи медианной фильтрации дает колоссальный прирост в скорости
---
