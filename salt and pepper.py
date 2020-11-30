import numpy as np
from timeit import default_timer as timer
import pycuda.autoinit
from pycuda.driver import In, Out, Context
from pycuda.compiler import SourceModule
from PIL import Image

BLOCK_SIZE = 32
BLOCK = (BLOCK_SIZE, BLOCK_SIZE, 1)
FILTER_SIZE = 3
ARRAY_SIZE = FILTER_SIZE ** 2
OFFSET = FILTER_SIZE // 2
FILE_NAMES = ["256.bmp", "512.bmp", "1024.bmp"]

kernel = SourceModule(
    """
    __global__ void median_filter(unsigned char* pixels, unsigned char* filtered, int* size){
        const int blockSize = %(BLOCK_SIZE)s;
        const int arraySize = %(ARRAY_SIZE)s;
        const int filterSize = %(FILTER_SIZE)s;
        const int offset = %(OFFSET)s;
        int width = size[0];
        int bx = blockIdx.x,
            by = blockIdx.y,
            tx = threadIdx.x,
            ty = threadIdx.y;

        int j = bx * blockDim.x + tx; // column
	    int i = by * blockDim.y + ty; // row

	    int x, y, index;

        __shared__ int local[blockSize][blockSize];
        int arr[arraySize];

        local[ty][tx] = pixels[i * width + j];
        __syncthreads ();

        for (int k = 0; k < filterSize; k++){
            x = max(0, min(ty + k - offset, blockSize - 1));
            for (int l = 0; l < filterSize; l++){
                index = k * filterSize + l;
                y = max(0, min(tx + l - offset, blockSize - 1));
                arr[index] = local[x][y];
            }
        }
        __syncthreads ();

        for (int k = 0; k < arraySize; k++){
            for (int l = k + 1; l < arraySize; l++){
                if (arr[k] > arr[l]){
                    unsigned char temp = arr[k];
                    arr[k] = arr[l];
                    arr[l] = temp;
                }
            }
        }

        filtered[i * width + j] = arr[int(arraySize / 2)];
    }
    """ % {
        'BLOCK_SIZE': BLOCK_SIZE,
        'ARRAY_SIZE': ARRAY_SIZE,
        'OFFSET': OFFSET,
        'FILTER_SIZE': FILTER_SIZE
    }
)

median_filter = kernel.get_function("median_filter")


def open_image(filename: str):
    image = Image.open(filename)
    pix = image.load()

    width = image.size[0]
    height = image.size[1]

    pixels = np.zeros((width, height), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            pixels[i, j] = pix[j, i]

    return pixels, width, height


def cpu_filter(pixels, width, height):
    filtered = np.zeros_like(pixels)

    median = ARRAY_SIZE // 2
    for i in range(height):
        for j in range(width):
            arr = np.zeros(ARRAY_SIZE)
            for k in range(FILTER_SIZE):
                x = max(0, min(i + k - OFFSET, height - 1))
                index = k * FILTER_SIZE
                for l in range(FILTER_SIZE):
                    y = max(0, min(j + l - OFFSET, width - 1))
                    arr[index + l] = pixels[x, y]
            arr.sort()
            filtered[i, j] = arr[median]
    return filtered


def gpu_filter(pixels, width, height):
    size = np.array([width, height])
    filtered = np.zeros_like(pixels)
    grid_dim = (width // BLOCK_SIZE, height // BLOCK_SIZE)
    median_filter(In(pixels), Out(filtered), In(size), block=BLOCK, grid=grid_dim)
    Context.synchronize()
    return filtered


def save_image(filtered, filename):
    new_image = Image.fromarray(filtered.astype('uint8'), mode='L')
    new_image.save(filename, format="BMP")


def test_cpu(pixels, width, height, save):
    start = timer()
    filtered = cpu_filter(pixels, width, height)
    cpu_time = timer() - start
    if save:
        save_image(filtered, "cpu" + filename)

    return cpu_time * 1000


def test_gpu(pixels, width, height, save):
    start = timer()
    filtered = gpu_filter(pixels, width, height)
    gpu_time = timer() - start
    if save:
        save_image(filtered, "gpu" + filename)

    return gpu_time * 1000

CPU_TEST_ROUND = 5
GPU_TEST_ROUND = 50

print("|   File   | CPU time, ms | GPU time, ms | Speedup |")
for filename in FILE_NAMES:
    pixels, width, height = open_image(filename)
    test_cpu(pixels, width, height, True)
    test_gpu(pixels, width, height, True)

    cpu_time = 0
    gpu_time = 0

    for i in range(CPU_TEST_ROUND):
        cpu_time += test_cpu(pixels, width, height, False)

    for i in range(GPU_TEST_ROUND):
        gpu_time += test_gpu(pixels, width, height, False)

    cpu_time /= CPU_TEST_ROUND
    gpu_time /= GPU_TEST_ROUND

    print("| {:8s} | {:12.3f} | {:12.3f} | {:7.2f} |".format(filename, cpu_time, gpu_time, cpu_time / gpu_time))