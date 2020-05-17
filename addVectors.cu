#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <helper_timer.h>
#include <time.h>

__global__ void add(int *a, int *b, int *c, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
    {
        c[tid] = a[tid] + b[tid];
    }
}

void cpu_add(int *a, int *b, int *c, int N)
{
    int i;
    for (i = 0; i < N; i++)
    {
        c[i] = a[i] + b[i];
    }
}

bool check_ans(int* c, int* dev_c, int N) {
    int i;
    bool are_right = true;
 
    for (i = 0; i < N; i++) {
        are_right = are_right && (c[i] == dev_c[i]);
    }
 
    return are_right;
}

int main(void)
{

    int N = atoi(argv[1], char *argv[]);
    int block_size = atoi(argv[2]);
    int grid_size = (int)ceil((float)N / block_size);

    int *a, *b, *c, *c_cpu;
    a = (int *)malloc(N * sizeof(int));
    b = (int *)malloc(N * sizeof(int));
    c = (int *)malloc(N * sizeof(int));
    c_cpu = (int *)malloc(N * sizeof(int));

    int *dev_a, *dev_b, *dev_c;
    cudaMalloc((void **)&dev_a, N * sizeof(int));
    cudaMalloc((void **)&dev_b, N * sizeof(int));
    cudaMalloc((void **)&dev_c, N * sizeof(int));
    for (int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = i * 2;
    }

    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);

    add<<<grid_size, block_size>>>(dev_a, dev_b, dev_c, N);

    cudaThreadSynchronize();
    sdkStopTimer(&timer);
    float gpu_time = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
    cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    clock_t t1, t2;
    t1 = clock();
    cpu_add(a, b, c_cpu, N);
    t2 = clock();
    float cpu_time = ((float)(t2 - t1) / CLOCKS_PER_SEC) * 1000;

    if (check_ans(c_cpu, c, N))
    {
        printf("%f,%f,%d,%d\n", gpu_time, cpu_time, block_size, grid_size);
    }
    else
    {
        printf("WRONG CALCULATIONS\n");
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    return 0;
}