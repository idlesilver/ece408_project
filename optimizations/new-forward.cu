#include <cmath>
#include <iostream>
#include <cuda_fp16.h>
#include "gpu-new-forward.h"


#define MODEL_NAME "best"
#define TILE_WIDTH 16
#define BLOCK_STREAM_SIZE 16
#define STREAM_NUM 10

cudaStream_t stream[STREAM_NUM];

__global__ void conv_forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
   
    int W_num = ceil(W_out / (TILE_WIDTH * 1.0));
    int H_num = ceil(H_out / (TILE_WIDTH * 1.0));
    
    int b = blockIdx.x, m = blockIdx.y;
    int w = (blockIdx.z % W_num) * TILE_WIDTH + threadIdx.x;
    int h = (blockIdx.z / W_num) * TILE_WIDTH + threadIdx.y;


    __half res = 0;
    if (w >= W_out || h >= H_out)return;
    // the same inner iteration from m1
    for (int c = 0; c < C; c++) {
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                res = __hadd(res, __hmul(__float2half(x4d(b,c,h+p,w+q)), __float2half(k4d(m,c,p,q))));
            }
        }
    }
    y4d(b, m, h ,w) = __half2float(res);


#undef y4d
#undef x4d
#undef k4d
}


/* cudaStream_t stream is not in param, so all is written in here */
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_y, const float *host_x, const float *host_k, float **device_y_ptr, float **device_x_ptr, float **device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K)
{

    // Allocate memory and copy over the relevant data structures to the GPU
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    std::cout<<"model name: "<< MODEL_NAME <<std::endl;

    int x_size = B*C*H*W*sizeof(float);             // x is the input
    int y_size = B*M*H_out*W_out*sizeof(float);     // y is the output
    int k_size = M*C*K*K*sizeof(float);             // k is the kernel, M kernels in total

    int x_stream_size = B*C*H*W/STREAM_NUM;             // x is the input
    int y_stream_size = B*M*H_out*W_out/STREAM_NUM;     // y is the output
    int k_stream_size = M*C*K*K;             // k is the kernel, M kernels in total

    cudaMalloc((void**) device_y_ptr, y_size);
    cudaMalloc((void**) device_x_ptr, x_size);
    cudaMalloc((void**) device_k_ptr, k_size);


    for (int i = 0; i < STREAM_NUM; ++i) {
        cudaStreamCreate(&stream[i]);
    }
    cudaMemcpyAsync(*device_k_ptr, host_k, k_stream_size * sizeof(float), cudaMemcpyHostToDevice, stream[0]);
    for (int i = 0; i < STREAM_NUM; ++i) {
        cudaMemcpyAsync((*device_x_ptr) + i*x_stream_size, host_x + i*x_stream_size, x_stream_size * sizeof(float), cudaMemcpyHostToDevice, stream[i]);
    }

    dim3 dimGrid( B/STREAM_NUM, M , ceil(H_out*1.0/BLOCK_STREAM_SIZE)*ceil(W_out*1.0/BLOCK_STREAM_SIZE) );
    dim3 dimBlock(BLOCK_STREAM_SIZE, BLOCK_STREAM_SIZE, 1);

    for (int i = 0; i < STREAM_NUM; ++i) {
        conv_forward_kernel<<<dimGrid, dimBlock, 0, stream[i]>>>( (*device_y_ptr) + i*y_stream_size, (*device_x_ptr) + i*x_stream_size, *device_k_ptr, B, M, C, H, W, K);
    }
    for (int i = 0; i < STREAM_NUM; ++i) {
        cudaMemcpyAsync((void*)(host_y + i*y_stream_size), (void*)((*device_y_ptr) + i*y_stream_size) , y_stream_size * sizeof(float), cudaMemcpyDeviceToHost, stream[i]);
    }
    cudaDeviceSynchronize();
    for (int i = 0; i < STREAM_NUM; ++i) {
      cudaStreamDestroy(stream[i]);
    }
    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.
    cudaFree(*device_x_ptr);
    cudaFree(*device_y_ptr);
    cudaFree(*device_k_ptr);

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
}

__host__ void GPUInterface::conv_forward_gpu(float *device_y, const float *device_x, const float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    ;
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_y, float *device_y, float *device_x, float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    ;
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
