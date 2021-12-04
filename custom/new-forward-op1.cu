/* optimization 1:
 * Tiled shared memory convolution (2 points)
 */
#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"


#define BLOCK_WIDTH 16
/* Set a tile width other than block width for distinguishing 
 * between two concept, althought the value is the same.
 */
#define TILE_WIDTH 16

__global__ void conv_forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
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


    // Insert your GPU convolution kernel code here
   
    int W_num = ceil(W_out * 1.0 / (TILE_WIDTH * 1.0));
    int H_num = ceil(H_out * 1.0 / (TILE_WIDTH * 1.0));
    int x_out_width = TILE_WIDTH + K - 1;
    
    // FIXME: x_out_width cannot used as const!!!
    // extern __shared__ float shareX[x_out_width*x_out_width];
    // extern __shared__ float shareW[];

    // use one place instead
    extern __shared__ float share[];

    float* shareX = &share[0];
    float* shareW = &share[x_out_width*x_out_width];

    int b = blockIdx.x;
    int m = blockIdx.y;
    int h = (blockIdx.z / W_num) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.z % W_num) * TILE_WIDTH + threadIdx.x;
    int h_start=(blockIdx.z / W_num)*TILE_WIDTH; 
    int w_start=(blockIdx.z % W_num)*TILE_WIDTH; 


    // return if thread out of board
    if (b>=B || m>=M || h>=H_out || w>=W_out)return;
    
    int c,p,q,i,j;
    float res = 0.0f;
    for (c = 0; c < C; ++c) {
        // copy W to shared memory
        if ((threadIdx.y<K) && (threadIdx.x<K)){
            shareW[threadIdx.y*K+threadIdx.x]=k4d(m,c,threadIdx.y,threadIdx.x);
        }
        __syncthreads();

        // copy X to shared memory
        for (i=h; i<h_start+x_out_width; i+=TILE_WIDTH){
		    for (j=w; j<w_start+x_out_width; j+=TILE_WIDTH){
			    if (i<H && j<W){
				    shareX[(i-h_start)*x_out_width+(j-w_start)]=x4d(b,c,i,j);
			    }
			    else{
				    shareX[(i-h_start)*x_out_width+(j-w_start)]=0;
			    }
		    }
	    }
        __syncthreads();

        // do the conv
        for (p = 0; p < K; ++p) {
            for (q = 0; q < K; ++q) {
                res += x4d(b, c, h+p, w+q) * k4d(m, c, p, q);
                if(((threadIdx.y+p) < x_out_width) && ((threadIdx.x+q) < x_out_width)){
				    res+=shareX[(threadIdx.y+p)*x_out_width + (threadIdx.x+q)] * shareW[p*K+q];
			    }
            }
        }
        __syncthreads();
    
    }
    y4d(b, m, h, w) = res;


#undef y4d
#undef x4d
#undef k4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_y, const float *host_x, const float *host_k, float **device_y_ptr, float **device_x_ptr, float **device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int y_size = B*M*H_out*W_out*sizeof(float);     // y is the output
    int x_size = B*C*H*W*sizeof(float);             // x is the input
    int k_size = M*C*K*K*sizeof(float);             // k is the kernel, M kernels in total

    cudaMalloc((void**) device_y_ptr, y_size);
    cudaMalloc((void**) device_x_ptr, x_size);
    cudaMalloc((void**) device_k_ptr, k_size);

    cudaMemcpy(*device_y_ptr, host_y, y_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_x_ptr, host_x, x_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_k_ptr, host_k, k_size, cudaMemcpyHostToDevice);
    
    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

}


__host__ void GPUInterface::conv_forward_gpu(float *device_y, const float *device_x, const float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Set the kernel dimensions and call the kernel
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    dim3 dimGrid(B, M, ceil(W_out / (TILE_WIDTH * 1.0)) * ceil(H_out / (TILE_WIDTH * 1.0)));
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    conv_forward_kernel<<<dimGrid, dimBlock>>>(device_y, device_x, device_k, B, M, C, H, W, K);

    cudaDeviceSynchronize();

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_y, float *device_y, float *device_x, float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Copy the output back to host
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int y_size = B*M*H_out*W_out*sizeof(float);     // y is the output
    int x_size = B*C*H*W*sizeof(float);             // x is the input
    int k_size = M*C*K*K*sizeof(float);             // k is the kernel, M kernels in total
    
    cudaMemcpy(host_y, device_y, y_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_y);
    cudaFree(device_x);
    cudaFree(device_k);

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

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
