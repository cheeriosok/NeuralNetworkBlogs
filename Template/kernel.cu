#include <iostream>
#include <cuda_runtime.h>
#define CUDA_CHECK(call) \
    { \
        const cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", " \
                      << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    }

__constant__ float d_kernel[16]; 

#define BLOCK_SIZE 16

__global__ void conv2d(const float* input, float* output, int inputWidth, int inputHeight) {
    __shared__ float sharedInput[BLOCK_SIZE + 3][BLOCK_SIZE + 3];

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    if (x < inputWidth && y < inputHeight) {
        sharedInput[ty + 1][tx + 1] = input[y * inputWidth + x];

        if (tx == 0 && x > 0) sharedInput[ty + 1][0] = input[y * inputWidth + x - 1];
        if (tx == blockDim.x - 1 && x < inputWidth - 1) sharedInput[ty + 1][blockDim.x + 2] = input[y * inputWidth + x + 1];
        if (ty == 0 && y > 0) sharedInput[0][tx + 1] = input[(y - 1) * inputWidth + x];
        if (ty == blockDim.y - 1 && y < inputHeight - 1) sharedInput[blockDim.y + 2][tx + 1] = input[(y + 1) * inputWidth + x];

        if (tx == 0 && ty == 0 && x > 0 && y > 0) sharedInput[0][0] = input[(y - 1) * inputWidth + x - 1];
        if (tx == blockDim.x - 1 && ty == 0 && x < inputWidth - 1 && y > 0) sharedInput[0][blockDim.x + 2] = input[(y - 1) * inputWidth + x + 1];
        if (tx == 0 && ty == blockDim.y - 1 && x > 0 && y < inputHeight - 1) sharedInput[blockDim.y + 2][0] = input[(y + 1) * inputWidth + x - 1];
        if (tx == blockDim.x - 1 && ty == blockDim.y - 1 && x < inputWidth - 1 && y < inputHeight - 1) sharedInput[blockDim.y + 2][blockDim.x + 2] = input[(y + 1) * inputWidth + x + 1];
    }

    __syncthreads();

    if (x < inputWidth && y < inputHeight) {
        float sum = 0.0f;
        for (int ky = 0; ky < 4; ++ky) {
            for (int kx = 0; kx < 4; ++kx) {
                sum += sharedInput[ty + ky][tx + kx] * d_kernel[ky * 4 + kx];
            }
        }
        output[y * inputWidth + x] = sum;
    }
}

__global__ void relu(float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void maxpool(const float* input, float* output, int inputWidth, int inputHeight, int poolSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < inputWidth / poolSize && y < inputHeight / poolSize) {
        float maxVal = -FLT_MAX;
        for (int py = 0; py < poolSize; ++py) {
            for (int px = 0; px < poolSize; ++px) {
                int ix = x * poolSize + px;
                int iy = y * poolSize + py;
                if (ix < inputWidth && iy < inputHeight) {
                    maxVal = fmaxf(maxVal, input[iy * inputWidth + ix]);
                }
            }
        }
        output[y * (inputWidth / poolSize) + x] = maxVal;
    }
}

__global__ void fullyConnectedLayer(const float* input, const float* weights, const float* biases, float* output, int inputSize, int outputSize) {
    extern __shared__ float sharedInput[]; 
    int threadIdx = threadIdx.x;
    int blockIdx = blockIdx.x;
    int blockDim = blockDim.x;
    int idx = blockIdx * blockDim + threadIdx;

    if (threadIdx < inputSize) {
        sharedInput[threadIdx] = input[threadIdx];
    }
    __syncthreads();

    if (idx < outputSize) {
        float sum = 0.0f;
        for (int i = 0; i < inputSize; i++) {
            sum += sharedInput[i] * weights[i * outputSize + idx];
        }
        output[idx] = sum + biases[idx];
    }
}


__global__ void conv2dBackward(const float* input, const float* gradOutput, const float* kernel, float* gradKernel, 
                               float* gradInput, int inputWidth, int inputHeight) {
    
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < inputWidth && y < inputHeight) {
        float grad = 0.0f;
        for (int ky = 0; ky < 4; ky++) {
            for (int kx = 0; kx < 4; kx++) {
                int ix = x + kx - 2;
                int iy = y + ky - 2;
                if (ix >= 0 && ix < inputWidth && iy >= 0 && iy < inputHeight) {
                    grad += gradOutput[iy * inputWidth + ix] + kernel[ky * 4 + kx];
                    atomicAdd(&gradKernel[ky * 4 + kx], gradOutput[iy * inputWidth + ix] * input[y * inputWidth + x]);
                }
            }
        }
        gradInput[y * inputWidth + x] = grad; 
    }
}

__global__ void reluBackward(const float* input, float* gradInput, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        gradInput[idx] = input[idx] > 0 ? gradInput[idx] : 0;
    }
}

__global__ void maxpoolBackward(const float* input, const float* gradOutput, float* gradInput, int inputWidth, int inputHeight, int poolSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < inputWidth / poolSize && y < inputHeight / poolSize) {
        float maxVal = -FLT_MAX;
        int maxX = 0;
        int maxY = 0;
        for (int py = 0; py < poolSize; ++py) {
            for (int px = 0; px < poolSize; ++px) {
                int ix = x * poolSize + px;
                int iy = y * poolSize + py;
                if (ix < inputWidth && iy < inputHeight) {
                    if (input[iy * inputWidth + ix] > maxVal) {
                        maxVal = input[iy * inputWidth + ix];
                        maxX = ix;
                        maxY = iy;
                    }
                }
            }
        }
        atomicAdd(&gradInput[maxY * inputWidth + maxX], gradOutput[y * (inputWidth / poolSize) * x]);
    }
}

__global__ void fullyConnectedLayerBackward(const float* input, const float* weights, const float* gradOutput, 
                                            float* gradWeights, float* gradBiases, float gradInput, int inputSize,
                                            int outputSize) {
      
   extern __shared__ float sharedInput[];
   
   int idx = blockIdx.x * blockDim.x + threadIdx.x;

   if (threadId < inputSize) {
        sharedInput[threadId] = input[threadId];
   }

   __syncthreads();

   if (idx < outputSize) {  
        float grad = gradOutput[idx];
        atomicAdd(&gradBiases[idx], grad);
        for (int i = 0; i < inputSize; ++i) {
            atomicAdd(&gradWeights[i * outputSize + idx], grad * sharedInput[i]);
            atomicAdd(&gradInput[i], grad * weights[i * outputSize + idx]);
        }
   }
}

__global__ void updateWeightsKernel(float* weights, const float* gradWeights, int size, float learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= learning_rate * gradWeights[idx];
    }
}