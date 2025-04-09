#include "stb_image.h" 
#include <cstdint>
#include <vector>
#include <iostream>
#include <cmath>

const int IMAGE_SIZE = 64;
const int CHANNELS = 3;
/*
unsigned char* stbi_load(
    char const* filename,
    int* x, int* y, int* channels_in_file,
    int desired_channels);
*/
[[nodiscard]] std::vector<uint8_t> loadImage(const std::string& filepath) {
    int width, height, channels; // get width, height, channels for our image
    uint8_t* image = stbi_load(filepath.c_str(), &width, &height, &channels, CHANNELS);
    if (!image) 
        throw std::runtime_error("OOPS! Failed to load image: " + filepath + " — Reason: " + stbi_failure_reason());
    if (width != IMAGE_SIZE || height != IMAGE_SIZE || channels != CHANNELS) {
        stbi_image_free(image);
        throw std::runtime_error("OOPS! Improper dimensions or corrupted image, expecting 64x64 RGB Images! Got: \n Width: "+ std::to_string(width) + "\nHeight: " + std::to_string(height) + "\nChannels: " + std::to_string(channels)); 
    }
    std::vector<uint8_t> flattened(image, image + width * height * channels);
    stbi_image_free(image);
    return flattened;
} 

[[nodiscard]] std::vector<float> normalizeImage(const std::vector<uint8_t> image) {
    std::vector<float> normalized(image.size());
    size_t i = 0;
    
    for (; i + 7 < image.size(); i += 8) {
        normalized[i] = image[i] / 255.0f;;
        normalized[i+1] = image[i+1] / 255.0f;;
        normalized[i+2] = image[i+2] / 255.0f;;
        normalized[i+3] = image[i+3] / 255.0f;;
        normalized[i+4] = image[i+4] / 255.0f;;
        normalized[i+5] = image[i+5] / 255.0f;;
        normalized[i+6] = image[i+6] / 255.0f;;
        normalized[i+7] = image[i+7] / 255.0f;;
    }
    for (; i < image.size(); i++) 
        normalized[i] = image[i] / 255.0f;
    
    return normalized;
}

int main() {
    auto raw = loadImage("peifeng.png");
    auto image = normalizeImage(raw);

    float *d_image_in;
    float *d_image_out;
    
    const int INPUT_H = 64;
    const int INPUT_W = 64;
    const int UPSCALE = 2;
    const int OUTPUT_H = INPUT_H * UPSCALE;
    const int OUTPUT_W = INPUT_W * UPSCALE;

    cudaMalloc(&d_image_in, image.size() * sizeof(float));
    cudaMalloc(&d_image_out, OUTPUT_H * OUTPUT_W * CHANNELS * sizeof(float));

   cudaError_t err = cudaMemcpy(d_image_in, image.data(), image.size() * sizeof(float), cudaMemcpyHostToDevice);
   if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed: " << cudaGetErrorString(err) << std::endl;
    }

    dim3 blockDim(16, 16);
    dim3 gridDim((IMAGE_SIZE_OUTPUT + 15) / 16, (IMAGE_SIZE_OUTPUT + 15) / 16);
    
    upsample2x_nearest<<<gridDim, blockDim>>>(d_image_in, d_image_out, INPUT_H, INPUT_W, CHANNELS);
    cudaDeviceSynchronize();  

    std::vector<float> output(OUTPUT_W * OUTPUT_H * CHANNELS);
    cudaMemcpy(output.data(), d_image_out, output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_image_in);
    cudaFree(d_image_out);
}