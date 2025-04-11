#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <filesystem>
#include <fstream>
#include <sstream>
#include "stb_image.h"
#include "data_loader.cpp"
#include "kernel.cu"

std::vector<float> normalizeImage(const std::vector<unsigned char>& image) {
    std::vector<float> normalized_image(image.size());
    for (size_t i = 0; i < image.size(); ++i) {
        normalized_image[i] = image[i] / 255.0f;
    }
    return normalized_image;
}

class CNN {
public:
    float conv1_kernel[32 * 3 * 3]; 
    float conv2_kernel[64 * 3 * 3];
    float conv3_kernel[128 * 3 * 3]; 
    float fc1_weights[8192 * 1024]; 
    float fc1_biases[1024]; 
    float fc2_weights[1024 * 200];
    float fc2_biases[200]; 

    float *d_input, *d_conv1_output, *d_conv2_output, *d_conv3_output
    float *d_fc1_output, *d_fc2_output;

    float *d_conv1_kernel, *d_conv2_kernel, *d_conv3_kernel;
    float *d_fc1_weights, *d_fc1_biases, *d_fc2_weights, *d_fc2_biases;
    
    float *d_conv1_output_grad, *d_conv2_output_grad, *d_conv3_output_grad;
    float *d_fc1_output_grad, *d_fc2_output_grad;

    

   CNN() {
        cudaMalloc(&d_input, 64 * 64 * 3 * sizeof(float));
        cudaMalloc(&d_conv1_output, 64 * 64 * 32 * sizeof(float));
        cudaMalloc(&d_conv2_output, 32 * 32 * 64 * sizeof(float));
        cudaMalloc(&d_conv3_output, 16 * 16 * 128 * sizeof(float));
        cudaMalloc(&d_fc1_output, 1024 * sizeof(float));
        cudaMalloc(&d_fc2_output, 200 * sizeof(float));

        cudaMalloc(&d_conv1_kernel, 32 * 3 * 3 * sizeof(float));
        cudaMalloc(&d_conv2_kernel, 64 * 3 * 3 * sizeof(float));
        cudaMalloc(&d_conv3_kernel, 128 * 3 * 3 * sizeof(float));
        cudaMalloc(&d_fc1_weights, 8192 * 1024 * sizeof(float));
        cudaMalloc(&d_fc1_biases, 1024 * sizeof(float));
        cudaMalloc(&d_fc2_weights, 1024 * 200 * sizeof(float));
        cudaMalloc(&d_fc2_biases, 200 * sizeof(float));

        cudaMalloc(&d_conv1_output_grad, 64 * 64 * 32 * sizeof(float));
        cudaMalloc(&d_conv2_output_grad, 32 * 32 * 64 * sizeof(float));
        cudaMalloc(&d_conv3_output_grad, 16 * 16 * 128 * sizeof(float));
        cudaMalloc(&d_fc1_output_grad, 1024 * sizeof(float));
        cudaMalloc(&d_fc2_output_grad, 200 * sizeof(float));

        cudaMalloc(&d_conv1_kernel_grad, 32 * 3 * 3 * sizeof(float));
        cudaMalloc(&d_conv2_kernel_grad, 64 * 3 * 3 * sizeof(float));
        cudaMalloc(&d_conv3_kernel_grad, 128 * 3 * 3 * sizeof(float));
        cudaMalloc(&d_fc1_weights_grad, 8192 * 1024 * sizeof(float));
        cudaMalloc(&d_fc1_biases_grad, 1024 * sizeof(float));
        cudaMalloc(&d_fc2_weights_grad, 1024 * 200 * sizeof(float));
        cudaMalloc(&d_fc2_biases_grad, 200 * sizeof(float));

        for (int i = 0; i < 32 * 3 * 3; ++i) conv1_kernel[i] = static_cast<float>(rand()) / RAND_MAX;
        for (int i = 0; i < 64 * 3 * 3; ++i) conv2_kernel[i] = static_cast<float>(rand()) / RAND_MAX;
        for (int i = 0; i < 128 * 3 * 3; ++i) conv3_kernel[i] = static_cast<float>(rand()) / RAND_MAX;
        for (int i = 0; i < 8192 * 1024; ++i) fc1_weights[i] = static_cast<float>(rand()) / RAND_MAX;
        for (int i = 0; i < 1024; ++i) fc1_biases[i] = static_cast<float>(rand()) / RAND_MAX;
        for (int i = 0; i < 1024 * 200; ++i) fc2_weights[i] = static_cast<float>(rand()) / RAND_MAX;
        for (int i = 0; i < 200; ++i) fc2_biases[i] = static_cast<float>(rand()) / RAND_MAX;

        cudaMemcpy(d_conv1_kernel, conv1_kernel, 32 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_conv2_kernel, conv2_kernel, 64 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_conv3_kernel, conv3_kernel, 128 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_fc1_weights, fc1_weights, 8192 * 1024 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_fc1_biases, fc1_biases, 1024 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_fc2_weights, fc2_weights, 1024 * 200 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_fc2_biases, fc2_biases, 200 * sizeof(float), cudaMemcpyHostToDevice);
    }

    void forward(const std::vector<float>& input, std::vector<float>& output) {
        cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
        
        // convolutional Layer 1
        dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);\
        dim3 gridDim((64 + BLOCK_SIZE - 1) / BLOCK_SIZE, (64 + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        conv2d<<<gridDim, blockDim>>>(d_input, d_conv1_output, 64, 64);
        relu<<<(64 * 64 * 32 + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_conv1_output, 64 * 64 * 32);
        maxpool<<<gridDim, blockDim>>>(d_conv1_output, d_conv2_output, 64, 64, 2);

        // convolutional Layer 2
        gridDim = dim3((32 + BLOCK_SIZE - 1) / BLOCK_SIZE, (32 + BLOCK_SIZE - 1) / BLOCK_SIZE);
        conv2d<<<gridDim, blockDim>>>(d_conv2_output, d_conv2_output, 32, 32);
        relu<<<(32 * 32 * 64 + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_conv2_output, 32 * 32 * 64);
        maxpool<<<gridDim, blockDim>>>(d_conv2_output, d_conv3_output, 32, 32, 2);

        // convolutional Layer 3
        gridDim = dim3((16 + BLOCK_SIZE - 1) / BLOCK_SIZE, (16 + BLOCK_SIZE - 1) / BLOCK_SIZE);
        conv2d<<<gridDim, blockDim>>>(d_conv3_output, d_conv3_output, 16, 16);
        relu<<<(16 * 16 * 128 + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_conv3_output, 16 * 16 * 128);
        maxpool<<<gridDim, blockDim>>>(d_conv3_output, d_conv3_output, 16, 16, 2);

        // fully Connected Layer 1
        fullyConnectedLayer<<<1, 1024>>>(d_conv3_output, d_fc1_weights, d_fc1_biases, d_fc1_output, 8192, 1024);
        relu<<<(1024 + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_fc1_output, 1024);

        // output Layer
        fullyConnectedLayer<<<1, 200>>>(d_fc1_output, d_fc2_weights, d_fc2_biases, d_fc2_output, 1024, 200);
        cudaMemcpy(output.data(), d_fc2_output, 200 * sizeof(float), cudaMemcpyDeviceToHost);
    }

    void backward(const std::vector<float>& input, const std::vector<float>& output, const std::vector<float>& target) {
        int outputSize = output.size();
        int inputSize = 1024;
        int fc2_inputSize = 1024;

        for (int i = 0; i < outputSize; ++i) {
            float error = output[i] - target[i];
            cudaMemcpy(&d_fc2_output_grad[i], &error, sizeof(float), cudaMemcpyHostToDevice);
        }

        fullyConnectedLayerBackward<<<1, 200, 1024 * sizeof(float)>>>(d_fc1_output, d_fc2_weights, d_fc2_output_grad, d_fc2_weights_grad, d_fc2_biases_grad, d_fc1_output_grad, 1024, 200);
    
        reluBackward<<<(1024 + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_fc1_output, d_fc1_output_grad, 1024);
        fullyConnectedLayerBackward<<<1, 1024, 64 * sizeof(float)>>>(d_conv3_output, d_fc1_weights, d_fc1_output_grad, d_fc1_weights_grad, d_fc1_biases_grad, d_conv3_output_grad, 8192, 1024);
        conv2dBackward<<<gridDim, blockDim>>>(d_conv2_output, d_conv3_output_grad, d_conv3_kernel, d_conv3_kernel_grad, d_conv3_output_grad, 32, 32);
        
        maxpoolBackward<<<gridDim, blockDim>>>(d_conv2_output, d_conv3_output_grad, d_conv2_output_grad, 32, 32, 2);
        reluBackward<<<(32 * 32 * 64 + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_conv2_output, d_conv2_output_grad, 32 * 32 * 64);
        conv2dBackward<<<gridDim, blockDim>>>(d_conv1_output, d_conv2_output_grad, d_conv2_kernel, d_conv2_kernel_grad, d_conv2_output_grad, 64, 64);
        
        maxpoolBackward<<<gridDim, blockDim>>>(d_conv1_output, d_conv2_output_grad, d_conv1_output_grad, 64, 64, 2); 
        reluBackward<<<(64 * 64 * 32 + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_conv1_output, d_conv1_output_grad, 64 * 64 * 32);
        conv2dBackward<<<gridDim, blockDim>>>(d_input, d_conv1_output_grad, d_conv1_kernel, d_conv1_kernel_grad, d_conv1_output_grad, 64, 64);
    }

    void updateWeights(float learning_rate) {
        updateWeightsKernel<<<(32 * 3 * 3 + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_conv1_kernel, d_conv1_kernel_grad, 32 * 3 * 3, learning_rate);
        updateWeightsKernel<<<(64 * 3 * 3 + BLOCK_SIZE - 1)
    }

    ~CNN() {
        cudaFree(d_input);
        cudaFree(d_conv1_output);
        cudaFree(d_conv2_output);
        cudaFree(d_conv3_output);
        cudaFree(d_fc1_output);
        cudaFree(d_fc2_output);
        cudaFree(d_conv1_kernel);
        cudaFree(d_conv2_kernel);
        cudaFree(d_conv3_kernel);
        cudaFree(d_fc1_weights);
        cudaFree(d_fc1_biases);
        cudaFree(d_fc2_weights);
        cudaFree(d_fc2_biases);
    }
};


float crossEntropyLoss(const std::vector<float>& output, const std::vector<float>& target) {
    float loss = 0.0f;
    for (size_t i = 0; i < output.size(); ++i) {
        loss -= target[i] * log(output[i]);
    }
    return loss;
}

void train(CNN& cnn, TinyImageNetDataLoader& data_loader, int epochs, 
           float initial_lr = 0.001f, float min_lr = 1e-6f) {
    LearningRateScheduler lr_scheduler(initial_lr, min_lr);
    BatchHandler batch_handler(32, data_loader.getNumTrainingSamples());
    std::vector<EpochMetrics> training_history;
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        EpochMetrics metrics = {0.0f, 0.0f, 0.0f, 0.0f, 0, 0};
        float current_lr = lr_scheduler.get_lr(epoch);
        
        std::vector<size_t> batch_indices;
        batch_handler.shuffle();
        
        while (data_loader.getNextTrainingBatch(batch_images, batch_labels)) {
            std::vector<std::vector<float>> input_batch;
            std::vector<std::vector<float>> target_batch;
            
            for (const auto& img : batch_images) {
                input_batch.push_back(normalizeImage(img));
            }
            
            for (int label : batch_labels) {
                std::vector<float> target(200, 0.0f);
                target[label] = 1.0f;
                target_batch.push_back(target);
            }
            
            std::vector<float> batch_predictions;
            for (size_t i = 0; i < input_batch.size(); ++i) {
                std::vector<float> output(200);
                
                cnn.forward(input_batch[i], output);
                batch_predictions.insert(batch_predictions.end(), 
                                      output.begin(), output.end());
                
                cnn.backward(input_batch[i], output, target_batch[i]);
                cnn.updateWeights(current_lr);
            }
            
            BatchMetrics batch_metrics = Validator::calculate_batch_metrics(
                batch_predictions, target_batch);
            
            metrics.train_loss += batch_metrics.loss * batch_metrics.batch_size;
            metrics.train_accuracy += batch_metrics.accuracy * batch_metrics.batch_size;
            metrics.num_train_samples += batch_metrics.batch_size;
        }
        
        if (metrics.num_train_samples > 0) {
            metrics.train_loss /= metrics.num_train_samples;
            metrics.train_accuracy /= metrics.num_train_samples;
        }
        
        EpochMetrics val_metrics = Validator::validate(cnn, data_loader, 32);
        metrics.val_loss = val_metrics.val_loss;
        metrics.val_accuracy = val_metrics.val_accuracy;
        metrics.num_val_samples = val_metrics.num_val_samples;
        
        training_history.push_back(metrics);
        
        std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                  << " - lr: " << current_lr
                  << " - train_loss: " << metrics.train_loss 
                  << " - train_acc: " << metrics.train_accuracy
                  << " - val_loss: " << metrics.val_loss 
                  << " - val_acc: " << metrics.val_accuracy << std::endl;
        
        data_loader.reset();
    }
}


int predict(CNN& cnn, const std::vector<float>& input) {
    std::vector<float> output(256);
    cnn.forward(input, output);
    return std::distance(output.begin(), std::max_element(output.begin(), output.end()));
}