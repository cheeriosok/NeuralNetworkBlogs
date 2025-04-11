// training_utils.hpp
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

struct BatchMetrics {
    float loss;
    float accuracy;
    size_t batch_size;
};

struct EpochMetrics {
    float train_loss;
    float train_accuracy;
    float val_loss;
    float val_accuracy;
    size_t num_train_samples;
    size_t num_val_samples;
};

class LearningRateScheduler {
private:
    float initial_lr;
    float min_lr;
    int warmup_epochs;
    int decay_epochs;
    
public:
    LearningRateScheduler(float initial_lr = 0.001, float min_lr = 1e-6, 
                         int warmup_epochs = 5, int decay_epochs = 50) 
        : initial_lr(initial_lr), min_lr(min_lr),
          warmup_epochs(warmup_epochs), decay_epochs(decay_epochs) {}
    
    float get_lr(int epoch) {
        if (epoch < warmup_epochs) {
            return min_lr + (initial_lr - min_lr) * (float)epoch / warmup_epochs;
        } else if (epoch < warmup_epochs + decay_epochs) {
            float progress = (float)(epoch - warmup_epochs) / decay_epochs;
            return min_lr + 0.5f * (initial_lr - min_lr) * 
                   (1.0f + cos(progress * M_PI));
        }
        return min_lr;
    }
};

class BatchHandler {
private:
    size_t batch_size;
    size_t current_idx;
    std::vector<size_t> indices;
    
public:
    BatchHandler(size_t batch_size, size_t dataset_size) 
        : batch_size(batch_size), current_idx(0) {
        indices.resize(dataset_size);
        std::iota(indices.begin(), indices.end(), 0);
    }
    
    void shuffle() {
        std::random_shuffle(indices.begin(), indices.end());
        current_idx = 0;
    }
    
    bool get_next_batch(std::vector<size_t>& batch_indices) {
        if (current_idx >= indices.size()) {
            return false;
        }
        
        batch_indices.clear();
        size_t remaining = std::min(batch_size, indices.size() - current_idx);
        
        for (size_t i = 0; i < remaining; ++i) {
            batch_indices.push_back(indices[current_idx + i]);
        }
        
        current_idx += remaining;
        return true;
    }
};

class Validator {
public:
    static EpochMetrics validate(CNN& model, TinyImageNetDataLoader& data_loader, 
                               size_t batch_size) {
        EpochMetrics metrics = {0.0f, 0.0f, 0.0f, 0.0f, 0, 0};
        std::vector<std::vector<unsigned char>> batch_images;
        std::vector<int> batch_labels;
        
        while (data_loader.getNextValidationBatch(batch_images, batch_labels)) {
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
            for (const auto& input : input_batch) {
                std::vector<float> output(200);
                model.forward(input, output);
                batch_predictions.insert(batch_predictions.end(), 
                                      output.begin(), output.end());
            }
            
            BatchMetrics batch_metrics = calculate_batch_metrics(
                batch_predictions, target_batch);
            
            metrics.val_loss += batch_metrics.loss * batch_metrics.batch_size;
            metrics.val_accuracy += batch_metrics.accuracy * batch_metrics.batch_size;
            metrics.num_val_samples += batch_metrics.batch_size;
        }
        
        if (metrics.num_val_samples > 0) {
            metrics.val_loss /= metrics.num_val_samples;
            metrics.val_accuracy /= metrics.num_val_samples;
        }
        
        return metrics;
    }
    
private:
    static BatchMetrics calculate_batch_metrics(
        const std::vector<float>& predictions,
        const std::vector<std::vector<float>>& targets) {
        
        BatchMetrics metrics = {0.0f, 0.0f, targets.size()};
        
        for (size_t i = 0; i < targets.size(); ++i) {
            float sample_loss = 0.0f;
            size_t pred_offset = i * 200;
            
            for (size_t j = 0; j < 200; ++j) {
                sample_loss -= targets[i][j] * 
                             log(std::max(predictions[pred_offset + j], 1e-7f));
            }
            metrics.loss += sample_loss;
            
            size_t pred_class = std::max_element(
                predictions.begin() + pred_offset,
                predictions.begin() + pred_offset + 200) - 
                (predictions.begin() + pred_offset);
                
            size_t true_class = std::max_element(
                targets[i].begin(), targets[i].end()) - targets[i].begin();
                
            metrics.accuracy += (pred_class == true_class) ? 1.0f : 0.0f;
        }
        
        metrics.loss /= metrics.batch_size;
        metrics.accuracy /= metrics.batch_size;
        
        return metrics;
    }
};