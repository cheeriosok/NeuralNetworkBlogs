// training_manager.hpp
#pragma once
#include "cnn.hpp"
#include "data_loader.hpp"
#include "training_utils.hpp"

class TrainingManager {
public:
    struct TrainingConfig {
        int epochs = 100;
        float initial_lr = 0.001f;
        float min_lr = 1e-6f;
        int batch_size = 32;
        bool enable_validation = true;
        std::string checkpoint_dir = "checkpoints";
        int save_frequency = 10; 
    };

    TrainingManager(const std::string& data_path) {
        data_loader = std::make_unique<TinyImageNetDataLoader>(data_path, 32);
        model = std::make_unique<CNN>();
    }

    void configure(const TrainingConfig& config) {
        this->config = config;
        lr_scheduler = std::make_unique<LearningRateScheduler>(
            config.initial_lr, 
            config.min_lr
        );
    }

    void train() {
        if (!model || !data_loader) {
            throw std::runtime_error("Model or data loader not initialized");
        }

        std::vector<EpochMetrics> history;
        BatchHandler batch_handler(config.batch_size, data_loader->getNumTrainingSamples());

        for (int epoch = 0; epoch < config.epochs; ++epoch) {
            auto metrics = train_epoch(batch_handler, lr_scheduler->get_lr(epoch));
            history.push_back(metrics);
            report_progress(epoch, metrics);
            
            if (config.save_frequency > 0 && (epoch + 1) % config.save_frequency == 0) {
                save_checkpoint(epoch);
            }
        }
    }

    void save_model(const std::string& path) {
        model->save(path);
    }

    void load_model(const std::string& path) {
        model->load(path);
    }

    std::vector<float> predict(const std::vector<unsigned char>& image) {
        auto normalized = normalizeImage(image);
        std::vector<float> output(200);
        model->forward(normalized, output);
        return output;
    }

private:
    std::unique_ptr<CNN> model;
    std::unique_ptr<TinyImageNetDataLoader> data_loader;
    std::unique_ptr<LearningRateScheduler> lr_scheduler;
    TrainingConfig config;

    EpochMetrics train_epoch(BatchHandler& batch_handler, float current_lr) {
        EpochMetrics metrics = {0.0f, 0.0f, 0.0f, 0.0f, 0, 0};
        batch_handler.shuffle();

        std::vector<std::vector<unsigned char>> batch_images;
        std::vector<int> batch_labels;

        while (data_loader->getNextTrainingBatch(batch_images, batch_labels)) {
            auto batch_metrics = process_batch(batch_images, batch_labels, current_lr);
            
            metrics.train_loss += batch_metrics.loss * batch_metrics.batch_size;
            metrics.train_accuracy += batch_metrics.accuracy * batch_metrics.batch_size;
            metrics.num_train_samples += batch_metrics.batch_size;
        }

        if (metrics.num_train_samples > 0) {
            metrics.train_loss /= metrics.num_train_samples;
            metrics.train_accuracy /= metrics.num_train_samples;
        }

        if (config.enable_validation) {
            auto val_metrics = Validator::validate(*model, *data_loader, config.batch_size);
            metrics.val_loss = val_metrics.val_loss;
            metrics.val_accuracy = val_metrics.val_accuracy;
            metrics.num_val_samples = val_metrics.num_val_samples;
        }

        data_loader->reset();
        return metrics;
    }

    BatchMetrics process_batch(const std::vector<std::vector<unsigned char>>& batch_images,
                             const std::vector<int>& batch_labels,
                             float learning_rate) {
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
            model->forward(input_batch[i], output);
            batch_predictions.insert(batch_predictions.end(), output.begin(), output.end());
            model->backward(input_batch[i], output, target_batch[i]);
            model->updateWeights(learning_rate);
        }
        
        return Validator::calculate_batch_metrics(batch_predictions, target_batch);
    }

    void save_checkpoint(int epoch) {
        std::string filename = config.checkpoint_dir + "/checkpoint_" + 
                             std::to_string(epoch + 1) + ".pt";
        save_model(filename);
    }

    void report_progress(int epoch, const EpochMetrics& metrics) {
        std::cout << "Epoch " << epoch + 1 << "/" << config.epochs 
                  << " - train_loss: " << metrics.train_loss 
                  << " - train_acc: " << metrics.train_accuracy;
        
        if (config.enable_validation) {
            std::cout << " - val_loss: " << metrics.val_loss 
                     << " - val_acc: " << metrics.val_accuracy;
        }
        std::cout << std::endl;
    }
};