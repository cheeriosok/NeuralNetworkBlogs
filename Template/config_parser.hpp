#pragma once
#include <fstream>
#include <nlohmann/json.hpp>

class ConfigParser {
public:
    static TrainingManager::TrainingConfig parseConfig(const std::string& config_path) {
        std::ifstream file(config_path);
        if (!file.is_open()) {
            throw std::runtime_error("Unable to open config file: " + config_path);
        }

        nlohmann::json json_config;
        file >> json_config;

        TrainingManager::TrainingConfig config;

        config.epochs = json_config["training"]["epochs"];
        config.initial_lr = json_config["training"]["initial_learning_rate"];
        config.min_lr = json_config["training"]["min_learning_rate"];
        config.batch_size = json_config["training"]["batch_size"];
        config.enable_validation = json_config["training"]["enable_validation"];

        if (json_config["checkpointing"]["enable"]) {
            config.save_frequency = json_config["checkpointing"]["save_frequency"];
            config.checkpoint_dir = json_config["checkpointing"]["checkpoint_dir"];
            config.save_best_only = json_config["checkpointing"]["save_best_only"];
        }

        return config;
    }

    static std::string getDatasetPath(const std::string& config_path) {
        std::ifstream file(config_path);
        if (!file.is_open()) {
            throw std::runtime_error("Unable to open config file: " + config_path);
        }

        nlohmann::json json_config;
        file >> json_config;

        return json_config["data"]["dataset_path"];
    }

    static CNNArchitecture parseArchitecture(const std::string& config_path) {
        std::ifstream file(config_path);
        if (!file.is_open()) {
            throw std::runtime_error("Unable to open config file: " + config_path);
        }

        nlohmann::json json_config;
        file >> json_config;

        CNNArchitecture arch;
        
        for (const auto& layer : json_config["model"]["conv_layers"]) {
            arch.conv_layers.push_back({
                layer["filters"],
                layer["kernel_size"]
            });
        }

        for (const auto& neurons : json_config["model"]["fc_layers"]) {
            arch.fc_layers.push_back(neurons);
        }

        arch.dropout_rate = json_config["model"]["dropout_rate"];

        return arch;
    }
};