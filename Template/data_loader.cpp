// https://pastebin.com/fa4Tqp4i

#pragma once

#include <mutex>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include <iostream>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <memory>
#include <thread>
#include <atomic>
#include <deque>
#include <filesystem>
#include <future>
#include <arm_neon.h>
#include "stb_image.h"

//===================================
//      Concurrency Primitives
//===================================

template<typename T>
class Queue {
public:

    Queue() : head(nullptr), tail(nullptr) {}
    
    ~Queue() {
        while (head != nullptr) { // standard linked list destructor
            Node* next = head->next;
            delete head;
            head = next;
        }
    }

    template<typename U>
    inline bool enqueue(U&& item) {
        Node* node = new Node(std::forward<U>(item)); // important thing to note here 1. forward preserves lvalue rvalue type and 2. we initialzie Node before our lock
        // as memory creation is pretty slow and doesn't really need thread safety.
        std::lock_guard<std::mutex> smart_mutex(mutex); // initalize block-level 'smart mutex'
        /* Critical Section */
        if (tail == nullptr) { // Our Queue is empty - so head, tail now point to the same address pointed to by node
            head = tail = node;
        } else { // not empty
            tail->next = node; // current tail points to last node in line, so lets have tail->next point to our new last in line 
            tail = node; // now tail points to this node last in line
        }
        return true; // once this returns our lockguard/smart_mutex lifts its lock
    }

    inline bool dequeue(T& item) {
        std::lock_guard<std::mutex> smart_mutex(mutex); // initalize block-level 'smart mutex'
        if (head == nullptr) { // node is empty - nothing to remove, return false;
            return false;
        } else { // 
            item = std::move(head->item); // data moved to input item from head 
            Node* next = head->next; // next is  the 2nd position
            delete head; // now we have head's data and his next pointer, we can delete
            head = next; // lets say head == next, and head is basically gone
            if (head == nullptr) { // if head == nullptr then empty list. tail should be nullptr
                tail = nullptr; // tail may still point to the last element, thus lets set it to nullptr
            }
            return true;
        }
    }

private:
    struct Node {
        T item;
        Node* next;
        template<typename U>
        Node(U&& item) : item(std::forward<U>(item)), next(nullptr) {}
    };

    std::mutex mutex;
    Node* head;
    Node* tail;
};

// i wish i was born a slug
class ThreadPool {
private:
    std::vector<std::thread> workers;
    Queue<std::function<void()>> tasks;
    std::atomic<bool> stop{false};
    std::condition_variable condition;
    std::mutex mutex;
    
    size_t active_tasks{0};
    std::atomic<size_t> total_tasks_completed{0};

public:
    ThreadPool(size_t threads) {
        workers.reserve(threads); 
        for (size_t i = 0; i < threads; ++i) { 
            workers.emplace_back([this] { 
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        condition.wait(lock, [this] {
                            return stop || !tasks.empty();
                        });
                        
                        if (stop && tasks.empty()) {
                            return;
                        }
                        
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    
                    active_tasks++;
                    try {
                        task();
                    } catch (...) {
                        // log error
                    }
                    active_tasks--;
                    total_tasks_completed++;
                }
            });
        }
    }

    template<class F>
    void enqueue(F&& f) {
        tasks.enqueue(std::forward<F>(f));
    }

    ~ThreadPool() {
        stop = true;
        for (auto& worker : workers) {
            worker.join();
        }
    }
};
//===================================
//          Data Loader
//===================================

class TinyImageNetDataLoader {
private:
    // image dimension constraints, 64x64 pixels, 3 channels (rgb)
    const int IMAGE_SIZE = 64;
    const int CHANNELS = 3;

    int batch_size; 
    std::atomic<size_t> curr_batch_idx{0}; 
    std::string data_path; 

    std::deque<std::vector<unsigned char>> train_images; // training images stored as bytes in a double-ended queue
    std::deque<std::vector<unsigned char>> val_images; // validation images stored as bytes in a double-ended queue
    std::vector<int> train_labels; // int labels for training images
    std::vector<int> val_labels; // int labels for validation images

    // mapping and organization 
    std::map<std::string, int> class_labels; // class names to numbers
    std::vector<size_t> shuffle_indices; // randomize order (in case our inputs are sorted!)

    // concurrency & randomnization
    std::unique_ptr<ThreadPool> thread_pool; // instance of our threadpool above
    alignas(16) std::mt19937 rng; 

    std::vector<float> normalizeImage(const std::vector<unsigned char>& image) const {
        std::vector<float> normalized(image.size()); // initialize our normalized vector given same size as image
        constexpr float scale = 1.0f / 255.0f; // scale 0-255 to a float between 0-1
        
        // Process 8 elements at a time - many compilers will auto-vectorize this
        // SIMD optimization - we normalize 8 images at a time for vectorization
        // normalization = image rgb * 1.0f / 255f
    
        const size_t step = 8; 
        const size_t vectorized_size = (image.size() / step) * step; // amt of iterations if we process 8 at a time
        
        for (size_t i = 0; i < vectorized_size; i += step) {
            normalized[i] = static_cast<float>(image[i]) * scale;
            normalized[i + 1] = static_cast<float>(image[i + 1]) * scale;
            normalized[i + 2] = static_cast<float>(image[i + 2]) * scale;
            normalized[i + 3] = static_cast<float>(image[i + 3]) * scale;
            normalized[i + 4] = static_cast<float>(image[i + 4]) * scale;
            normalized[i + 5] = static_cast<float>(image[i + 5]) * scale;
            normalized[i + 6] = static_cast<float>(image[i + 6]) * scale;
            normalized[i + 7] = static_cast<float>(image[i + 7]) * scale;
        }

        // handle remaining elements
        for (size_t i = vectorized_size; i < image.size(); ++i) {
            normalized[i] = static_cast<float>(image[i]) * scale;
        }

        return normalized;
    }
        // performance optimization to use CPU cache prefetching to improve speeds - can ignore
        // __builtin_prefetch is compiler-inrinsic function call with 3 arguments - address, 0/1 reading/witing, and fetch highest temporal locality (3) 
        void prefetchNextBatch() {
            const size_t next_idx = (curr_batch_idx + 1) % train_images.size();
            __builtin_prefetch(&train_images[next_idx], 0, 3); 
            __builtin_prefetch(&train_labels[next_idx], 0, 3);
        }

    void loadClassLabels() {
        std::ifstream label_file(data_path + "/webld.txt"); // ifstream method 
        if (!label_file) {
            throw std::runtime_error("Unable to open labels file: " + data_path + "/webld.txt");
        }

        std::string line;
        int label_index = 0;
        while (std::getline(label_file, line)) {
            line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
            if (!line.empty()) {
                class_labels[line] = label_index++;
            }
        }

        if (class_labels.empty()) {
            throw std::runtime_error("No class labels loaded");
        }
    }

    void loadDataset(const std::string& subset_path, std::deque<std::vector<unsigned char>>& images, std::vector<int>& labels) {
        if (!std::filesystem::exists(subset_path)) {
            throw std::runtime_error("Dataset path does not exist: " + subset_path);
        }

        struct LoadTask {
            std::string filepath;
            int label;
        };

        std::vector<LoadTask> tasks;
        std::atomic<size_t> total_images{0};
        std::atomic<size_t> failed_loads{0};

        for (const auto& entry : std::filesystem::directory_iterator(subset_path)) {
            if (std::filesystem::is_directory(entry)) {
                std::string class_name = entry.path().filename().string();
                if (class_labels.find(class_name) == class_labels.end()) {
                    std::cerr << "Warning: Unknown class folder found: " << class_name << std::endl;
                    continue;
                }
                int label = class_labels[class_name];
                for (const auto& img_entry : std::filesystem::directory_iterator(entry.path() / "images")) {
                    tasks.push_back({img_entry.path().string(), label});
                }
            }
        }

        const size_t num_tasks = tasks.size();
        images.resize(num_tasks);
        labels.resize(num_tasks);

        std::vector<std::future<void>> futures;
        const size_t chunk_size = std::max(size_t(1), num_tasks / std::thread::hardware_concurrency());

        for (size_t i = 0; i < num_tasks; i += chunk_size) {
            const size_t end = std::min(i + chunk_size, num_tasks);
            futures.push_back(std::async(std::launch::async, [this, &tasks, &images, &labels, i, end, &total_images, &failed_loads]() {
                for (size_t j = i; j < end; ++j) {
                    try {
                        auto image = loadImage(tasks[j].filepath);
                        if (!image.empty()) {
                            images[j] = std::move(image);
                            labels[j] = tasks[j].label;
                            total_images++;
                        } else {
                            failed_loads++;
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "Error loading image " << tasks[j].filepath << ": " << e.what() << std::endl;
                        failed_loads++;
                    }
                }
            }));
        }

        for (auto& future : futures) {
            future.wait();
        }

        if (total_images == 0) {
            throw std::runtime_error("No images loaded from " + subset_path);
        }

        if (failed_loads > 0) {
            std::cerr << "Warning: Failed to load " << failed_loads << " images" << std::endl;
        }

        std::cout << "Loaded " << total_images << " images from " << subset_path << std::endl;
    }

    void loadTrainingData() {
        loadDataset(data_path + "/train", train_images, train_labels);
        initializeShuffleIndices();
    }

    void loadValidationData() {
        loadDataset(data_path + "/val", val_images, val_labels);
    }

    std::vector<unsigned char> loadImage(const std::string& filepath) {
        int width, height, channels;
        unsigned char* img = stbi_load(filepath.c_str(), &width, &height, &channels, CHANNELS);

        if (!img) {
            throw std::runtime_error("Failed to load image: " + filepath + " - " + stbi_failure_reason());
        }

        if (width != IMAGE_SIZE || height != IMAGE_SIZE) {
            stbi_image_free(img);
            throw std::runtime_error("Invalid image dimensions: " + filepath);
        }

        std::vector<unsigned char> image(img, img + width * height * CHANNELS);
        stbi_image_free(img);
        return image;
    }

    void initializeShuffleIndices() {
        shuffle_indices.resize(train_images.size());
        std::iota(shuffle_indices.begin(), shuffle_indices.end(), 0);
        shuffleTrainingData();
    }

public:
    TinyImageNetDataLoader(const std::string& data_path, int batch_size, unsigned int seed = std::random_device{}()) 
        : data_path(data_path), batch_size(batch_size), rng(seed) {
        thread_pool = std::make_unique<ThreadPool>(std::thread::hardware_concurrency());
        loadClassLabels();
        loadTrainingData();
        loadValidationData();
    }

    void shuffleTrainingData() {
        std::shuffle(shuffle_indices.begin(), shuffle_indices.end(), rng);
        curr_batch_idx = 0;
    }

    size_t getNumTrainingSamples() const { return train_images.size(); }
    size_t getNumValidationSamples() const { return val_images.size(); }
    size_t getNumClasses() const { return class_labels.size(); }

    bool getNextTrainingBatch(std::vector<std::vector<unsigned char>>& images, std::vector<int>& labels, bool shuffle_on_epoch_end = true) {
        if (curr_batch_idx >= train_images.size()) {
            if (shuffle_on_epoch_end) {
                shuffleTrainingData();
            }
            return false;
        }

        images.clear();
        labels.clear();

        for (int i = 0; i < batch_size && curr_batch_idx < train_images.size(); i++) {
            size_t idx = shuffle_indices[curr_batch_idx++];
            images.push_back(train_images[idx]);
            labels.push_back(train_labels[idx]);
        }

        return true;
    }

    bool getNextValidationBatch(std::vector<std::vector<unsigned char>>& images, std::vector<int>& labels) {
        if (curr_batch_idx >= val_images.size()) {
            return false;
        }

        images.clear();
        labels.clear();

        for (int i = 0; i < batch_size && curr_batch_idx < val_images.size(); i++) {
            images.push_back(val_images[curr_batch_idx]);
            labels.push_back(val_labels[curr_batch_idx]);
            curr_batch_idx++;
        }

        return true;
    }

    void reset(bool shuffle = true) {
        curr_batch_idx = 0;
        if (shuffle) {
            shuffleTrainingData();
        }
    }

    std::string getLabelName(int label) const {
        for (const auto& pair : class_labels) {
            if (pair.second == label) {
                return pair.first;
            }
        }
        throw std::runtime_error("Invalid label: " + std::to_string(label));
    }

    ~TinyImageNetDataLoader() = default;


    };