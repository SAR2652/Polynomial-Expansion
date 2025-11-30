#pragma once
#include <string>
#include <fstream>
#include <nlohmann/json.hpp>

using nlohmann::json;




class Decoder {
public:
    void build_from_metadata(const json& meta);
    void load_weights(const json& meta, std::ifstream& bin);
    std::vector<float> forward(const std::vector<int32_t>& input_ids);
};


class Model {
public:
    // Constructor: builds model from metadata, optionally loads weights
    Model(const std::string& metadata_path, const std::string& weights_path = "");

    // Deferred weight loading
    void load_weights(const std::string& weights_path);

    // Inference API
    std::vector<float> forward(const std::vector<int32_t>& input_ids);

private:
    Encoder encoder;
    Decoder decoder;
    nlohmann::json metadata;
    std::ifstream weights_stream;
    bool weights_loaded = false;

    void build_model_structure(const nlohmann::json& meta);
};
