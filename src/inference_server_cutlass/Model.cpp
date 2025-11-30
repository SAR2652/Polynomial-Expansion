#include "Model.h"
#include <stdexcept>
#include <iostream>

Model::Model(const std::string& metadata_path, const std::string& weights_path)
{
    std::ifstream meta_file(metadata_path);
    if (!meta_file) throw std::runtime_error("Failed to open metadata file.");

    meta_file >> metadata;
    build_model_structure(metadata);

    if (!weights_path.empty()) {
        load_weights(weights_path);
    }
}

void Model::build_model_structure(const nlohmann::json& meta) {
    if (meta.contains("encoder")) {
        encoder.build_from_metadata(meta["encoder"]);
    }
    if (meta.contains("decoder")) {
        decoder.build_from_metadata(meta["decoder"]);
    }
}

void Model::load_weights(const std::string& weights_path) {
    weights_stream.open(weights_path, std::ios::binary);
    if (!weights_stream) throw std::runtime_error("Failed to open weights file.");

    encoder.load_weights(metadata["encoder"], weights_stream);
    decoder.load_weights(metadata["decoder"], weights_stream);
    weights_loaded = true;
}

std::vector<float> Model::forward(const std::vector<int32_t>& input_ids) {
    if (!weights_loaded) throw std::runtime_error("Weights not loaded.");

    auto encoder_output = encoder.forward(input_ids);
    auto decoder_output = decoder.forward(encoder_output);
    return decoder_output;
}


