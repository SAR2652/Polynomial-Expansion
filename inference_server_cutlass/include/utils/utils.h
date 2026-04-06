#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <nlohmann/json.hpp>


enum class DTypeTag { Int8, Int32, Float32 };

DTypeTag dtype_to_tag(const std::string& s);

template <DTypeTag> struct dtype_map;
template <> struct dtype_map<DTypeTag::Int8>    { using type = int8_t; };
template <> struct dtype_map<DTypeTag::Int32>   { using type = int32_t; };
template <> struct dtype_map<DTypeTag::Float32> { using type = float; };

template <DTypeTag tag>
using dtype_t = typename dtype_map<tag>::type;


struct TensorInfo
{
    std::vector<int> shape;
    std::string dtype;
    int offset;
    int size;
};

class WeightsMetadata {

    private:
        TensorInfo get_kernel_bias_data(nlohmann::json& layer_metadata);

    public:
        WeightsMetadata(const std::string json_path,
                const std::string weights_bin_path);
        std::vector<char> get_data(std::string dtype,
                            int offset, int size) const;
                
        nlohmann::json metadata;

        // allowing modification of internal state inside const methods when
        // the state is logically not part of the object’s “constness”.
        mutable std::ifstream weights_bin;

};

std::tuple<int, int> get_threads_and_blocks(int total_tokens, int threads);
