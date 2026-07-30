#pragma once
#include <string>
#include <fstream>
#include <nlohmann/json.hpp>

using nlohmann::json;


class Model
{
    private:
        Encoder encoder;
        Decoder decoder;

    public:
        Model(const std::string& metadata_path, const std::string& weights_path = "");

        // void forward()

};
