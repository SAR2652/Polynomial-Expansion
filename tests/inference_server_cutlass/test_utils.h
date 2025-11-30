#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

class WeightsMetadataTest : public ::testing::Test {
protected:
    std::string json_path;
    std::string bin_path;

    void SetUp() override {
        json_path = std::filesystem::absolute("../../../output/metadata.json")
        .string();
        bin_path = std::filesystem::absolute("../../../output/weights.bin")
        .string();


        // Create dummy metadata.json
        // std::ofstream json_file(json_path);
        // json_file << R"({"layers": [{"name": "embedding", "dtype": "int8"}]})";
        // json_file.close();

        // // Create dummy weights.bin with known content
        // std::ofstream bin_file(bin_path, std::ios::binary);
        // for (int i = 0; i < 100; ++i) {
        //     int8_t val = static_cast<int8_t>(i);
        //     bin_file.write(reinterpret_cast<char*>(&val), sizeof(int8_t));
        // }
        // bin_file.close();
    }

    void TearDown() override {
        // fs::remove(json_path);
        // fs::remove(bin_path);
    }
};
