#include <fstream>
#include <vector>
#include <iostream>
#include <nlohmann/json.hpp>

struct TensorInfo {
    std::vector<int> shape;
    std::string dtype;
    size_t offset;
    size_t size;
    float scale;
    int zero_point;
};

int main() {

    // Load metadata
    nlohmann::json meta;
    std::ifstream jf("../../output/metadata.json");
    if (!jf) {
        std::cerr << "Failed to open metadata.json\n";
        return 1;
    }
    jf >> meta;

    std::ifstream bin("../../output/weights.bin", std::ios::binary);
    if (!bin) {
        std::cerr << "Failed to open weights.bin\n";
        return 1;
    }


    std::cout << "All device memory freed. Done.\n";
    return 0;
}