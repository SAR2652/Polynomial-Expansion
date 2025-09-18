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
    std::ifstream jf("metadata.json");
    if (!jf) {
        std::cerr << "Failed to open metadata.json\n";
        return 1;
    }
    jf >> meta;

    std::ifstream bin("weights.bin", std::ios::binary);
    if (!bin) {
        std::cerr << "Failed to open weights.bin\n";
        return 1;
    }

    std::vector<void*> device_ptrs;

    for (auto& [name, info] : meta.items()) {
        TensorInfo t;
        t.shape = info["shape"].get<std::vector<int>>();
        t.dtype = info["dtype"];
        t.offset = info["offset"];
        t.size = info["size"];

        if (t.dtype == "int8") {
            t.scale = info["scale"];
            t.zero_point = info["zero_point"];
        }

        size_t nbytes = (t.dtype == "int8") ? t.size * sizeof(int8_t)
                                            : t.size * sizeof(uint16_t);

        std::vector<char> buffer(nbytes);
        bin.seekg(t.offset, std::ios::beg);
        bin.read(buffer.data(), nbytes);

        void* d_ptr;
        cudaError_t err = cudaMalloc(&d_ptr, nbytes);
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc failed for " << name << ": "
                      << cudaGetErrorString(err) << "\n";
            return 1;
        }

        err = cudaMemcpy(d_ptr, buffer.data(), nbytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "cudaMemcpy failed for " << name << ": "
                      << cudaGetErrorString(err) << "\n";
            cudaFree(d_ptr);
            return 1;
        }

        device_ptrs.push_back(d_ptr);

        std::cout << "Loaded tensor: " << name
                  << " (" << t.dtype << ", size=" << t.size << ")\n";
    }

    // ---- Use the weights here (in real inference code) ----

    // Free all allocated device memory
    for (void* ptr : device_ptrs) {
        cudaFree(ptr);
    }

    std::cout << "All device memory freed. Done.\n";
    return 0;
}