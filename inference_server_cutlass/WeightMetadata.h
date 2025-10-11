#include <vector>
#include <string>
#include <unordered_map>

struct TensorInfo {
    std::vector<int> shape;
    float scale;
    int zero_point;
    std::string dtype;
    size_t offset;
    size_t size;
};

class WeightMetadata {
public:
    WeightMetadata(const std::string& json_path);
    const TensorInfo& get(const std::string& name) const;

private:
    std::unordered_map<std::string, TensorInfo> metadata_;
};
