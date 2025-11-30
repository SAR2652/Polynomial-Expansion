#include "../../src/inference_server_cutlass/utils.h"
#include "test_utils.h"

TEST_F(WeightsMetadataTest, ConstructorSucceedsWithValidFiles) {
    EXPECT_NO_THROW(WeightsMetadata wm(json_path, bin_path));
}

TEST_F(WeightsMetadataTest, ConstructorThrowsOnMissingJson) {
    json_path = "";
    EXPECT_THROW(WeightsMetadata wm(json_path, bin_path), std::invalid_argument);
}

TEST_F(WeightsMetadataTest, ConstructorThrowsOnMissingWeightsBin) {
    bin_path = "";
    EXPECT_THROW(WeightsMetadata wm(json_path, bin_path), std::invalid_argument);
}
