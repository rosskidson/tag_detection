#pragma once

#include <opencv2/core.hpp>

#include "tag_detection/types.h"

namespace tag_detection {

std::vector<RawQuad> FindQuads(const cv::Mat& img, const bool debug = false);

}  // namespace tag_detection
