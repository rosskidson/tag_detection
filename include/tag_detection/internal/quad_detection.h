#pragma once

#include <opencv2/core.hpp>

#include "tag_detection/types.h"

namespace tag_detection {

std::vector<RawQuad> DetectQuadsInternal(const cv::Mat& img, const cv::Mat& greyscale_img,
                                         bool debug = false);

}  // namespace tag_detection
