#pragma once

#include <Eigen/Core>
#include <opencv2/core.hpp>

namespace tag_detection {

struct ImageGradients {
  cv::Mat abs;
  cv::Mat direction;
};

struct Line {
  Eigen::Vector2i start;
  Eigen::Vector2i end;
};

struct LineEnds {
  Eigen::Vector2i line_end_a;
  Eigen::Vector2i line_end_b;
};

}  // namespace tag_detection
