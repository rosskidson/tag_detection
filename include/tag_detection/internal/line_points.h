#pragma once

#include <Eigen/Core>
#include <cmath>

namespace tag_detection {

class LinePoints {
 public:
  void AddPoint(const Eigen::Vector2i &point, const double gradient_direction) {
    points_.push_back(point);
    angles_.push_back(gradient_direction);
    x_sum_ += std::cos(gradient_direction);
    y_sum_ += std::sin(gradient_direction);
  }

  double GetMeanDirection() const {
    const auto ave_y = y_sum_ / points_.size();
    const auto ave_x = x_sum_ / points_.size();
    const auto ave = std::atan2(ave_y, ave_x);
    return ave;
  }

  std::size_t Size() const { return points_.size(); }
  bool Empty() const { return points_.empty(); }
  const std::vector<Eigen::Vector2i> &Points() const { return points_; }

 private:
  std::vector<Eigen::Vector2i> points_;
  std::vector<double> angles_;
  double x_sum_{};
  double y_sum_{};
};

}  // namespace tag_detection
