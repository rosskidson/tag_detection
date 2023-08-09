#include "tag_detection/internal/utils.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <complex>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <optional>

namespace tag_detection {

constexpr double kPi = 3.14159265358979323846;

double ToDegrees(double radians) {
  return radians * 180.0 / kPi;
}

double ToPositiveDegrees(double radians) {
  auto deg = ToDegrees(radians);
  while (deg < 0) {
    deg += 360.0;
  }
  return deg;
}

double DeltaAngle(const double ang_1, const double ang_2) {
  auto delta = std::abs(ang_1 - ang_2);
  if (delta > kPi) {
    delta -= 2 * kPi;
  }
  return std::abs(delta);
}

bool LinesAreConnected(const Line &line_a, const Line &line_b, const double squared_distance) {
  return (line_a.start - line_b.start).squaredNorm() < squared_distance ||
         (line_a.start - line_b.end).squaredNorm() < squared_distance ||
         (line_a.end - line_b.start).squaredNorm() < squared_distance ||
         (line_a.end - line_b.end).squaredNorm() < squared_distance;
}

double GetDistance(const LineEnds &line_ends) {
  return (line_ends.line_end_a - line_ends.line_end_b).squaredNorm();
}

bool operator<(const LineEnds &lhs, const LineEnds &rhs) {
  return GetDistance(lhs) < GetDistance(rhs);
}

LineEnds GetConnectedLineEnds(const Line &line_a, const Line &line_b) {
  std::vector<LineEnds> all_line_connections;
  all_line_connections.push_back({line_a.start, line_b.start});
  all_line_connections.push_back({line_a.start, line_b.end});
  all_line_connections.push_back({line_a.end, line_b.start});
  all_line_connections.push_back({line_a.end, line_b.end});
  return std::min(all_line_connections[0],
                  std::min(all_line_connections[1],
                           std::min(all_line_connections[2], all_line_connections[3])));
}

Line ExtendLine(const Line &line, const double extension_factor) {
  const auto new_end =
      ((line.end.cast<double>() - line.start.cast<double>()) * extension_factor).cast<int>() +
      line.start;
  const auto new_start =
      ((line.start.cast<double>() - line.end.cast<double>()) * extension_factor).cast<int>() +
      line.end;
  return {new_start, new_end};
}

// Source: https://stackoverflow.com/a/1968345/1524751
std::optional<Eigen::Vector2d> GetIntersection(const Line &line_a, const Line &line_b) {
  // When converting from int to float, add 0.5. Later this will be floored when converting back.
  const Eigen::Vector2d offset{0.5, 0.5};
  const Eigen::Vector2d a0 = line_a.start.cast<double>() + offset;
  const Eigen::Vector2d a1 = line_a.end.cast<double>() + offset;
  const Eigen::Vector2d b0 = line_b.start.cast<double>() + offset;
  const Eigen::Vector2d b1 = line_b.end.cast<double>() + offset;
  const Eigen::Vector2d s1 = a1 - a0;
  const Eigen::Vector2d s2 = b1 - b0;

  const auto den = (-s2.x() * s1.y() + s1.x() * s2.y());
  if (den == 0) {
    return std::nullopt;
  }

  const auto s = (-s1.y() * (a0.x() - b0.x()) + s1.x() * (a0.y() - b0.y())) / den;
  const auto t = (s2.x() * (a0.y() - b0.y()) - s2.y() * (a0.x() - b0.x())) / den;

  if (s >= 0.0 && s <= 1.0 && t >= 0.0 && t <= 1.0) {
    return Eigen::Vector2d{a0.x() + (t * s1.x()), a0.y() + (t * s1.y())};
  }
  return std::nullopt;
}

std::optional<cv::Mat> ToGreyscale(const cv::Mat &image) try {
  if (image.channels() == 1) {
    return image;
  }
  if (image.channels() == 3) {
    cv::Mat bw;
    cv::cvtColor(image, bw, cv::COLOR_BGR2GRAY);
    return bw;
  }
  return std::nullopt;
} catch (const cv::Exception &e) {
  return std::nullopt;
}

Eigen::Vector3d FitParabola(const std::array<Eigen::Vector2d, 3> &pts) {
  Eigen::Matrix3d A{Eigen::Matrix3d::Zero()};
  for (int i = 0; i < 3; ++i) {
    const auto &x = pts[i].x();
    A(i, 0) = x * x;
    A(i, 1) = x;
    A(i, 2) = 1;
  }

  const Eigen::Vector3d B{pts[0].y(), pts[1].y(), pts[2].y()};

  return A.inverse() * B;
}

Eigen::Vector2d FindMinOrMax(const Eigen::Vector3d &parabola_eq) {
  const auto &A = parabola_eq.x();
  const auto &B = parabola_eq.y();
  const auto &C = parabola_eq.z();
  return {-B / (2 * A), C - (B * B) / (4 * A)};
}

}  // namespace tag_detection
