#include "tag_detection/internal/utils.h"

#include <Eigen/Core>
#include <complex>

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

bool get_line_intersection(float p0_x, float p0_y, float p1_x, float p1_y, float p2_x, float p2_y,
                           float p3_x, float p3_y, float *i_x, float *i_y) {
  float s1_x, s1_y, s2_x, s2_y;
  s1_x = p1_x - p0_x;
  s1_y = p1_y - p0_y;
  s2_x = p3_x - p2_x;
  s2_y = p3_y - p2_y;

  float den_0 = (-s2_x * s1_y + s1_x * s2_y);
  float den_1 = (-s2_x * s1_y + s1_x * s2_y);
  if (den_0 == 0 || den_1 == 0) {
    return false;
  }

  float s, t;
  s = (-s1_y * (p0_x - p2_x) + s1_x * (p0_y - p2_y)) / den_0;
  t = (s2_x * (p0_y - p2_y) - s2_y * (p0_x - p2_x)) / den_1;

  if (s >= 0 && s <= 1 && t >= 0 && t <= 1) {
    // Collision detected
    if (i_x != NULL) *i_x = p0_x + (t * s1_x);
    if (i_y != NULL) *i_y = p0_y + (t * s1_y);
    return true;
  }

  return false;  // No collision
}

std::optional<Eigen::Vector2d> GetIntersection(const Line &line_a, const Line &line_b) {
  float x;
  float y;
  const auto success = get_line_intersection(line_a.start.x(), line_a.start.y(), line_a.end.x(),
                                             line_a.end.y(), line_b.start.x(), line_b.start.y(),
                                             line_b.end.x(), line_b.end.y(), &x, &y);
  if (success) {
    return Eigen::Vector2f(x, y).cast<double>();
  } else {
    return std::nullopt;
  }
}

};  // namespace tag_detection
