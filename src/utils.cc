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

};  // namespace tag_detection
