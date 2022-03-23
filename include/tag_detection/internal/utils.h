#pragma once

#include "tag_detection/internal/internal_types.h"

namespace tag_detection {

double ToDegrees(double radians);

double ToPositiveDegrees(double radians);

double DeltaAngle(const double ang_1, const double ang_2);

std::optional<Eigen::Vector2d> GetIntersection(const Line &line_a, const Line &line_b);

}  // namespace tag_detection
