#pragma once

#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <optional>

#include "tag_detection/internal/internal_types.h"

namespace tag_detection {

double ToDegrees(double radians);

double ToPositiveDegrees(double radians);

double DeltaAngle(double ang_1, double ang_2);

bool LinesAreConnected(const Line &line_a, const Line &line_b, const double squared_distance);

LineEnds GetConnectedLineEnds(const Line &line_a, const Line &line_b);

Line ExtendLine(const Line &line, const double extension_factor);

std::optional<Eigen::Vector2d> GetIntersection(const Line &line_a, const Line &line_b);

std::optional<cv::Mat> ToGreyscale(const cv::Mat &image);

}  // namespace tag_detection
