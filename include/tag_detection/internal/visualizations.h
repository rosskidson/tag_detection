#pragma once

#include <map>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <set>

#include "tag_detection/internal/internal_types.h"
#include "tag_detection/internal/line_points.h"
#include "tag_detection/types.h"

namespace tag_detection {

cv::Mat VisualizeGradientDirections(const ImageGradients &gradients);
void VisualizeImageGradients(const ImageGradients &gradients);
void VisualizeNonMaxImageGradients(const ImageGradients &gradients, const cv::Mat &non_max_pts);
cv::Mat VisualizeLinePoints(const std::vector<LinePoints> &lines, int rows, int cols);
cv::Mat VisualizeLines(const cv::Mat &img, const std::vector<Line> &lines);

cv::Mat VisualizeLineConnectivity(const cv::Mat &img, const std::vector<Line> &lines,
                                  const std::map<int, std::set<int>> &line_connectivity);

template <typename Quad>
cv::Mat VisualizeQuads(const cv::Mat &img, const std::vector<Quad> &quads);

cv::Mat VisualizeFinalDetections(const cv::Mat &img, const std::vector<Tag> &detected_tags);

template <typename Quad>
cv::Mat VisualizeQuads(const cv::Mat &img, const std::vector<Quad> &quads) {
  cv::Mat viz = img.clone();
  int quad_counter{};
  for (const auto &quad : quads) {
    for (int i = 0; i < 4; ++i) {
      const auto pt_a = quad.corners[i].template cast<int>();
      const auto pt_b = quad.corners[(i + 1) % 4].template cast<int>();
      cv::line(viz, {pt_a.x(), pt_a.y()}, {pt_b.x(), pt_b.y()}, {0, 255, 0}, 1);
      const auto tag_loc = quad.corners[i];
    }
  }
  return viz;
}

}  // namespace tag_detection
