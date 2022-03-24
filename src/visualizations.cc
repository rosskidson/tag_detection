#include "tag_detection/internal/visualizations.h"

#include <map>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <set>
#include <vector>

#include "tag_detection/internal/internal_types.h"
#include "tag_detection/internal/line_points.h"
#include "tag_detection/internal/utils.h"
#include "tag_detection/types.h"

namespace tag_detection {

cv::Vec3b HSVtoBGR(const cv::Vec3f &hsv) {
  cv::Mat_<cv::Vec3f> hsv_vec(hsv);
  cv::Mat_<cv::Vec3f> bgr_vec;

  cv::cvtColor(hsv_vec, bgr_vec, cv::COLOR_HSV2BGR);

  bgr_vec *= 255;

  return bgr_vec(0);
}

cv::Mat VisualizeGradientDirections(const ImageGradients &gradients) {
  std::vector<double> vals;
  for (int y = 0; y < gradients.abs.rows; ++y) {
    for (int x = 0; x < gradients.abs.cols; ++x) {
      vals.push_back(gradients.abs.at<double>(y, x));
    }
  }
  std::sort(vals.begin(), vals.end());
  const auto max_gradient = vals[vals.size() * 0.8];

  // NOLINTNEXTLINE
  cv::Mat grad_dir_viz(gradients.abs.rows, gradients.abs.cols, CV_8UC3, cv::Scalar::all(0));
  for (int y = 0; y < gradients.direction.rows; ++y) {
    for (int x = 0; x < gradients.direction.cols; ++x) {
      cv::Vec3f hsv{float(ToPositiveDegrees(gradients.direction.at<double>(y, x))),  //
                    1.0,                                                             //
                    float(std::min(gradients.abs.at<double>(y, x) / max_gradient, 1.0))};
      auto &bgr = grad_dir_viz.at<cv::Vec3b>(y, x);
      bgr = HSVtoBGR(hsv);
    }
  }
  return grad_dir_viz;
}

void VisualizeImageGradients(const ImageGradients &gradients) {
  cv::imwrite("01a_img_grad.png", gradients.abs);
  cv::imwrite("01b_img_dir.png", VisualizeGradientDirections(gradients));
}

void VisualizeNonMaxImageGradients(const ImageGradients &gradients, const cv::Mat &non_max_pts) {
  auto abs_viz = gradients.abs.clone();
  auto grad_viz = gradients.direction.clone();
  for (int y = 0; y < non_max_pts.rows; ++y) {
    for (int x = 0; x < non_max_pts.cols; ++x) {
      if (non_max_pts.at<uint8_t>(y, x) > 0) {
        abs_viz.at<double>(y, x) = 0;
        grad_viz.at<double>(y, x) = 0;
      }
    }
  }
  cv::imwrite("02a_img_grad_non_max.png", abs_viz);
  cv::imwrite("02b_img_dir_non_max.png", VisualizeGradientDirections({abs_viz, grad_viz}));
}

cv::Mat VisualizeLinePoints(const std::vector<LinePoints> &lines, const int rows, const int cols) {
  cv::Mat viz_clusters(rows, cols, CV_8UC3, cv::Scalar::all(0));  // NOLINT
  for (const auto &line : lines) {
    const float rand_h = rand() % 360;  // NOLINT rand is fine.
    const auto bgr = HSVtoBGR({rand_h, 1.0, 1.0});
    for (const auto &point : line.Points()) {
      viz_clusters.at<cv::Vec3b>(point.y(), point.x()) = bgr;
    }
  }
  return viz_clusters;
}

cv::Mat VisualizeLines(const cv::Mat &img, const std::vector<Line> &lines) {
  cv::Mat viz_lines = img.clone();
  for (const auto &line : lines) {
    cv::line(viz_lines, {line.start.x(), line.start.y()}, {line.end.x(), line.end.y()}, {0, 255, 0},
             1);
  }
  return viz_lines;
}

cv::Mat VisualizeLineConnectivity(const cv::Mat &img, const std::vector<Line> &lines,
                                  const std::map<int, std::set<int>> &line_connectivity) {
  cv::Mat viz_lines = img.clone();
  for (const auto &[line_id, other_lines] : line_connectivity) {
    for (const auto &other_line_id : other_lines) {
      const auto &line = lines[line_id];
      const auto &other_line = lines[other_line_id];

      const Eigen::Vector2i line_centroid = (line.start + line.end) / 2;
      const Eigen::Vector2i other_line_centroid = (other_line.start + other_line.end) / 2;

      const auto connection_points = GetConnectedLineEnds(line, other_line);

      const Eigen::Vector2i line_pt = (line_centroid + connection_points.line_end_a) / 2;
      const Eigen::Vector2i other_line_pt =
          (other_line_centroid + connection_points.line_end_b) / 2;

      cv::line(viz_lines, {line_pt.x(), line_pt.y()}, {other_line_pt.x(), other_line_pt.y()},
               {255, 0, 255}, 1);
    }
  }
  return viz_lines;
}

cv::Mat VisualizeFinalDetections(const cv::Mat &img, const std::vector<Tag> &detected_tags) {
  constexpr bool kShowCornerIndices = false;

  auto labeled_tags = VisualizeQuads(img, detected_tags);
  for (const auto &tag : detected_tags) {
    const auto corner_0 = tag.corners.front();
    cv::putText(labeled_tags, std::to_string(tag.tag_id),
                cv::Point2i{int(corner_0.x()), int(corner_0.y())}, cv::FONT_HERSHEY_PLAIN, 0.8,
                cv::Scalar(0, 255, 0), 1);
    if constexpr (kShowCornerIndices) {
      for (int i = 0; i < 4; ++i) {
        const auto tag_loc = tag.corners[i];
        cv::putText(labeled_tags, std::to_string(i),
                    cv::Point2i{int(tag_loc.x()), int(tag_loc.y())}, cv::FONT_HERSHEY_PLAIN, 0.6,
                    cv::Scalar(0, 255, 0), 1);
      }
    }
  }
  return labeled_tags;
}

}  // namespace tag_detection
