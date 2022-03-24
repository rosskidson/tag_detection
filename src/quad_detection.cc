#include "tag_detection/internal/quad_detection.h"

#include <Eigen/Core>
#include <deque>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <set>

#include "tag_detection/internal/internal_types.h"
#include "tag_detection/internal/line_points.h"
#include "tag_detection/internal/timer.h"
#include "tag_detection/internal/utils.h"
#include "tag_detection/internal/visualizations.h"
#include "tag_detection/types.h"

namespace tag_detection {

// Tuning parameters

// Gaussian blue kernel size for blurring the initial input image.
constexpr int kBlurKernelSize = 5;

// Minimum image gradient when thresholding edges.
constexpr double kAbsImgGradientThresh = 100;

// Maximum allowable difference in image gradient angle for accepting a point into a line cluster.
constexpr double kMaxAngleClusterDiff = M_PI / 8;

// Minimum number of pixels for a valid line cluster.
constexpr int kMinLineClusterSize = 5;

// Minimum allowable candidate line length.
constexpr double kMinLineLength = 8;

// How much to extend candidate lines. This is to help finding intersections at quad corners.
constexpr double kLineExtensionFactor = 1.25;

// Half the window size for subpixel refinement. The total window size will be (kWinSize * 2) + 1.
constexpr int kWinSize = 2;

ImageGradients CalculateImageGradients(const cv::Mat &mat) {
  ImageGradients gradients{};
  cv::Mat grad_x, grad_y;
  cv::Sobel(mat, grad_x, CV_32F, 1, 0);  // Opt: try 16S (short int)
  cv::Sobel(mat, grad_y, CV_32F, 0, 1);

  gradients.abs = cv::Mat(mat.rows, mat.cols, CV_64F, cv::Scalar(0));  // Opt: use short
  gradients.direction = cv::Mat(mat.rows, mat.cols, CV_64F,
                                cv::Scalar(0));  // Opt: use short? (when int atan2 possible)
  for (int y = 0; y < mat.rows; ++y) {
    for (int x = 0; x < mat.cols; ++x) {
      const auto &dx = grad_x.at<float>(y, x);
      const auto &dy = grad_y.at<float>(y, x);
      gradients.abs.at<double>(y, x) =
          std::sqrt(dx * dx + dy * dy);  // Removing sqrt had no effect on runtime
      gradients.direction.at<double>(y, x) = std::atan2(dy, dx);
    }
  }
  return gradients;
}

cv::Mat NonMaximaSuppression(const ImageGradients &gradients, const double min_abs_gradient) {
  const static std::vector<std::vector<Eigen::Vector2i>> gradient_lookup{
      {Eigen::Vector2i{-1, 0}, Eigen::Vector2i{1, 0}},   // 0
      {Eigen::Vector2i{-1, -1}, Eigen::Vector2i{1, 1}},  // 45
      {Eigen::Vector2i{0, 1}, Eigen::Vector2i{0, -1}},   // 90
      {Eigen::Vector2i{-1, 1}, Eigen::Vector2i{1, -1}},  // 135
      {Eigen::Vector2i{-1, 0}, Eigen::Vector2i{1, 0}},   // 180
      {Eigen::Vector2i{-1, -1}, Eigen::Vector2i{1, 1}},  // 225
      {Eigen::Vector2i{0, 1}, Eigen::Vector2i{0, -1}},   // 270
      {Eigen::Vector2i{-1, 1}, Eigen::Vector2i{1, -1}},  // 315
      {Eigen::Vector2i{-1, 0}, Eigen::Vector2i{1, 0}}};  // 360

  // 0 = max point, 1 = non max point
  cv::Mat non_max_pts(gradients.abs.rows, gradients.abs.cols, CV_8U, cv::Scalar(0));

  non_max_pts.reserve(gradients.abs.cols * gradients.abs.rows);
  for (int y = 1; y < gradients.abs.rows - 1; ++y) {
    for (int x = 1; x < gradients.abs.cols - 1; ++x) {
      auto &current_val = gradients.abs.at<double>(y, x);
      if (current_val < min_abs_gradient) {
        non_max_pts.at<uint8_t>(y, x) = 1;
        continue;
      }
      const double direction_deg = ToPositiveDegrees(gradients.direction.at<double>(y, x));
      assert(direction_deg <= 360 && direction_deg >= 0);
      const int index = std::round(direction_deg / 45.0);
      const auto &directions = gradient_lookup[index];
      const auto &val_0 = gradients.abs.at<double>(y + directions[0].y(), x + directions[0].x());
      const auto &val_1 = gradients.abs.at<double>(y + directions[1].y(), x + directions[1].x());
      if (val_0 > current_val || val_1 > current_val) {
        non_max_pts.at<uint8_t>(y, x) = 1;
      }
    }
  }
  return non_max_pts;
}

LinePoints ClusterPoints(const Eigen::Vector2i &start_pt, const double ang_thresh,
                         const ImageGradients &gradients, cv::Mat *processed_points) {
  const static std::vector<Eigen::Vector2i> CONNECT_FOUR{{-1, 0}, {0, -1}, {1, 0}, {0, 1}};
  const static std::vector<Eigen::Vector2i> CONNECT_EIGHT{{-1, -1}, {0, -1}, {1, -1}, {1, 0},
                                                          {1, 1},   {0, 1},  {-1, 1}, {-1, 0}};
  LinePoints points{};
  std::deque<Eigen::Vector2i> open_points;
  open_points.push_back(start_pt);
  while (not open_points.empty()) {
    auto current_point = open_points.back();
    auto current_dir = gradients.direction.at<double>(current_point.y(), current_point.x());
    open_points.pop_back();

    points.AddPoint(current_point, current_dir);
    processed_points->at<uint8_t>(current_point.y(), current_point.x()) = 1;

    for (const auto &dir : CONNECT_EIGHT) {
      const auto &candidate = current_point + dir;
      if (candidate.x() < 0 || candidate.x() >= gradients.abs.cols || candidate.y() < 0 ||
          candidate.y() >= gradients.abs.rows) {
        continue;
      }
      if (processed_points->at<uint8_t>(candidate.y(), candidate.x()) > 0) {
        continue;
      }

      auto grad_dir = gradients.direction.at<double>(candidate.y(), candidate.x());
      if (DeltaAngle(points.GetMeanDirection(), grad_dir) < ang_thresh) {
        open_points.push_back(candidate);
      }
    }
  }

  return points;
}

std::vector<LinePoints> ClusterGradientDirections(const ImageGradients &gradients,
                                                  const cv::Mat &non_max_pts,
                                                  const double ang_thresh,
                                                  const int min_cluster_size) {
  std::vector<LinePoints> lines;
  cv::Mat processed_points = non_max_pts.clone();

  for (int y = 0; y < gradients.abs.rows; ++y) {
    for (int x = 0; x < gradients.abs.cols; ++x) {
      if (processed_points.at<uint8_t>(y, x) > 0) {
        continue;
      }
      const auto cluster =
          ClusterPoints(Eigen::Vector2i{x, y}, ang_thresh, gradients, &processed_points);
      if (cluster.Size() >= min_cluster_size) {
        lines.push_back(cluster);
      }
    }
  }

  return lines;
}

Eigen::Vector2i GetFurthestPoint(const Eigen::Vector2i &compare_pt, const LinePoints &line_points) {
  Eigen::Vector2i ret_val = compare_pt;
  double max_dist = 0;
  for (const auto &pt : line_points.Points()) {
    const double dist = (compare_pt - pt).squaredNorm();
    if (dist > max_dist) {
      ret_val = pt;
      max_dist = dist;
    }
  }
  return ret_val;
}

Line MakeLine(const LinePoints &line_points) {
  auto any_pt = line_points.Points()[line_points.Size() / 2];
  Line line{};
  line.start = GetFurthestPoint(any_pt, line_points);
  line.end = GetFurthestPoint(line.start, line_points);

  const auto new_end =
      ((line.end.cast<double>() - line.start.cast<double>()) * kLineExtensionFactor).cast<int>() +
      line.start;
  const auto new_start =
      ((line.start.cast<double>() - line.end.cast<double>()) * kLineExtensionFactor).cast<int>() +
      line.end;
  line.start = new_start;
  line.end = new_end;
  return line;
}

std::vector<Line> MakeLines(const std::vector<LinePoints> &lines_points,
                            const double min_line_length) {
  const double min_line_length_squared = min_line_length * min_line_length;
  std::vector<Line> lines;
  lines.reserve(lines_points.size());
  for (const auto &line_points : lines_points) {
    auto line = MakeLine(line_points);
    if ((line.start - line.end).squaredNorm() < min_line_length_squared) {
      continue;
    }
    lines.push_back(std::move(line));
  }
  return lines;
}

std::map<int, std::set<int>> MakeLineConnectivity(const std::vector<Line> &lines) {
  std::map<int, std::set<int>> line_connectivity;
  for (int i = 0; i < lines.size(); ++i) {
    for (int j = i + 1; j < lines.size(); ++j) {
      const auto intersect = GetIntersection(lines[i], lines[j]);
      if (intersect.has_value()) {
        line_connectivity[i].insert(j);
        line_connectivity[j].insert(i);
      }
    }
  }
  return line_connectivity;
}

RawQuad CreateQuad(const std::vector<int> &quad_line_ids, const std::vector<Line> &lines) {
  RawQuad quad{};
  for (int i = 0; i < 4; ++i) {
    const auto intersect =
        GetIntersection(lines[quad_line_ids[i]], lines[quad_line_ids[(i + 1) % 4]]);
    quad.corners[i] = intersect.value();
  }

  // TODO:: put corner ordering in a function
  // Calculate centroid. (put in function)
  Eigen::Vector2d centroid{0, 0};
  for (const auto &corner : quad.corners) {
    centroid += corner;
  }
  centroid /= 4.0;

  struct CornerWithAngle {
    Eigen::Vector2d corner;
    double angle;
  };

  // Get the angle of each point from the center.
  std::vector<CornerWithAngle> corners;
  corners.reserve(4);
  for (const auto &corner : quad.corners) {
    auto &corner_with_angle = corners.emplace_back();
    corner_with_angle.corner = corner;
    auto &angle = corner_with_angle.angle;
    angle = std::atan2(corner.y() - centroid.y(), corner.x() - centroid.x());
    if (angle < 0) {
      angle += 2 * M_PI;
    }
  }

  // Sort the points by angle, descending to get them in clockwise order. This results in
  // anti-clockwise when viewing the image with on a screen, where the y axis is inverted. The first
  // point will be in the top right quadrant.
  std::sort(corners.begin(), corners.end(), [](const auto &corner_1, const auto &corner_2) {
    return corner_1.angle > corner_2.angle;
  });

  // After sorting, we want the first corner to be in the bottom right quadrant, to be consistent
  // with the output convention that the first corner is bottom right, then rotate around
  // anti-clockwise. So move all corners around by two.
  // If the tag needs rotating because it is not oriented upright in the image frame, this will
  // happen later after decoding the tag and determining its orientation.

  for (int i = 0; i < 4; ++i) {
    quad.corners[i] = corners[(i + 2) % 4].corner;
  }

  return quad;
}

struct UniqueQuad {
  UniqueQuad(const std::vector<int> &quad_line_ids) {
    assert(quad_line_ids.size() == 4);
    for (int i = 0; i < 4; ++i) {
      line_ids[i] = quad_line_ids[i];
    }
    std::sort(line_ids.begin(), line_ids.end());
  }

  std::array<int, 4> line_ids{};
};

bool operator<(const UniqueQuad &lhs, const UniqueQuad &rhs) {
  for (int i = 0; i < 4; ++i) {
    if (lhs.line_ids[i] < rhs.line_ids[i]) {
      return true;
    }
    if (lhs.line_ids[i] > rhs.line_ids[i]) {
      return false;
    }
  }
  return false;
}

bool LinesConnected(const int id_1, const int id_2,
                    const std::map<int, std::set<int>> &line_connectivity) {
  return line_connectivity.count(id_1) == 1 && line_connectivity.at(id_1).count(id_2) == 1;
}

template <typename T>
bool VectorContainsVal(const std::vector<T> &vec, const T &val) {
  return std::find(vec.begin(), vec.end(), val) != vec.end();
}

std::vector<std::vector<int>> FindQuadsFromStartLine(
    const std::vector<Line> &lines,                         //
    const std::map<int, std::set<int>> &line_connectivity,  //
    const int start_line_id) {                              //
  std::vector<std::vector<int>> quads;
  std::deque<std::vector<int>> potential_quads;
  potential_quads.push_back({start_line_id});
  while (not potential_quads.empty()) {
    const auto potential_quad = potential_quads.back();
    potential_quads.pop_back();

    if (potential_quad.size() == 4 &&
        LinesConnected(potential_quad.front(), potential_quad.back(), line_connectivity)) {
      quads.push_back(potential_quad);
    }
    if (potential_quad.size() < 4) {
      for (const auto &new_edge : line_connectivity.at(potential_quad.back())) {
        if (VectorContainsVal(potential_quad, new_edge)) {
          continue;
        }

        auto new_potential_quad = potential_quad;
        new_potential_quad.push_back(new_edge);
        potential_quads.push_back(new_potential_quad);
      }
    }
  }

  return quads;
}

bool CheckQuadSideLengths(const RawQuad &quad, const double side_length) {
  const double side_length_squared = side_length * side_length;
  for (int i = 0; i < 4; ++i) {
    const auto pt_a = quad.corners[i];
    const auto pt_b = quad.corners[(i + 1) % 4];
    if ((pt_a - pt_b).squaredNorm() < side_length_squared) {
      return false;
    }
  }
  return true;
}

bool CheckQuadInsideImage(const RawQuad &quad, const int width, const int height) {
  for (const auto &corner : quad.corners) {
    if (corner.x() < 0 || corner.x() > width || corner.y() < 0 || corner.y() > height) {
      return false;
    }
  }
  return true;
}

template <typename EigenPointContainer>
std::vector<cv::Point2f> ToCvPoints(const EigenPointContainer &points) {
  std::vector<cv::Point2f> cv_points;
  for (const auto &pt : points) {
    cv_points.push_back({float(pt.x()), float(pt.y())});
  }
  return cv_points;
}

void RefineEdges(const cv::Mat &image, RawQuad *quad) {
  auto corners = ToCvPoints(quad->corners);
  auto term_criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 40, 0.001);
  cv::cornerSubPix(image, corners, {kWinSize, kWinSize}, {-1, -1}, term_criteria);
  for (int i = 0; i < 4; ++i) {
    quad->corners[i] = {corners[i].x, corners[i].y};
  }
}

std::vector<RawQuad> FindQuads(const std::vector<Line> &lines,
                               const std::map<int, std::set<int>> &line_connectivity,
                               const int img_width, const int img_height,
                               const double min_side_length) {
  std::vector<RawQuad> quads;
  std::set<UniqueQuad> unique_quads;
  for (const auto &[line_id, other_line_ids] : line_connectivity) {
    auto potential_quads =
        FindQuadsFromStartLine(lines, line_connectivity, line_id);  // TODO:: fix var name
    for (const auto &quad_line_ids : potential_quads) {
      const auto [itr, success] = unique_quads.insert(UniqueQuad(quad_line_ids));
      if (success) {
        auto quad = CreateQuad(quad_line_ids, lines);
        if (CheckQuadSideLengths(quad, min_side_length) &&
            CheckQuadInsideImage(quad, img_width, img_height)) {
          quads.push_back(std::move(quad));
        }
      }
    }
  }
  return quads;
}

std::vector<RawQuad> DetectQuadsInternal(const cv::Mat &img, const cv::Mat &greyscale_img,
                                         const bool debug) {
  time_logger::TimeLogger timer;

  cv::Mat blurred_img;
  cv::GaussianBlur(greyscale_img, blurred_img, cv::Size2i{kBlurKernelSize, kBlurKernelSize}, 0, 0);
  timer.logEvent("00 Image blurring");

  auto img_gradients = CalculateImageGradients(blurred_img);
  timer.logEvent("01 Image gradients");

  const auto non_max_pts = NonMaximaSuppression(img_gradients, kAbsImgGradientThresh);
  timer.logEvent("02 non max suppresion");

  const auto lines_points = ClusterGradientDirections(img_gradients, non_max_pts,
                                                      kMaxAngleClusterDiff, kMinLineClusterSize);
  timer.logEvent("03 Cluster gradients");

  const auto lines = MakeLines(lines_points, kMinLineLength);
  timer.logEvent("04 Make lines");

  const auto line_connectivity = MakeLineConnectivity(lines);
  timer.logEvent("05 Make line connectivity");

  auto quads = FindQuads(lines, line_connectivity, img.cols, img.rows, kMinLineLength);
  timer.logEvent("06 Find quads");

  std::vector<RawQuad> non_refined_quads;
  if (debug) {
    non_refined_quads = quads;
  }

  for (auto &quad : quads) {
    RefineEdges(greyscale_img, &quad);
  }
  timer.logEvent("07 subpix refine");

  if (debug) {
    timer.printLoggedEvents();

    cv::imwrite("00_blurred_img.png", blurred_img);
    VisualizeImageGradients(img_gradients);
    VisualizeNonMaxImageGradients(img_gradients, non_max_pts);
    cv::imwrite("03_line_clusters.png", VisualizeLinePoints(lines_points, img.rows, img.cols));

    const auto lines_img = VisualizeLines(img, lines);
    cv::imwrite("04_lines.png", lines_img);
    cv::imwrite("05_line_connectivity.png",
                VisualizeLineConnectivity(lines_img, lines, line_connectivity));
    cv::imwrite("06_quads.png", VisualizeQuads(img, non_refined_quads));
    cv::imwrite("07_quads_refined.png", VisualizeQuads(img, quads));
  }
  return quads;
}

}  // namespace tag_detection
