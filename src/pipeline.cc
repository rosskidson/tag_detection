#include <Eigen/Core>
#include <deque>
#include <iostream>
#include <map>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <set>
#include <unordered_set>

#include "tag_detection/timer.h"

double rad2deg(double rad) {
  return rad * 180.0 / M_PI;
}

int rad2posdeg(double rad) {
  auto deg = rad2deg(rad);
  while (deg < 0) {
    deg += 360.0;
  }
  return deg;
}

double rad2posrad(double rad) {
  constexpr double kTwoPi = 2 * 3.14159265358979323;
  while (rad < 0) {
    rad += kTwoPi;
  }
  return rad;
}

struct ImageGradients {
  cv::Mat abs;
  cv::Mat direction;
};

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
  std::cout << " max grad " << max_gradient << std::endl;

  cv::Mat grad_dir_viz(gradients.abs.rows, gradients.abs.cols, CV_8UC3, cv::Scalar::all(0));
  for (int y = 0; y < gradients.direction.rows; ++y) {
    for (int x = 0; x < gradients.direction.cols; ++x) {
      cv::Vec3f hsv{float(rad2posdeg(gradients.direction.at<double>(y, x))),  //
                    1.0,                                                      //
                    float(std::min(gradients.abs.at<double>(y, x) / max_gradient, 1.0))};
      auto &bgr = grad_dir_viz.at<cv::Vec3b>(y, x);
      bgr = HSVtoBGR(hsv);
    }
  }
  return grad_dir_viz;
}

// TODO:: Missing: Non maxima suppression
ImageGradients CalculateImageGradients(const cv::Mat &mat, const bool debug) {
  ImageGradients gradients;
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
  if (debug) {
    cv::imwrite("01_img_grad.png", gradients.abs);
    cv::imwrite("02_img_dir.png", VisualizeGradientDirections(gradients));
  }
  return gradients;
}

// const std::vector<std::vector<Eigen::Vector2i>> gradient_lookup{
//    {Eigen::Vector2i{-1, 0}, Eigen::Vector2i{1, 0}},   // 0 - 22
//    {Eigen::Vector2i{-1, -1}, Eigen::Vector2i{1, 1}},  // 22 - 45
//    {Eigen::Vector2i{-1, -1}, Eigen::Vector2i{1, 1}},  // 45 - 77
//    {Eigen::Vector2i{0, 1}, Eigen::Vector2i{0, -1}},   // 77 - 90
//    {Eigen::Vector2i{0, 1}, Eigen::Vector2i{0, -1}},   // 90 - 112
//    {Eigen::Vector2i{-1, 1}, Eigen::Vector2i{1, -1}},  // 112 - 135
//    {Eigen::Vector2i{-1, 1}, Eigen::Vector2i{1, -1}},  // 135 - 157
//    {Eigen::Vector2i{-1, 0}, Eigen::Vector2i{1, 0}},   // 157 - 180
//    {Eigen::Vector2i{-1, 0}, Eigen::Vector2i{1, 0}},   // 180 - 202
//    {Eigen::Vector2i{-1, -1}, Eigen::Vector2i{1, 1}},  // 202 - 225
//    {Eigen::Vector2i{-1, -1}, Eigen::Vector2i{1, 1}},  // 225 - 247
//    {Eigen::Vector2i{0, 1}, Eigen::Vector2i{0, -1}},   // 247 - 270
//    {Eigen::Vector2i{0, 1}, Eigen::Vector2i{0, -1}},   // 270 - 292
//    {Eigen::Vector2i{-1, 1}, Eigen::Vector2i{1, -1}},  // 292 - 315
//    {Eigen::Vector2i{-1, 1}, Eigen::Vector2i{1, -1}},  // 315 - 337
//    {Eigen::Vector2i{-1, 0}, Eigen::Vector2i{1, 0}}};  // 337 - 360

const std::vector<std::vector<Eigen::Vector2i>> gradient_lookup{
    {Eigen::Vector2i{-1, 0}, Eigen::Vector2i{1, 0}},   // 0
    {Eigen::Vector2i{-1, -1}, Eigen::Vector2i{1, 1}},  // 45
    {Eigen::Vector2i{0, 1}, Eigen::Vector2i{0, -1}},   // 90
    {Eigen::Vector2i{-1, 1}, Eigen::Vector2i{1, -1}},  // 135
    {Eigen::Vector2i{-1, 0}, Eigen::Vector2i{1, 0}},   // 180
    {Eigen::Vector2i{-1, -1}, Eigen::Vector2i{1, 1}},  // 225
    {Eigen::Vector2i{0, 1}, Eigen::Vector2i{0, -1}},   // 270
    {Eigen::Vector2i{-1, 1}, Eigen::Vector2i{1, -1}},  // 315
    {Eigen::Vector2i{-1, 0}, Eigen::Vector2i{1, 0}}};  // 360

cv::Mat NonMaximaSuppression(ImageGradients *gradients, const double min_abs_gradient,
                             const bool debug) {
  // 0 = max point, 1 = non max point
  cv::Mat non_max_pts(gradients->abs.rows, gradients->abs.cols, CV_8U, cv::Scalar(0));

  non_max_pts.reserve(gradients->abs.cols * gradients->abs.rows);
  for (int y = 1; y < gradients->abs.rows - 1; ++y) {
    for (int x = 1; x < gradients->abs.cols - 1; ++x) {
      auto &current_val = gradients->abs.at<double>(y, x);
      if (current_val < min_abs_gradient) {
        non_max_pts.at<uint8_t>(y, x) = 1;
        continue;
      }
      const double direction_deg = rad2posdeg(gradients->direction.at<double>(y, x));
      assert(direction_deg <= 360 && direction_deg >= 0);
      const int index = std::round(direction_deg / 45.0);
      const auto &directions = gradient_lookup[index];
      const auto &val_0 = gradients->abs.at<double>(y + directions[0].y(), x + directions[0].x());
      const auto &val_1 = gradients->abs.at<double>(y + directions[1].y(), x + directions[1].x());
      if (val_0 > current_val || val_1 > current_val) {
        non_max_pts.at<uint8_t>(y, x) = 1;
      }
    }
  }

  if (debug) {
    auto abs_viz = gradients->abs.clone();
    auto grad_viz = gradients->direction.clone();
    for (int y = 0; y < non_max_pts.rows; ++y) {
      for (int x = 0; x < non_max_pts.cols; ++x) {
        if (non_max_pts.at<uint8_t>(y, x) > 0) {
          abs_viz.at<double>(y, x) = 0;
          grad_viz.at<double>(y, x) = 0;
        }
      }
    }
    cv::imwrite("01a_img_grad.png", abs_viz);
    cv::imwrite("02b_img_dir.png", VisualizeGradientDirections({abs_viz, grad_viz}));
  }
  return non_max_pts;
}

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

const std::vector<Eigen::Vector2i> CONNECT_FOUR{{-1, 0}, {0, -1}, {1, 0}, {0, 1}};
const std::vector<Eigen::Vector2i> CONNECT_EIGHT{{-1, -1}, {0, -1}, {1, -1}, {1, 0},
                                                 {1, 1},   {0, 1},  {-1, 1}, {-1, 0}};

namespace Eigen {
bool operator<(const Eigen::Vector2i &a, const Eigen::Vector2i &b) {
  return a.x() < b.x() || (a.x() == b.x() && a.y() < b.y());
}
}  // namespace Eigen

double DeltaAngle(const double ang_1, const double ang_2) {
  auto delta = std::abs(ang_1 - ang_2);
  if (delta > M_PI) {
    delta -= 2 * M_PI;
  }
  return std::abs(delta);
}

LinePoints ClusterPoints(const Eigen::Vector2i &start_pt, const double ang_thresh,
                         const ImageGradients &gradients, cv::Mat *processed_points) {
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

      // auto abs = gradients.abs.at<double>(candidate.y(), candidate.x());
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

cv::Mat GetThresholdedGradient(const ImageGradients &img_gradients, const double threshold) {
  cv::Mat grad_abs_thresh(img_gradients.abs.rows, img_gradients.abs.cols, CV_8U, cv::Scalar(0));
  for (int y = 0; y < grad_abs_thresh.rows; ++y) {
    for (int x = 0; x < grad_abs_thresh.cols; ++x) {
      if (img_gradients.abs.at<double>(y, x) > threshold) {
        grad_abs_thresh.at<uchar>(y, x) = 255;
      }
    }
  }
  return grad_abs_thresh;
}

cv::Mat VisualizeLinePoints(const std::vector<LinePoints> &lines, const int rows, const int cols) {
  cv::Mat viz_clusters(rows, cols, CV_8UC3, cv::Scalar::all(0));
  for (const auto &line : lines) {
    const float rand_h = rand() % 360;
    const auto bgr = HSVtoBGR({rand_h, 1.0, 1.0});
    for (const auto &point : line.Points()) {
      viz_clusters.at<cv::Vec3b>(point.y(), point.x()) = bgr;
    }
  }
  return viz_clusters;
}

struct Line {
  Eigen::Vector2i start;
  Eigen::Vector2i end;
};

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

cv::Mat VisualizeLines(const cv::Mat &img, const std::vector<Line> &lines) {
  cv::Mat viz_lines = img.clone();
  for (const auto &line : lines) {
    cv::line(viz_lines, {line.start.x(), line.start.y()}, {line.end.x(), line.end.y()}, {0, 255, 0},
             1);
  }
  return viz_lines;
}

bool LinesAreConnected(const Line &line_a, const Line &line_b, const double squared_distance) {
  return (line_a.start - line_b.start).squaredNorm() < squared_distance ||
         (line_a.start - line_b.end).squaredNorm() < squared_distance ||
         (line_a.end - line_b.start).squaredNorm() < squared_distance ||
         (line_a.end - line_b.end).squaredNorm() < squared_distance;
}

struct LineEnds {
  Eigen::Vector2i line_end_a;
  Eigen::Vector2i line_end_b;
};

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

std::map<int, std::set<int>> MakeLineConnectivity(const std::vector<Line> &lines,
                                                  const double &distance) {
  const auto distance_squared = distance * distance;
  std::map<int, std::set<int>> line_connectivity;
  for (int i = 0; i < lines.size(); ++i) {
    for (int j = i + 1; j < lines.size(); ++j) {
      if (LinesAreConnected(lines[i], lines[j], distance_squared)) {
        line_connectivity[i].insert(j);
        line_connectivity[j].insert(i);
      }
    }
  }
  return line_connectivity;
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

struct Quad {
  std::array<Eigen::Vector2d, 4> corners{};
};

// TODO:: Missing:: subpix refine
Quad CreateQuad(const std::vector<int> &quad_line_ids, const std::vector<Line> &lines) {
  Quad quad{};
  for (int i = 0; i < 4; ++i) {
    auto line_ends =
        GetConnectedLineEnds(lines[quad_line_ids[i]], lines[quad_line_ids[(i + 1) % 4]]);
    quad.corners[i] =
        (line_ends.line_end_a.cast<double>() + line_ends.line_end_b.cast<double>()) / 2.0;
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
    angle = std::atan2(centroid.y() - corner.y(), centroid.x() - corner.x());
    if (angle < 0) {
      angle += 2 * M_PI;
    }
  }

  // Sort the points by angle, descending to get them in clockwise order.
  std::sort(corners.begin(), corners.end(), [](const auto &corner_1, const auto &corner_2) {
    return corner_1.angle > corner_2.angle;
  });

  for (int i = 0; i < 4; ++i) {
    quad.corners[i] = corners[i].corner;
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

bool CheckQuadSideLengths(const Quad &quad, const double side_length) {
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

template <typename EigenPointContainer>
std::vector<cv::Point2f> ToCvPoints(const EigenPointContainer &points) {
  std::vector<cv::Point2f> cv_points;
  for (const auto &pt : points) {
    cv_points.push_back({float(pt.x()), float(pt.y())});
  }
  return cv_points;
}

void RefineEdges(const cv::Mat &image, Quad *quad) {
  constexpr int kWinSize = 2;
  auto corners = ToCvPoints(quad->corners);
  auto term_criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 40, 0.001);
  cv::cornerSubPix(image, corners, {kWinSize, kWinSize}, {-1, -1}, term_criteria);
  for (int i = 0; i < 4; ++i) {
    quad->corners[i] = {corners[i].x, corners[i].y};
  }
}

std::vector<Quad> FindQuads(const std::vector<Line> &lines,
                            const std::map<int, std::set<int>> &line_connectivity,
                            const double min_side_length) {
  std::vector<Quad> quads;
  std::set<UniqueQuad> unique_quads;
  for (const auto &[line_id, other_line_ids] : line_connectivity) {
    auto potential_quads =
        FindQuadsFromStartLine(lines, line_connectivity, line_id);  // TODO:: fix var name
    for (const auto &quad_line_ids : potential_quads) {
      const auto [itr, success] = unique_quads.insert(UniqueQuad(quad_line_ids));
      if (success) {
        auto quad = CreateQuad(quad_line_ids, lines);
        if (CheckQuadSideLengths(quad, min_side_length)) {
          quads.push_back(std::move(quad));
        }
      }
    }
  }
  return quads;
}

cv::Mat VisualizeQuads(const cv::Mat &img, const std::vector<Quad> &quads) {
  cv::Mat viz = img.clone();
  int quad_counter{};
  for (const auto &quad : quads) {
    for (int i = 0; i < 4; ++i) {
      // std::cout << "  " << quad.corners[i].x() << ", " << quad.corners[i].y() << std::endl;
      const auto pt_a = quad.corners[i].cast<int>();
      const auto pt_b = quad.corners[(i + 1) % 4].cast<int>();
      cv::line(viz, {pt_a.x(), pt_a.y()}, {pt_b.x(), pt_b.y()}, {0, 255, 0}, 1);
    }
  }
  return viz;
}

int quad_counter = 0;

template <typename T>
T GetMatMedian(const cv::Mat &mat) {
  std::vector<T> vals;
  vals.reserve(mat.cols * mat.rows);
  for (int y = 0; y < mat.rows; ++y) {
    for (int x = 0; x < mat.cols; ++x) {
      vals.push_back(mat.at<T>(y, x));
    }
  }
  std::nth_element(vals.begin(), vals.begin() + (vals.size() / 2), vals.end());
  return vals[vals.size() / 2];
}

// TODO::rename function
Eigen::MatrixXd DecodeTag(const cv::Mat &tag_img, const int width, const int height,
                          const int intensity_thresh) {
  const int cell_width = tag_img.cols / width;
  const int cell_height = tag_img.rows / height;
  Eigen::MatrixXd tag_matrix(height, width);  // TODO:: double check constructor order
  for (int i = 0; i < width; ++i) {
    for (int j = 0; j < height; ++j) {
      cv::Rect roi(i * cell_width, j * cell_height, cell_width, cell_height);
      cv::Mat cell(tag_img, roi);
      const int cell_val = GetMatMedian<uchar>(cell);
      tag_matrix(j, i) = cell_val > intensity_thresh;
    }
  }
  return tag_matrix;
}

int DecodeQuad(const cv::Mat &img, const Quad &quad, const int tag_bits, const int border) {
  const int total_tag_bits = tag_bits + (2 * border);
  std::vector<cv::Point2d> corner_pts;
  corner_pts.reserve(4);
  for (const auto &corner : quad.corners) {  // TODO:: range based/transform
    corner_pts.push_back({corner.x(), corner.y()});
  }
  const double rectified_size_x = total_tag_bits * 8;
  const double rectified_size_y = total_tag_bits * 8;
  std::vector<cv::Point2d> rectified_pts{
      {0, 0}, {0, rectified_size_y}, {rectified_size_x, rectified_size_y}, {rectified_size_x, 0}};
  const auto H = cv::findHomography(corner_pts, rectified_pts);
  cv::Mat tag_rectified;
  cv::warpPerspective(img, tag_rectified, H, {int(rectified_size_x), int(rectified_size_y)});
  // std::cout << "Quad " << quad_counter << std::endl;
  // cv::imwrite("quad_" + std::to_string(quad_counter++) + ".png", tag_rectified);

  constexpr int kMagicThreshold = 110;
  const auto tag_matrix = DecodeTag(tag_rectified, total_tag_bits, total_tag_bits, kMagicThreshold);
  // std::cout << "Quad matrix: " << std::endl << tag_matrix << std::endl;
  int corrupted_border_count{};
  unsigned int code = 0;
  int current_bit = (tag_bits * tag_bits);
  for (int i = 0; i < total_tag_bits; ++i) {
    for (int j = 0; j < total_tag_bits; ++j) {
      if (i < border || total_tag_bits - 1 - i < border || j < border ||
          total_tag_bits - 1 - j < border) {
        if (tag_matrix(j, i) != 0) {
          corrupted_border_count++;
          continue;
        }
      } else {  // Not a border pixel
        if (tag_matrix(j, i) > 0) {
          code |= 1UL << current_bit;
        }
        current_bit--;
      }
    }
  }

  std::cout << "corruption count: " << corrupted_border_count << std::endl;
  std::cout << "code              " << code << std::endl;
  return code;
}

std::vector<int> DecodeQuads(const cv::Mat &img, const std::vector<Quad> &quads) {
  std::vector<int> quad_values;
  quad_values.reserve(quads.size());
  for (const auto &quad : quads) {
    quad_values.push_back(DecodeQuad(img, quad, 6, 1));
  }
  return quad_values;
}

void RunDetection(const cv::Mat &mat) {
  time_logger::TimeLogger timer;
  time_logger::TimeLogger full_timer;
  const bool debug = true;
  cv::Mat bw_mat;
  cv::cvtColor(mat, bw_mat, cv::COLOR_BGR2GRAY);
  timer.logEvent("01_convert color");
  auto img_gradients = CalculateImageGradients(bw_mat, debug);

  timer.logEvent("02_Image gradients");

  constexpr double kAbsImgGradientThresh = 100;  // * 100;
  const auto non_max_pts = NonMaximaSuppression(&img_gradients, kAbsImgGradientThresh, debug);

  timer.logEvent("03_non max ");

  constexpr double kMaxAngleClusterDiff = M_PI / 8;
  constexpr int kMinLineClusterSize = 5;

  const auto lines_points = ClusterGradientDirections(img_gradients, non_max_pts,
                                                      kMaxAngleClusterDiff, kMinLineClusterSize);
  timer.logEvent("04_Cluster gradients");
  std::cout << lines_points.size() << " clusters found." << std::endl;
  if (debug) {
    cv::imwrite("03_gradient_thresh.png",
                GetThresholdedGradient(img_gradients, kAbsImgGradientThresh));
    const auto viz_clusters = VisualizeLinePoints(lines_points, mat.rows, mat.cols);
    cv::imwrite("04_clusters.png", viz_clusters);
  }

  constexpr double kMinLineLength = 8;
  const auto lines = MakeLines(lines_points, kMinLineLength);
  timer.logEvent("05_Make lines");
  if (debug) {
    cv::imwrite("05_lines.png", VisualizeLines(mat, lines));
  }

  constexpr double kMaxInterLineDistance = 4;
  const auto line_connectivity = MakeLineConnectivity(lines, kMaxInterLineDistance);
  timer.logEvent("06_Make line connectivity");
  if (debug) {
    const auto base_img = VisualizeLines(mat, lines);
    cv::imwrite("06_line_connectivity.png",
                VisualizeLineConnectivity(base_img, lines, line_connectivity));
  }

  auto quads = FindQuads(lines, line_connectivity, kMinLineLength);
  timer.logEvent("07_Find quads");
  if (debug) {
    cv::imwrite("07_quads.png", VisualizeQuads(mat, quads));
  }
  cv::Mat mat_bw;
  cv::cvtColor(mat, mat_bw, cv::COLOR_BGR2GRAY);
  for (auto &quad : quads) {
    RefineEdges(mat_bw, &quad);
  }
  std::cout << "Found " << quads.size() << " quads." << std::endl;
  if (debug) {
    cv::imwrite("08_quads_refined.png", VisualizeQuads(mat, quads));
  }

  auto codes = DecodeQuads(bw_mat, quads);
  timer.logEvent("09_Decode quads");
  timer.printLoggedEvents();
  full_timer.logEvent("everything");
  full_timer.printLoggedEvents();
}

// TODO
//
// 2. Fit lines to get line equation. Edges from line intersections
//    This is unlikely to change the result after subpix refine, however it _might_ be better
//    than a refined corner.
//

int main(int argc, char **argv) {
  std::string image_path(argv[1]);
  cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
  if (img.empty()) {
    std::cout << "Could not read the image: " << image_path << std::endl;
    return 1;
  }

  RunDetection(img);
  std::cout << "all good" << std::endl;
  return 0;
}
