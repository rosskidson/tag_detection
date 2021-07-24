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

double rad2posdeg(double rad) {
  auto deg = rad2deg(rad);
  while (deg < 0) {
    deg += 360.0;
  }
  return deg;
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
      gradients.abs.at<double>(y, x) = std::sqrt(dx * dx + dy * dy);  // Opt: remove sqrt
      gradients.direction.at<double>(y, x) = std::atan2(dy, dx);
    }
  }
  if (debug) {
    cv::imwrite("01_img_grad.png", gradients.abs);
    cv::imwrite("02_img_dir.png", VisualizeGradientDirections(gradients));
  }
  return gradients;
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

LinePoints ClusterPoints(const Eigen::Vector2i &start_pt, const double abs_thresh,
                         const double ang_thresh, const ImageGradients &gradients,
                         std::set<Eigen::Vector2i> *processed_points) {
  LinePoints points{};
  std::deque<Eigen::Vector2i> open_points;
  open_points.push_back(start_pt);
  while (not open_points.empty()) {
    auto current_point = open_points.back();
    auto current_dir = gradients.direction.at<double>(current_point.y(), current_point.x());
    open_points.pop_back();

    points.AddPoint(current_point, current_dir);
    processed_points->insert(current_point);

    for (const auto &dir : CONNECT_FOUR) {
      const auto &candidate = current_point + dir;
      if (candidate.x() < 0 || candidate.x() >= gradients.abs.cols || candidate.y() < 0 ||
          candidate.y() >= gradients.abs.rows) {
        continue;
      }
      if (processed_points->count(candidate) == 1) {
        continue;
      }

      auto abs = gradients.abs.at<double>(candidate.y(), candidate.x());
      auto grad_dir = gradients.direction.at<double>(candidate.y(), candidate.x());
      if (abs > abs_thresh && DeltaAngle(points.GetMeanDirection(), grad_dir) < ang_thresh) {
        open_points.push_back(candidate);
      }
    }
  }

  return points;
}

std::vector<LinePoints> ClusterGradientDirections(const ImageGradients &gradients,
                                                  const double abs_thresh, const double ang_thresh,
                                                  const int min_cluster_size) {
  std::vector<LinePoints> lines;
  std::set<Eigen::Vector2i> processed_points;

  for (int y = 0; y < gradients.abs.rows; ++y) {
    for (int x = 0; x < gradients.abs.cols; ++x) {
      if (processed_points.count({x, y}) == 1) {
        continue;
      }
      const auto &abs_val = gradients.abs.at<double>(y, x);
      if (abs_val < abs_thresh) {
        continue;
      }
      const auto cluster = ClusterPoints(Eigen::Vector2i{x, y}, abs_thresh, ang_thresh, gradients,
                                         &processed_points);
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
std::vector<cv::Point2d> ToCvPoints(const EigenPointContainer &points) {
  std::vector<cv::Point2d> cv_points;
  for (const auto &pt : points) {
    cv_points.push_back({pt.x(), pt.y()});
  }
  return cv_points;
}

void RefineEdges(const cv::Mat &image, Quad *quad) {
  auto corners = ToCvPoints(quad->corners);
  auto term_criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 40, 0.001);
  cv::cornerSubPix(image, corners, {5, 5}, {-1, -1}, term_criteria);
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
  std::partial_sort(vals.begin(), vals.begin() + vals.size() / 2, vals.end());
  return vals[vals.size() / 2];
}

// TODO::rename function
Eigen::MatrixXd DecodeTag(const cv::Mat &tag_img, const int width, const int height,
                          const int border, const int intensity_thresh) {
  const int total_width = width + border;
  const int total_height = height + border;
  const int cell_width = tag_img.cols / total_width;
  const int cell_height = tag_img.rows / total_height;
  Eigen::MatrixXd tag_matrix(total_height, total_width);  // TODO:: double check constructor order
  for (int i = 0; i < total_width; ++i) {
    for (int j = 0; j < total_height; ++j) {
      cv::Rect roi(i * cell_width, j * cell_height, cell_width, cell_height);
      cv::Mat cell(tag_img, roi);
      const auto cell_val = GetMatMedian<uchar>(cell);
      tag_matrix(j, i) = cell_val > intensity_thresh;
    }
  }
  return tag_matrix;
}

int DecodeQuad(const cv::Mat &img, const Quad &quad) {
  std::vector<cv::Point2d> corner_pts;
  corner_pts.reserve(4);
  for (const auto &corner : quad.corners) {  // TODO:: range based/transform
    corner_pts.push_back({corner.x(), corner.y()});
  }
  std::vector<cv::Point2d> rectified_pts{{0, 0}, {0, 100}, {100, 100}, {100, 0}};
  const auto H = cv::findHomography(corner_pts, rectified_pts);
  cv::Mat tag_rectified;
  cv::warpPerspective(img, tag_rectified, H, {100, 100});
  // std::cout << "Quad " << quad_counter << std::endl;
  // cv::imwrite("quad_" + std::to_string(quad_counter++) + ".png", tag_rectified);
  const auto tag_matrix = DecodeTag(tag_rectified, 6, 6, 1, 128);
  // std::cout << "Quad matrix: " << std::endl << tag_matrix << std::endl;
  return -1;
}

std::vector<int> DecodeQuads(const cv::Mat &img, const std::vector<Quad> &quads) {
  std::vector<int> quad_values;
  quad_values.reserve(quads.size());
  for (const auto &quad : quads) {
    quad_values.push_back(DecodeQuad(img, quad));
  }
  return quad_values;
}

void RunDetection(const cv::Mat &mat) {
  time_logger::TimeLogger timer;
  time_logger::TimeLogger full_timer;
  const bool debug = false;
  cv::Mat bw_mat;
  cv::cvtColor(mat, bw_mat, cv::COLOR_BGR2GRAY);
  timer.logEvent("01_convert color");
  const auto img_gradients = CalculateImageGradients(bw_mat, debug);

  timer.logEvent("02_Image gradients");

  constexpr double kAbsImgGradientThresh = 100;
  constexpr double kMaxAngleClusterDiff = M_PI / 8;
  constexpr int kMinLineClusterSize = 10;

  const auto lines_points = ClusterGradientDirections(img_gradients, kAbsImgGradientThresh,
                                                      kMaxAngleClusterDiff, kMinLineClusterSize);
  timer.logEvent("03_Cluster gradients");
  std::cout << lines_points.size() << " clusters found." << std::endl;
  if (debug) {
    cv::imwrite("03_gradient_thresh.png",
                GetThresholdedGradient(img_gradients, kAbsImgGradientThresh));
    const auto viz_clusters = VisualizeLinePoints(lines_points, mat.rows, mat.cols);
    cv::imwrite("04_clusters.png", viz_clusters);
  }

  constexpr double kMinLineLength = 8;
  const auto lines = MakeLines(lines_points, kMinLineLength);
  timer.logEvent("04_Make lines");
  if (debug) {
    cv::imwrite("05_lines.png", VisualizeLines(mat, lines));
  }

  constexpr double kMaxInterLineDistance = 8;
  const auto line_connectivity = MakeLineConnectivity(lines, kMaxInterLineDistance);
  timer.logEvent("05_Make line connectivity");
  if (debug) {
    const auto base_img = VisualizeLines(mat, lines);
    cv::imwrite("06_line_connectivity.png",
                VisualizeLineConnectivity(base_img, lines, line_connectivity));
  }

  auto quads = FindQuads(lines, line_connectivity, kMinLineLength);
  timer.logEvent("06_Find quads");
  // for (auto &quad : quads) {
  //  RefineEdges(mat, &quad);
  //}
  std::cout << "Found " << quads.size() << " quads." << std::endl;
  if (debug) {
    cv::imwrite("07_quads.png", VisualizeQuads(mat, quads));
  }

  auto codes = DecodeQuads(mat, quads);
  timer.logEvent("07_Decode quads");
  timer.printLoggedEvents();
  full_timer.logEvent("everything");
  full_timer.printLoggedEvents();
}

int main() {
  std::string image_path = "cube_cal_eg.png";
  cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
  if (img.empty()) {
    std::cout << "Could not read the image: " << image_path << std::endl;
    return 1;
  }

  RunDetection(img);
  std::cout << "all good" << std::endl;
  return 0;
}
