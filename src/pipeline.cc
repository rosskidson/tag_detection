#include <Eigen/Core>
#include <bitset>
#include <deque>
#include <iostream>
#include <map>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <set>
#include <unordered_set>

#include "tag_detection/internal/internal_types.h"
#include "tag_detection/internal/quad_detection.h"
#include "tag_detection/internal/timer.h"
#include "tag_detection/internal/visualizations.h"
#include "tag_detection/tag_detection.h"
#include "tag_detection/tag_family_lookup.h"
#include "tag_detection/types.h"

namespace tag_detection {

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

Eigen::MatrixXd ThresholdQuadBits(const cv::Mat &tag_img, const int width, const int height,
                                  const int intensity_thresh,
                                  const std::optional<std::string> &debug_filename = std::nullopt) {
  const int cell_width = tag_img.cols / width;
  const int cell_height = tag_img.rows / height;
  Eigen::MatrixXd tag_matrix(height, width);
  for (int i = 0; i < width; ++i) {
    for (int j = 0; j < height; ++j) {
      cv::Rect roi(i * cell_width, j * cell_height, cell_width, cell_height);
      cv::Mat cell(tag_img, roi);
      const int cell_val = GetMatMedian<uchar>(cell);
      tag_matrix(j, i) = cell_val > intensity_thresh;
    }
  }
  if (debug_filename.has_value()) {
    cv::Mat debug_img = tag_img.clone();
    for (int i = 0; i < width; ++i) {
      cv::line(debug_img, cv::Point2i{(i * width), 0}, cv::Point2i{(i * width), tag_img.rows},
               cv::Scalar(128));
    }
    for (int j = 0; j < height; ++j) {
      cv::line(debug_img, cv::Point2i{0, (j * height)}, cv::Point2i{tag_img.cols, (j * height)},
               cv::Scalar(128));
    }
    cv::imwrite(debug_filename.value(), debug_img);
  }
  return tag_matrix;
}

QuadWithBits ReadQuadData(const cv::Mat &img, const RawQuad &quad, const int tag_bits,
                          const int border, const bool debug) {
  const int total_tag_bits = tag_bits + (2 * border);
  std::vector<cv::Point2d> corner_pts;
  corner_pts.reserve(4);
  for (const auto &corner : quad.corners) {  // TODO:: range based/transform
    corner_pts.push_back({corner.x(), corner.y()});
  }
  const double rectified_size_x = total_tag_bits * 8;
  const double rectified_size_y = total_tag_bits * 8;
  // Points are ordered with the first point in the bottom left, then rotate around clockwise. Note
  // that this is in image coordinates, so bottom right is x = 0, y = 1.
  std::vector<cv::Point2d> rectified_pts{
      //{0, 0}, {rectified_size_x, 0}, {rectified_size_x, rectified_size_y}, {0, rectified_size_y}};
      {0, rectified_size_y},
      {rectified_size_x, rectified_size_y},
      {rectified_size_x, 0},
      {0, 0}};
  const auto H = cv::findHomography(corner_pts, rectified_pts);
  cv::Mat tag_rectified;
  cv::warpPerspective(img, tag_rectified, H, {int(rectified_size_x), int(rectified_size_y)});

  // TODO do this inside threshold, remove the param and use the tag_rectified_norm also as the
  // debug mat to save another copy.
  cv::Mat tag_rectified_norm;
  cv::normalize(tag_rectified, tag_rectified_norm, 255, 0, cv::NORM_MINMAX);

  std::optional<std::string> debug_filename{std::nullopt};
  if (debug) {
    debug_filename = "quad_" + std::to_string(quad_counter++) + ".png";
  }
  constexpr int kMagicThreshold = 128;
  const auto tag_matrix = ThresholdQuadBits(tag_rectified_norm, total_tag_bits, total_tag_bits,
                                            kMagicThreshold, debug_filename);
  return {quad.corners, tag_matrix};
}

std::vector<QuadWithBits> DetectQuadBits(const cv::Mat &img, const std::vector<RawQuad> &quads,
                                         const bool debug) {
  std::vector<QuadWithBits> quad_values;
  quad_values.reserve(quads.size());
  for (const auto &quad : quads) {
    quad_values.push_back(ReadQuadData(img, quad, 6, 1, debug));
  }
  return quad_values;
}

// Bits are read left to right, top to bottom, with the most significant bit top left.
unsigned long int ReadQuadBits(const QuadWithBits &quad, const int tag_bits, const int border) {
  const int total_tag_bits = tag_bits + (2 * border);
  // std::cout << "Quad matrix: " << std::endl << quad.bits << std::endl;
  int corrupted_border_count{};
  unsigned long int code = 0;
  int current_bit = (tag_bits * tag_bits) - 1;
  for (int j = 0; j < total_tag_bits; ++j) {
    for (int i = 0; i < total_tag_bits; ++i) {
      // Check if it is a border bit.
      if (i < border || total_tag_bits - 1 - i < border || j < border ||
          total_tag_bits - 1 - j < border) {
        if (quad.bits(j, i) != 0) {
          corrupted_border_count++;
          continue;
        }
      } else {  // Not a border bit.
        if (quad.bits(j, i) > 0) {
          code |= 1UL << current_bit;
        }
        current_bit--;
      }
    }
  }
  return code;
}

std::vector<QuadWithCode> ReadQuadBits(const std::vector<QuadWithBits> &quads) {
  std::vector<QuadWithCode> decoded_quads;
  decoded_quads.reserve(quads.size());
  int i{};
  for (const auto &quad : quads) {
    std::cout << "quad id " << i++ << std::endl;
    decoded_quads.push_back({quad.corners, ReadQuadBits(quad, 6, 1)});
    std::cout << "quad code " << std::dec << decoded_quads.back().code << std::endl;
    std::cout << "hex       " << std::hex << decoded_quads.back().code << std::endl;
    std::cout << "bin       " << std::bitset<36>(decoded_quads.back().code) << std::endl;
    std::cout << std::dec;
  }
  return decoded_quads;
}

// Rotates the corners of the quad around by an integer number of clockwise rotations.
std::array<Eigen::Vector2d, 4> RotateQuad(const std::array<Eigen::Vector2d, 4> &quad,
                                          const int rotation) {
  std::array<Eigen::Vector2d, 4> rotated_quad{};
  for (int i = 0; i < 4; ++i) {
    const int rotated_index = (i + rotation) % 4;
    rotated_quad[rotated_index] = quad[i];
  }
  return rotated_quad;
}

std::vector<Tag> SearchForTagFamilyId(const std::vector<QuadWithCode> &quads,
                                      const TagFamilyLookup &tag_family) {
  std::vector<Tag> detected_tags;
  detected_tags.reserve(quads.size());

  int i{};
  for (const auto &quad : quads) {
    TagId tag_id{};
    if (tag_family.LookupTagId(quad.code, &tag_id)) {
      std::cout << "Found Match. quad id " << i << " quad code " << std::hex << quad.code
                << std::dec << " tag id " << tag_id.id << " rotation " << tag_id.rotation
                << std::endl;
      detected_tags.push_back({RotateQuad(quad.corners, tag_id.rotation), tag_id.id});
    }
    i++;
  }
  return detected_tags;
}

void RunDetection(const cv::Mat &mat) {
  time_logger::TimeLogger full_timer;

  const bool debug = true;
  const auto quads = FindQuads(mat, debug);

  // TODO:: This is duplicated
  cv::Mat bw_mat;
  cv::cvtColor(mat, bw_mat, cv::COLOR_BGR2GRAY);

  const auto quads_with_bits = DetectQuadBits(bw_mat, quads, debug);
  // timer.logEvent("08_read quads");

  const auto quads_with_codes = ReadQuadBits(quads_with_bits);
  // timer.logEvent("09_decode quads");

  TagFamilyLookup tag_family(TagFamily::Tag36h9);
  const auto detected_tags = SearchForTagFamilyId(quads_with_codes, tag_family);
  // timer.logEvent("10_lookup tag ids");

  if (debug) {
    constexpr bool kShowCornerIndices = false;
    auto labeled_tags = VisualizeQuads(mat, detected_tags);
    for (const auto tag : detected_tags) {
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
    cv::imwrite("09_labelled_tags.png", labeled_tags);
  }

  full_timer.logEvent("everything");
  full_timer.printLoggedEvents();
}
}  // namespace tag_detection

using namespace tag_detection;

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
