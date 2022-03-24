#include <Eigen/Core>
#include <algorithm>
#include <bitset>
#include <iomanip>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <vector>

#include "tag_detection/tag_family_lookup.h"
#include "tag_detection/types.h"

namespace tag_detection {

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
                                  const std::optional<std::string> &debug_filename = std::nullopt) {
  cv::Mat tag_img_norm;
  cv::normalize(tag_img, tag_img_norm, 255, 0, cv::NORM_MINMAX);
  // After normalization to 0-255, no threshold tuning is required, 128 works well.
  constexpr int kThreshold = 128;

  const int cell_width = tag_img_norm.cols / width;
  const int cell_height = tag_img_norm.rows / height;
  Eigen::MatrixXd tag_matrix(height, width);
  for (int i = 0; i < width; ++i) {
    for (int j = 0; j < height; ++j) {
      cv::Rect roi(i * cell_width, j * cell_height, cell_width, cell_height);
      cv::Mat cell(tag_img_norm, roi);
      const int cell_val = GetMatMedian<uchar>(cell);
      tag_matrix(j, i) = cell_val > kThreshold;
    }
  }
  if (debug_filename.has_value()) {
    for (int i = 0; i < width; ++i) {
      cv::line(tag_img_norm, cv::Point2i{(i * cell_width), 0},
               cv::Point2i{(i * cell_width), tag_img.rows}, cv::Scalar(128));
    }
    for (int j = 0; j < height; ++j) {
      cv::line(tag_img_norm, cv::Point2i{0, (j * cell_height)},
               cv::Point2i{tag_img.cols, (j * cell_height)}, cv::Scalar(128));
    }
    cv::imwrite(debug_filename.value(), tag_img_norm);
  }
  return tag_matrix;
}

QuadWithBits DetectQuadBits(const cv::Mat &greyscale_img, const RawQuad &quad,
                            const int total_tag_bits, const std::optional<int> &debug_quad_id) {
  std::vector<cv::Point2d> corner_pts;
  corner_pts.reserve(4);
  for (const auto &corner : quad.corners) {
    corner_pts.emplace_back(corner.x(), corner.y());
  }
  const double rectified_size_x = total_tag_bits * 8;
  const double rectified_size_y = total_tag_bits * 8;
  // Points are ordered with the first point in the bottom left, then rotate around clockwise. Note
  // that 'bottom right' refers to the position on a rendered image, and this is in image
  // coordinates (y axis inverted), so bottom right is x = 0, y = 1.
  std::vector<cv::Point2d> rectified_pts{
      {0, rectified_size_y}, {rectified_size_x, rectified_size_y}, {rectified_size_x, 0}, {0, 0}};
  const auto H = cv::findHomography(corner_pts, rectified_pts);
  cv::Mat tag_rectified;
  cv::warpPerspective(greyscale_img, tag_rectified, H,
                      {int(rectified_size_x), int(rectified_size_y)});

  std::optional<std::string> debug_filename{std::nullopt};
  if (debug_quad_id.has_value()) {
    static int quad_counter{0};
    std::stringstream ss;
    ss << "quad_" << std::setw(4) << std::setfill('0') << *debug_quad_id << ".png";
    debug_filename = ss.str();
  }
  const auto tag_matrix =
      ThresholdQuadBits(tag_rectified, total_tag_bits, total_tag_bits, debug_filename);

  return {quad.corners, tag_matrix};
}

std::vector<QuadWithBits> DetectQuadBits(const cv::Mat &greyscale_img,
                                         const std::vector<RawQuad> &quads,
                                         const int total_tag_bits, const bool debug) {
  std::vector<QuadWithBits> quad_values;
  quad_values.reserve(quads.size());
  int quad_counter{};
  for (const auto &quad : quads) {
    const std::optional<int> debug_id = (debug ? quad_counter : std::optional<int>(std::nullopt));
    quad_counter++;
    quad_values.push_back(DetectQuadBits(greyscale_img, quad, total_tag_bits, debug_id));
  }
  return quad_values;
}

// Bits are read left to right, top to bottom, with the most significant bit top left.
uint64_t ReadQuadBits(const QuadWithBits &quad, const int tag_bits, const int border_bits) {
  const int total_tag_bits = tag_bits + (2 * border_bits);
  int corrupted_border_count{};
  uint64_t code = 0;
  uint current_bit = (tag_bits * tag_bits) - 1;
  for (int j = 0; j < total_tag_bits; ++j) {
    for (int i = 0; i < total_tag_bits; ++i) {
      // Check if it is a border bit.
      if (i < border_bits || total_tag_bits - 1 - i < border_bits || j < border_bits ||
          total_tag_bits - 1 - j < border_bits) {
        if (quad.bits(j, i) != 0) {
          corrupted_border_count++;
          continue;
        }
      } else {  // This is not a border bit.
        if (quad.bits(j, i) > 0) {
          code |= 1UL << current_bit;
        }
        current_bit--;
      }
    }
  }
  return code;
}

std::vector<QuadWithCode> ReadQuadBits(const std::vector<QuadWithBits> &quads, const int tag_bits,
                                       const int border_bits, bool debug) {
  std::vector<QuadWithCode> decoded_quads;
  decoded_quads.reserve(quads.size());
  int quad_counter{};
  for (const auto &quad : quads) {
    decoded_quads.push_back({quad.corners, ReadQuadBits(quad, tag_bits, border_bits)});
    if (debug) {
      std::cout << "quad id " << quad_counter++ << std::endl;
      std::cout << "quad matrix: \n" << quad.bits << std::endl;
      std::cout << "quad code " << std::dec << decoded_quads.back().code << std::endl;
      std::cout << "hex       " << std::hex << decoded_quads.back().code << std::endl;
      std::cout << "bin       " << std::bitset<36>(decoded_quads.back().code) << std::endl;
      std::cout << std::dec;
    }
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

std::vector<Tag> TagDetectionsFromDecodedQuads(const std::vector<QuadWithCode> &quads,
                                               const TagFamilyLookup &tag_family,
                                               const bool debug) {
  std::vector<Tag> detected_tags;
  detected_tags.reserve(quads.size());

  for (const auto &quad : quads) {
    TagId tag_id{};
    if (tag_family.LookupTagId(quad.code, &tag_id)) {
      detected_tags.push_back({RotateQuad(quad.corners, tag_id.rotation), tag_id.id});
      if (debug) {
        std::cout << "Tag detected. ID: " << tag_id.id << " rotation: " << tag_id.rotation
                  << " original code: " << quad.code << std::endl;
      }
    }
  }
  return detected_tags;
}

}  // namespace tag_detection
