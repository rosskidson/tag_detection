#pragma once

#include <Eigen/Core>
#include <array>

namespace tag_detection {

// Four corners describing a tag boundaries in image space.
struct RawQuad {
  std::array<Eigen::Vector2d, 4> corners{};
};

// Quad with the bits stored as a binary matrix.
struct UndecodedQuad {
  std::array<Eigen::Vector2d, 4> corners{};
  Eigen::MatrixXd bits{};
};

// Quad with the bits encoded into a number.
struct DecodedQuad {
  std::array<Eigen::Vector2d, 4> corners{};
  unsigned long int code{};
};

// A tag with a proper tag id from a tag family. Corners rotated according to tag orientation.
// First corner bottom right then rotate around anti clockwise.
struct Tag {
  std::array<Eigen::Vector2d, 4> corners{};
  int tag_id{};
};

}  // namespace tag_detection
