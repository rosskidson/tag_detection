#include "tag_detection/tag_family_lookup.h"

#include <Eigen/Core>
#include <unordered_map>
#include <vector>

#include "tag_detection/internal/tag16h5.h"
#include "tag_detection/internal/tag25h7.h"
#include "tag_detection/internal/tag25h9.h"
#include "tag_detection/internal/tag36h11.h"
#include "tag_detection/internal/tag36h9.h"

namespace tag_detection {

std::vector<uint64_t> GenerateRotations(const uint64_t non_rotated_code, const int tag_bits) {
  std::vector<uint64_t> rotated_codes;
  // Represent the code as a matrix.
  Eigen::MatrixXd tag_matrix(tag_bits, tag_bits);
  auto code = non_rotated_code;
  for (int j = tag_bits - 1; j >= 0; --j) {
    for (int i = tag_bits - 1; i >= 0; --i) {
      tag_matrix(j, i) = code & 1;
      code >>= 1;
    }
  }

  // No rotation.
  rotated_codes.push_back(non_rotated_code);

  // 1 90 degree anti-clockwise rotation of the original tag.
  {
    uint64_t code = 0;
    int current_bit = (tag_bits * tag_bits) - 1;
    for (int i = 0; i < tag_bits; ++i) {
      for (int j = tag_bits - 1; j >= 0; --j) {
        if (tag_matrix(j, i) > 0) {
          code |= 1UL << current_bit;
        }
        current_bit--;
      }
    }
    rotated_codes.push_back(code);
  }

  // 2 90 degree anti-clockwise rotations of the original tag.
  {
    uint64_t code = 0;
    int current_bit = (tag_bits * tag_bits) - 1;
    for (int j = tag_bits - 1; j >= 0; --j) {
      for (int i = tag_bits - 1; i >= 0; --i) {
        if (tag_matrix(j, i) > 0) {
          code |= 1UL << current_bit;
        }
        current_bit--;
      }
    }
    rotated_codes.push_back(code);
  }

  // 3 90 degree anti-clockwise rotations of the original tag.
  {
    uint64_t code = 0;
    int current_bit = (tag_bits * tag_bits) - 1;
    for (int i = tag_bits - 1; i >= 0; --i) {
      for (int j = 0; j < tag_bits; ++j) {
        if (tag_matrix(j, i) > 0) {
          code |= 1UL << current_bit;
        }
        current_bit--;
      }
    }
    rotated_codes.push_back(code);
  }

  return rotated_codes;
}

int GetFamilySize(const TagFamily& family) {
  switch (family) {
    case TagFamily::Tag16h5:
      return 4;
    case TagFamily::Tag25h7:
      return 5;
    case TagFamily::Tag25h9:
      return 5;
    case TagFamily::Tag36h9:
      return 6;
    case TagFamily::Tag36h11:
      return 6;
  }
  return {};
}

std::vector<uint64_t> GetFamilyCodes(const TagFamily& family) {
  switch (family) {
    case TagFamily::Tag16h5:
      return t16h5;
    case TagFamily::Tag25h7:
      return t25h7;
    case TagFamily::Tag25h9:
      return t25h9;
    case TagFamily::Tag36h9:
      return t36h9;
    case TagFamily::Tag36h11:
      return t36h11;
  }
  return {};
}

TagFamilyLookup::TagFamilyLookup(const TagFamily& family)
    : TagFamilyLookup(GetFamilyCodes(family), GetFamilySize(family)) {}

TagFamilyLookup::TagFamilyLookup(const std::vector<uint64_t>& family, const int tag_bits)
    : tag_bits_(tag_bits) {
  for (int id = 0; id < family.size(); ++id) {
    const auto rotations = GenerateRotations(family[id], tag_bits);
    for (int r = 0; r < rotations.size(); ++r) {
      family_codes_.insert({rotations[r], {id, r}});
    }
  }
}

bool TagFamilyLookup::LookupTagId(const uint64_t& code, TagId* tag_id) const {
  if (family_codes_.count(code)) {
    *tag_id = family_codes_.at(code);
    return true;
  } else {
    return false;
  }
}

int TagFamilyLookup::GetTagBits() const {
  return tag_bits_;
}

}  // namespace tag_detection
