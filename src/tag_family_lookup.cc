#include "tag_detection/tag_family_lookup.h"

#include <Eigen/Core>
#include <unordered_map>
#include <vector>

namespace tag_detection {

std::vector<unsigned long long int> GenerateRotations(const unsigned long long int non_rotated_code,
                                                      const int tag_bits) {
  std::vector<unsigned long long int> rotated_codes;
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
    unsigned long long int code = 0;
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
    unsigned long long int code = 0;
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
    unsigned long long int code = 0;
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

TagFamilyLookup::TagFamilyLookup(const std::vector<unsigned long long>& family,
                                 const int tag_bits) {
  for (int id = 0; id < family.size(); ++id) {
    const auto rotations = GenerateRotations(family[id], tag_bits);
    for (int r = 0; r < rotations.size(); ++r) {
      family_codes_.insert({rotations[r], {id, r}});
    }
  }
}

bool TagFamilyLookup::LookupTagId(const unsigned long long& code, TagId* tag_id) const {
  if (family_codes_.count(code)) {
    *tag_id = family_codes_.at(code);
    return true;
  } else {
    return false;
  }
}

}  // namespace tag_detection
