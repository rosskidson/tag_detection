#pragma once

#include <unordered_map>
#include <vector>

namespace tag_detection {

struct TagId {
  int id;        // Id from the tag family.
  int rotation;  // Number of 90 degree anti_clockwise rotations of the original tag.
};

// Given a family of tag codes, provide a lookup to find the id given a code. This class also
// considers all rotations of the tag and returns the rotation in addition to the id.
class TagFamilyLookup {
 public:
  TagFamilyLookup(const std::vector<unsigned long long>& family, const int tag_bits);

  bool LookupTagId(const unsigned long long& code, TagId* tag_id) const;

 private:
  std::unordered_map<unsigned long long, TagId> family_codes_{};
};

}  // namespace tag_detection
