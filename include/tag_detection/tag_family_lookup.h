#pragma once

#include <unordered_map>
#include <vector>

namespace tag_detection {

struct TagId {
  int id;        // Id from the tag family.
  int rotation;  // Number of 90 degree anti_clockwise rotations of the original tag.
};

enum class TagFamily { Tag16h5, Tag25h7, Tag25h9, Tag36h9, Tag36h11 };

// Given a family of tag codes, provide a lookup to find the id given a code. This class also
// considers all rotations of the tag and returns the rotation in addition to the id.
class TagFamilyLookup {
 public:
  explicit TagFamilyLookup(const TagFamily& family);
  TagFamilyLookup(const std::vector<uint64_t>& family, int tag_bits);

  bool LookupTagId(const uint64_t& code, TagId* tag_id) const;

  int GetTagBits() const;

 private:
  std::unordered_map<uint64_t, TagId> family_codes_{};
  int tag_bits_{};
};

}  // namespace tag_detection
