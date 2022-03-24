#pragma once

#include <opencv2/core.hpp>
#include <vector>

#include "tag_detection/tag_family_lookup.h"
#include "tag_detection/types.h"

namespace tag_detection {

std::vector<QuadWithBits> DetectQuadBits(const cv::Mat &greyscale_img,
                                         const std::vector<RawQuad> &quads, int total_tag_bits,
                                         bool debug);

std::vector<QuadWithCode> ReadQuadBits(const std::vector<QuadWithBits> &quads, int tag_bits,
                                       int border_bits, bool debug);

std::vector<Tag> TagDetectionsFromDecodedQuads(const std::vector<QuadWithCode> &quads,
                                               const TagFamilyLookup &tag_family, bool debug);

}  // namespace tag_detection
