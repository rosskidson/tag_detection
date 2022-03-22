#pragma once

#include <opencv2/core/mat.hpp>
#include <vector>

#include "tag_detection/tag_family_lookup.h"
#include "tag_detection/types.h"

namespace tag_detection {

/**
 *  Detect tags in an image.
 *
 *  @param img         The input image
 *  @param tag_family  A tag family lookup object.
 *  @param debug       Generate debug images if set to true.
 *
 *  @return  A vector of tag detections.
 *
 */
std::vector<Tag> DetectTags(const cv::Mat& img, const TagFamilyLookup& tag_family,
                            const bool debug = false);

/**
 * Detect candidate quads in an image and the tag codes. This may be used if the raw tag value is
 * required.
 *
 *  @param img         The input image.
 *  @param tag_bits    The number of bits (side length) of the tag not including border.
 *  @param border_bits How many border bits.
 *  @param debug       Generate debug images if set to true.
 *
 *  @return  A vector of quad detections containing a quad corner and the tag code.
 *
 */
// std::vector<DecodedQuad> DetectQuads(const cv::Mat& img, const int tag_bits, const int border,
//                                     const bool debug = false);

/**
 * Detect candidate quads in an image and the tag bit information. This may be used if the raw tag
 * bits are required, for instance, for reading the bits in a custom order.
 *
 *  @param img            The input image.
 *  @param total_tag_bits The number of bits (side length) of the tag including the border.
 *  @param debug          Generate debug images if set to true.
 *
 *  @return  A vector of quad detections containing a quad corner and a matrix of the tag bits.
 *
 */
// std::vector<UndecodedQuad> DetectUndecodedQuads(const cv::Mat& img, const int total_tag_bits,
//                                                const bool debug = false);

}  // namespace tag_detection
