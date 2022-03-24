
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "tag_detection/internal/quad_decoding.h"
#include "tag_detection/internal/quad_detection.h"
#include "tag_detection/internal/timer.h"
#include "tag_detection/internal/utils.h"
#include "tag_detection/internal/visualizations.h"
#include "tag_detection/tag_family_lookup.h"
#include "tag_detection/types.h"

namespace tag_detection {

std::vector<Tag> DetectTags(const cv::Mat& img, const TagFamilyLookup& tag_family,
                            const int border_bits, const bool debug) {
  const auto greyscale_img = ToGreyscale(img);
  if (not greyscale_img.has_value()) {
    std::cout << "Failed converting input image to greyscale." << std::endl;
    return {};
  }

  const auto raw_quads = DetectQuadsInternal(img, *greyscale_img, debug);

  const int total_bits = tag_family.GetTagBits() + (2 * border_bits);
  const auto quads_with_bits = DetectQuadBits(*greyscale_img, raw_quads, total_bits, debug);

  const auto quads_with_codes =
      ReadQuadBits(quads_with_bits, tag_family.GetTagBits(), border_bits, debug);

  // Declared non const for automatic move.
  auto detections = TagDetectionsFromDecodedQuads(quads_with_codes, tag_family, debug);

  if (debug) {
    cv::imwrite("09_labelled_tags.png", VisualizeFinalDetections(img, detections));
  }

  return detections;
}

std::vector<QuadWithCode> DetectQuads(const cv::Mat& img, const int tag_bits, const int border_bits,
                                      const bool debug) {
  const auto greyscale_img = ToGreyscale(img);
  if (not greyscale_img.has_value()) {
    std::cout << "Failed converting input image to greyscale." << std::endl;
    return {};
  }

  const auto raw_quads = DetectQuadsInternal(img, *greyscale_img, debug);

  const int total_tag_bits = tag_bits + (2 * border_bits);
  const auto quads_with_bits = DetectQuadBits(*greyscale_img, raw_quads, total_tag_bits, debug);

  return ReadQuadBits(quads_with_bits, tag_bits, border_bits, debug);
}

std::vector<QuadWithBits> DetectQuads(const cv::Mat& img, const int total_tag_bits,
                                      const bool debug) {
  const auto greyscale_img = ToGreyscale(img);
  if (not greyscale_img.has_value()) {
    std::cout << "Failed converting input image to greyscale." << std::endl;
    return {};
  }

  const auto raw_quads = DetectQuadsInternal(img, *greyscale_img, debug);

  return DetectQuadBits(*greyscale_img, raw_quads, total_tag_bits, debug);
}

std::vector<RawQuad> DetectQuads(const cv::Mat& img, const bool debug) {
  const auto greyscale_img = ToGreyscale(img);
  if (not greyscale_img.has_value()) {
    std::cout << "Failed converting input image to greyscale." << std::endl;
    return {};
  }

  return DetectQuadsInternal(img, *greyscale_img, debug);
}

}  // namespace tag_detection
