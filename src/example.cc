#include <iostream>
#include <opencv2/highgui.hpp>
#include <string>

#include "tag_detection/tag_detection.h"
#include "tag_detection/tag_family_lookup.h"

using namespace tag_detection;

int main(int argc, char **argv) {
  std::string image_path(argv[1]);
  cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
  if (img.empty()) {
    std::cout << "Could not read the image: " << image_path << std::endl;
    return 1;
  }

  TagFamilyLookup family(TagFamily::Tag36h11);
  const auto tags = DetectTags(img, family, 1, true);
  return 0;
}
