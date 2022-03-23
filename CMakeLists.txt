cmake_minimum_required(VERSION 2.8)
project( DisplayImage )

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_library(tag_detection 
  src/quad_detection.cc
  src/quad_decoding.cc
  src/tag_detection.cc
  src/tag_family_lookup.cc
  src/utils.cc
  src/visualizations.cc
)
target_link_libraries(tag_detection ${OpenCV_LIBS} Eigen3::Eigen)

add_executable(example src/example.cc)
target_link_libraries(example tag_detection)
