cmake_minimum_required(VERSION 2.8)
project( DisplayImage )

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( pipeline src/pipeline.cc )
target_link_libraries( pipeline ${OpenCV_LIBS} Eigen3::Eigen)

