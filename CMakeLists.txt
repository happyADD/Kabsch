cmake_minimum_required(VERSION 3.28)
project(Kabsch)

find_package(OpenCV REQUIRED)
link_directories(${OpenCV_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

set(CMAKE_CXX_STANDARD 17)

add_executable(Kabsch main.cpp)

target_link_libraries(Kabsch ${OpenCV_INCLUDE_DIRS})
