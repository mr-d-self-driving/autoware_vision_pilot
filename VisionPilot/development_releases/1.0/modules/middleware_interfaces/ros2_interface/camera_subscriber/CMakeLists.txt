cmake_minimum_required(VERSION 3.22.1)

add_library(camera_subscriber
        src/ros2_to_opencv.cpp
)

target_include_directories(camera_subscriber
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)