cmake_minimum_required(VERSION 3.22.1)

add_library(control_cmd_publisher
        src/cmd_to_ros2.cpp
)

target_include_directories(control_cmd_publisher
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)