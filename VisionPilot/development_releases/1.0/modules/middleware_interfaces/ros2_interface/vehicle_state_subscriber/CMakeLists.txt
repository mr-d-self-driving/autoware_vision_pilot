cmake_minimum_required(VERSION 3.22.1)

add_library(vehicle_state_subscriber
        src/ros2_to_can.cpp
)

target_include_directories(vehicle_state_subscriber
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)