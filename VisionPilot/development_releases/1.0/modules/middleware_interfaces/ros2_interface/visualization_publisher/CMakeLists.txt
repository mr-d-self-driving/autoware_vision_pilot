cmake_minimum_required(VERSION 3.22.1)

add_library(visualization_publisher
        src/visualization_publisher.cpp
)

target_include_directories(visualization_publisher
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)