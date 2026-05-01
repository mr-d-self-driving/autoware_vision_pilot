cmake_minimum_required(VERSION 3.22.1)

add_library(control
        src/longitudinal_control.cpp
        src/lateral_control.cpp
)

target_include_directories(control
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)