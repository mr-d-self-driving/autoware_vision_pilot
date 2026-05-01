cmake_minimum_required(VERSION 3.22.1)

add_library(v4l2_interface
        src/v4l2_reader.cpp
)

target_include_directories(v4l2_interface
        PUBLIC
        include
)