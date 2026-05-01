cmake_minimum_required(VERSION 3.22.1)

add_library(auto_steer
        src/auto_steer.cpp
)

target_include_directories(auto_steer
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)