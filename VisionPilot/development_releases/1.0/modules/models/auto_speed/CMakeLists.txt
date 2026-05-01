cmake_minimum_required(VERSION 3.22.1)

add_library(auto_speed
        src/auto_speed.cpp
)

target_include_directories(auto_speed
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)