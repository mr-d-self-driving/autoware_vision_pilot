cmake_minimum_required(VERSION 3.22.1)

add_library(fusion
        src/longitudinal_fusion.cpp
        src/lateral_fusion.cpp

)

target_include_directories(fusion
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)