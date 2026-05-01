cmake_minimum_required(VERSION 3.22.1)

add_library(planning
        src/longitudinal_planning.cpp
        src/lateral_planning.cpp

)

target_include_directories(planning
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)