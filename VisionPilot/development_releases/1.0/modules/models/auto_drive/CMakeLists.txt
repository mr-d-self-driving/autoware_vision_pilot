cmake_minimum_required(VERSION 3.22.1)

add_library(auto_drive
        src/auto_drive.cpp
)

target_include_directories(auto_drive
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)