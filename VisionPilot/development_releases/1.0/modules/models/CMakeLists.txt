cmake_minimum_required(VERSION 3.22.1)

add_library(models
        src/auto_speed.cpp
        src/auto_steer.cpp
        src/auto_drive.cpp
)

target_include_directories(models
        PUBLIC
        include
)