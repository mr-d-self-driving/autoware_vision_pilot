cmake_minimum_required(VERSION 3.22.1)

add_library(can_interface
        src/can_reader.cpp
        src/can_writer.cpp
)

target_include_directories(can_interface
        PUBLIC
        include
)