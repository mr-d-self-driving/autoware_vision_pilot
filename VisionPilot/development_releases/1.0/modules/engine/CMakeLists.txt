cmake_minimum_required(VERSION 3.22.1)

add_library(engine
        src/trt_engine.cpp
)

target_include_directories(engine
        PUBLIC
        include
)