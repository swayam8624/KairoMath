cmake_minimum_required(VERSION 4.2)
project(Math)

set(CMAKE_CXX_STANDARD 23)

add_library(Math STATIC library.cpp
        Matrix.cppm
        Vector.cppm)
