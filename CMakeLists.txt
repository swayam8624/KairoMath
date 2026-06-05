cmake_minimum_required(VERSION 3.28)
project(Math LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler options
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(-Wall -Wextra -Wno-unused-parameter)
elseif(MSVC)
    add_compile_options(/W4)
endif()

# Create Math static library target
add_library(KairoMath)

# Configure the C++ Modules File Set
target_sources(KairoMath
    PUBLIC
        FILE_SET CXX_MODULES
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
            Vector.cppm
            Matrix.cppm
            Quaternion.cppm
            Transform.cppm
            DynamicMatrix.cppm
            LinearSolve.cppm
            Decomposition.cppm
            Eigen.cppm
            SVD.cppm
            Statistics.cppm
)

# Catch2 Unit Test Integration
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.4
)
FetchContent_MakeAvailable(Catch2)

# MathTests executable
add_executable(MathTests tests/MathTests.cpp)
target_link_libraries(MathTests PRIVATE KairoMath Catch2::Catch2WithMain)

# MathVisualizer executable
add_executable(MathVisualizer visual/VisualTests.cpp)
target_link_libraries(MathVisualizer PRIVATE KairoMath)

