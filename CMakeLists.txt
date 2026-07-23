cmake_minimum_required(VERSION 3.28)
project(Math LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(KAIRO_MATH_BUILD_TESTS "Build KairoMath unit tests" ${PROJECT_IS_TOP_LEVEL})
option(KAIRO_MATH_BUILD_VISUALIZER "Build the standalone KairoMath visual laboratory" ${PROJECT_IS_TOP_LEVEL})
option(KAIRO_MATH_USE_SCHEDULER "Build optional Tensor execution kernels on KairoScheduler" OFF)
option(KAIRO_MATH_USE_GPU "Build optional Float32 Tensor execution kernels on KairoGPU" OFF)
set(KAIRO_SCHEDULER_DIR "/Users/swayamsingal/Desktop/Programming/Kairo/KairoScheduler" CACHE PATH "Path to KairoScheduler.")
set(KAIRO_GPU_DIR "/Users/swayamsingal/Desktop/Programming/Kairo/KairoGPU" CACHE PATH "Path to KairoGPU.")

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
            Tensor.cppm
            TensorAutograd.cppm
            TensorTraining.cppm
            TensorData.cppm
            LinearSolve.cppm
            Decomposition.cppm
            Eigen.cppm
            SVD.cppm
            Statistics.cppm
            MatrixFunctions.cppm
            Optimization.cppm
            Probability.cppm
            KairoMath.cppm
)

if(KAIRO_MATH_BUILD_TESTS)
    add_executable(TensorAutogradSmoke tests/TensorAutogradSmoke.cpp)
    target_link_libraries(TensorAutogradSmoke PRIVATE KairoMath)
    add_test(NAME KairoMath.TensorAutogradSmoke COMMAND TensorAutogradSmoke)
    add_executable(TensorTrainingSmoke tests/TensorTrainingSmoke.cpp)
    target_link_libraries(TensorTrainingSmoke PRIVATE KairoMath)
    add_test(NAME KairoMath.TensorTrainingSmoke COMMAND TensorTrainingSmoke)
    add_executable(TensorDataSmoke tests/TensorDataSmoke.cpp)
    target_link_libraries(TensorDataSmoke PRIVATE KairoMath)
    add_test(NAME KairoMath.TensorDataSmoke COMMAND TensorDataSmoke)
    add_executable(TensorDTypeSmoke tests/TensorDTypeSmoke.cpp)
    target_link_libraries(TensorDTypeSmoke PRIVATE KairoMath)
    add_test(NAME KairoMath.TensorDTypeSmoke COMMAND TensorDTypeSmoke)
endif()

if(KAIRO_MATH_USE_SCHEDULER)
    if(NOT EXISTS "${KAIRO_SCHEDULER_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "KAIRO_MATH_USE_SCHEDULER is ON, but KairoScheduler was not found at: ${KAIRO_SCHEDULER_DIR}")
    endif()
    add_subdirectory("${KAIRO_SCHEDULER_DIR}" "${CMAKE_BINARY_DIR}/KairoScheduler")
    target_sources(KairoMath PUBLIC FILE_SET CXX_MODULES
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES TensorExecution.cppm)
    target_link_libraries(KairoMath PUBLIC KairoScheduler)

    add_executable(TensorExecutionSmoke tests/TensorExecutionSmoke.cpp)
    target_link_libraries(TensorExecutionSmoke PRIVATE KairoMath KairoScheduler)
    add_test(NAME KairoMath.TensorExecutionSmoke COMMAND TensorExecutionSmoke)
endif()

if(KAIRO_MATH_USE_GPU)
    if(NOT EXISTS "${KAIRO_GPU_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "KAIRO_MATH_USE_GPU is ON, but KairoGPU was not found at: ${KAIRO_GPU_DIR}")
    endif()
    add_subdirectory("${KAIRO_GPU_DIR}" "${CMAKE_BINARY_DIR}/KairoGPU")
    target_sources(KairoMath PUBLIC FILE_SET CXX_MODULES
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES TensorGPU.cppm)
    target_link_libraries(KairoMath PUBLIC KairoGPU)

    add_executable(TensorGPUSmoke tests/TensorGPUSmoke.cpp)
    target_link_libraries(TensorGPUSmoke PRIVATE KairoMath KairoGPU)
    add_test(NAME KairoMath.TensorGPUSmoke COMMAND TensorGPUSmoke)
endif()

if(KAIRO_MATH_BUILD_TESTS)
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
    add_test(NAME KairoMath.MathTests COMMAND MathTests)

endif()

# The visual laboratory is a developer-facing executable, not a library test.
# Keep it independently selectable so parent superbuilds do not inherit its
# large generated report strings or local server workflow.
if(KAIRO_MATH_BUILD_VISUALIZER)
    add_executable(MathVisualizer visual/VisualTests.cpp)
    target_link_libraries(MathVisualizer PRIVATE KairoMath)
endif()
