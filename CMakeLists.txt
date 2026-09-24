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
set(KAIRO_SCHEDULER_DIR "" CACHE PATH "Optional explicit path to KairoScheduler.")
set(KAIRO_SIMD_DIR "" CACHE PATH "Optional explicit path to KairoSIMD.")
set(KAIRO_GPU_DIR "" CACHE PATH "Optional explicit path to KairoGPU.")

function(kairo_resolve_optional_repo output explicit_dir repo_name)
    if(explicit_dir AND EXISTS "${explicit_dir}/CMakeLists.txt")
        set(${output} "${explicit_dir}" PARENT_SCOPE)
        return()
    endif()

    foreach(candidate IN ITEMS
        "${CMAKE_CURRENT_SOURCE_DIR}/../${repo_name}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../${repo_name}")
        if(EXISTS "${candidate}/CMakeLists.txt")
            set(${output} "${candidate}" PARENT_SCOPE)
            return()
        endif()
    endforeach()

    set(${output} "" PARENT_SCOPE)
endfunction()

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
    if(NOT TARGET KairoScheduler)
        kairo_resolve_optional_repo(_kairo_scheduler_source "${KAIRO_SCHEDULER_DIR}" KairoScheduler)
        if(NOT _kairo_scheduler_source)
            message(FATAL_ERROR "KAIRO_MATH_USE_SCHEDULER is ON, but KairoScheduler was not found. Set KAIRO_SCHEDULER_DIR or place it beside the Kairo workspace.")
        endif()
        add_subdirectory("${_kairo_scheduler_source}" "${CMAKE_BINARY_DIR}/KairoScheduler")
    endif()
    if(NOT TARGET KairoSIMD)
        kairo_resolve_optional_repo(_kairo_simd_source "${KAIRO_SIMD_DIR}" KairoSIMD)
        if(NOT _kairo_simd_source)
            message(FATAL_ERROR "KAIRO_MATH_USE_SCHEDULER is ON, but KairoSIMD was not found. Set KAIRO_SIMD_DIR or place it beside the Kairo workspace.")
        endif()
        add_subdirectory("${_kairo_simd_source}" "${CMAKE_BINARY_DIR}/KairoSIMD")
    endif()
    target_sources(KairoMath PUBLIC FILE_SET CXX_MODULES
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES TensorExecution.cppm)
    target_link_libraries(KairoMath PUBLIC KairoScheduler KairoSIMD)

    add_executable(TensorExecutionSmoke tests/TensorExecutionSmoke.cpp)
    target_link_libraries(TensorExecutionSmoke PRIVATE KairoMath KairoScheduler KairoSIMD)
    add_test(NAME KairoMath.TensorExecutionSmoke COMMAND TensorExecutionSmoke)
endif()

if(KAIRO_MATH_USE_GPU)
    if(NOT TARGET KairoGPU)
        kairo_resolve_optional_repo(_kairo_gpu_source "${KAIRO_GPU_DIR}" KairoGPU)
        if(NOT _kairo_gpu_source)
            message(FATAL_ERROR "KAIRO_MATH_USE_GPU is ON, but KairoGPU was not found. Set KAIRO_GPU_DIR or place it beside the Kairo workspace.")
        endif()
        add_subdirectory("${_kairo_gpu_source}" "${CMAKE_BINARY_DIR}/KairoGPU")
    endif()
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

    add_executable(KairoMathCertification tests/FoundationCertification.cpp)
    target_link_libraries(KairoMathCertification PRIVATE KairoMath Catch2::Catch2WithMain)
    add_test(NAME KairoMath.Certification COMMAND KairoMathCertification)

endif()

# The visual laboratory is a developer-facing executable, not a library test.
# Keep it independently selectable so parent superbuilds do not inherit its
# large generated report strings or local server workflow.
if(KAIRO_MATH_BUILD_VISUALIZER)
    add_executable(MathVisualizer visual/VisualTests.cpp)
    target_link_libraries(MathVisualizer PRIVATE KairoMath)
endif()

# Kairo portfolio warning policy: Kairo-owned source warnings are build failures.
# Catch2's __COUNTER__ registration macro is the only bounded Clang 23 test
# exception; production targets receive no blanket warning suppression.
foreach(_kairo_target IN ITEMS KairoMath KairoMathCertification)
    if(TARGET ${_kairo_target})
        set_property(TARGET ${_kairo_target} PROPERTY COMPILE_WARNING_AS_ERROR ON)
    endif()
endforeach()
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 23)
    foreach(_kairo_test_target IN ITEMS KairoMathCertification)
        if(TARGET ${_kairo_test_target})
            target_compile_options(${_kairo_test_target} PRIVATE -Wno-c2y-extensions)
        endif()
    endforeach()
endif()
