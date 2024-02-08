cmake_minimum_required(VERSION 3.20)

project( duney LANGUAGES CUDA CXX )

enable_language(CUDA)

find_package(CUDAToolkit 10.0 REQUIRED)

file(GLOB HEADERS ${PROJECT_SOURCE_DIR}/include/*.cuh)

add_compile_definitions("DEBUG=$<CONFIG:Debug>")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(DEBUG_OPTIM_FLAGS "-g" "-G" "-O0")
else()
    set(DEBUG_OPTIM_FLAGS "-O3")
endif ()

add_executable(main main.cu ${HEADERS} ext/lodepng/lodepng.h ext/lodepng/lodepng.cpp)
target_include_directories(main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/)
target_include_directories(main PUBLIC ext/lodepng)
set_property(TARGET main PROPERTY CUDA_ARCHITECTURES 89)

target_compile_options(main PUBLIC "${DEBUG_OPTIM_FLAGS}" "")

target_link_libraries( main CUDA::cuda_driver )
