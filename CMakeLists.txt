cmake_minimum_required(VERSION 3.24)

# TENSOR_EMULATE=ON compila tudo com o compilador C++ comum, sem CUDA, emulando os
# kernels na CPU (tests/emulate/cuda_runtime.h). Serve para testar sem GPU.
option(TENSOR_EMULATE "Emular o CUDA na CPU (sem nvcc nem GPU)" OFF)
option(TENSOR_BUILD_TESTS "Compilar os testes" ON)

if(TENSOR_EMULATE)
    project(tensor LANGUAGES CXX)
else()
    # Padrão: a GPU desta máquina (ex.: RTX 4070 Ti -> 89). Para outra: -DCMAKE_CUDA_ARCHITECTURES=89
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    project(tensor LANGUAGES CXX CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MSVC: os fontes têm acentos e símbolos (×, Σ) em UTF-8.
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()

set(TENSOR_SOURCES
    src/status.cpp
    src/stride.cpp
    src/host/layout.cpp
)

if(TENSOR_EMULATE)
    # Os .cu viram C++: layout.cu não tem kernel e compila como está; em ops.cu,
    # kernel<<<grid, block>>>(args) é trocado por LAUNCH(kernel, grid, block, args).
    file(READ ${CMAKE_CURRENT_SOURCE_DIR}/src/device/ops.cu OPS_SOURCE)
    string(REGEX REPLACE "([A-Za-z_][A-Za-z0-9_]*)<<<([^,]+), *([^>]+)>>>\\("
                         "LAUNCH(\\1, \\2, \\3, " OPS_SOURCE "${OPS_SOURCE}")
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/emulate/ops.cpp "${OPS_SOURCE}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/device/ops.cu)

    set_source_files_properties(src/device/layout.cu PROPERTIES LANGUAGE CXX)
    list(APPEND TENSOR_SOURCES src/device/layout.cu ${CMAKE_CURRENT_BINARY_DIR}/emulate/ops.cpp)
else()
    list(APPEND TENSOR_SOURCES src/device/layout.cu src/device/ops.cu)
endif()

add_library(tensor ${TENSOR_SOURCES})
target_include_directories(tensor
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})        # config/device.hpp

if(TENSOR_EMULATE)
    target_include_directories(tensor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/tests/emulate)   # cuda_runtime.h emulado
    target_compile_definitions(tensor PUBLIC TENSOR_EMULATE)
else()
    target_link_libraries(tensor PUBLIC CUDA::cudart)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tensor PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
endif()

if(TENSOR_BUILD_TESTS)
    enable_testing()
    foreach(name layout ops tensor)
        add_executable(test_${name} tests/test_${name}.cpp)
        target_link_libraries(test_${name} PRIVATE tensor)
        add_test(NAME ${name} COMMAND test_${name})
    endforeach()
endif()
