cmake_minimum_required(VERSION 3.24)

# TENSOR_EMULATE=ON compila tudo com o compilador C++ comum, sem CUDA, emulando os
# kernels na CPU (tests/emulate/cuda_runtime.h). Serve para testar sem GPU.
option(TENSOR_EMULATE "Emular o CUDA na CPU (sem nvcc nem GPU)" OFF)
option(TENSOR_BUILD_TESTS "Compilar os testes" ON)

if(TENSOR_EMULATE)
    project(tensor VERSION 0.1.0 LANGUAGES CXX)
else()
    # Padrão: a GPU desta máquina (ex.: RTX 4070 Ti -> 89). Para outra: -DCMAKE_CUDA_ARCHITECTURES=89
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    project(tensor VERSION 0.1.0 LANGUAGES CXX CUDA)
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

include(GNUInstallDirs)

add_library(tensor ${TENSOR_SOURCES})
add_library(tensor::tensor ALIAS tensor)    # mesmo nome do find_package(tensor): quem usa não muda
set_target_properties(tensor PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Os headers ficam em include/tensor/: quem usa escreve #include "tensor/host/tensor.hpp".
target_include_directories(tensor
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})        # config/device.hpp

if(TENSOR_EMULATE)
    target_include_directories(tensor PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tests/emulate>)   # cuda_runtime.h emulado
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

# Instalação: cmake --install build (padrão /usr/local; outro lugar: --prefix /caminho)
#   <prefix>/include/tensor/...          headers
#   <prefix>/lib/libtensor.a
#   <prefix>/lib/cmake/tensor/           find_package(tensor) -> tensor::tensor
# Só no build com CUDA: o emulado troca o cuda_runtime.h e não serve fora dos testes.
if(TENSOR_EMULATE)
    message(STATUS "tensor: TENSOR_EMULATE=ON, sem regras de instalação")
else()
    include(CMakePackageConfigHelpers)

    install(TARGETS tensor EXPORT tensorTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(DIRECTORY include/tensor DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    install(EXPORT tensorTargets
        NAMESPACE tensor::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tensor)

    configure_package_config_file(cmake/tensorConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/tensorConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tensor)
    write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/tensorConfigVersion.cmake
        COMPATIBILITY SameMinorVersion)
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/tensorConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/tensorConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tensor)
endif()
