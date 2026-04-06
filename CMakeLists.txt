cmake_minimum_required(VERSION 3.20)
project(dynamic_bitvector_test_suite LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_ASAN "Enable AddressSanitizer" OFF)

add_executable(bitvector_tests
    tests/main.cpp
)

target_include_directories(bitvector_tests PRIVATE include)

target_compile_options(bitvector_tests PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

if(ENABLE_ASAN)
    target_compile_options(bitvector_tests PRIVATE -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(bitvector_tests PRIVATE -fsanitize=address)
endif()

enable_testing()
add_test(NAME bitvector_tests COMMAND bitvector_tests)
