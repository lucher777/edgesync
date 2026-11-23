cmake_minimum_required(VERSION 3.16)
project(edgesync VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_library(edgesync_static STATIC csrc/edgesync.c)
target_include_directories(edgesync_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/csrc)
target_compile_options(edgesync_static PRIVATE
    $<$<C_COMPILER_ID:GNU,Clang>:-O3 -Wall -Wextra -Werror -ffreestanding -fno-builtin>
    $<$<C_COMPILER_ID:MSVC>:/O2 /W4 /WX>
)

add_library(edgesync SHARED csrc/edgesync.c)
target_include_directories(edgesync PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/csrc)
target_compile_options(edgesync PRIVATE
    $<$<C_COMPILER_ID:GNU,Clang>:-O3 -Wall -Wextra -Werror -ffreestanding -fno-builtin>
    $<$<C_COMPILER_ID:MSVC>:/O2 /W4 /WX>
)

add_executable(bench_edgesync csrc/bench_edgesync.c)
target_link_libraries(bench_edgesync PRIVATE edgesync_static)
if(UNIX)
    target_link_libraries(bench_edgesync PRIVATE m)
endif()
