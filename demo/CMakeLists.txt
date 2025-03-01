
add_executable(
    demo 
    demo.cpp
)

target_link_libraries(demo 
    torch 
    torch_library
)

target_include_directories(
    demo
    PUBLIC
    ${CMAKE_SOURCE_DIR}/torch/csrc
)