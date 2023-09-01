cmake_minimum_required(VERSION 3.20)
project(categorizer)

set(CMAKE_CXX_STANDARD 23)

find_package(Boost 1.82.0)
if(Boost_FOUND)
    include_directories(. ${Boost_INCLUDE_DIRS})
    add_executable(categorizer
            main.cpp
            categorizer/BlackBoxCategorizer.h
            categorizer/TensorTransformer.h)
else()
    message("Boost not found. Aborting build.")
endif()
