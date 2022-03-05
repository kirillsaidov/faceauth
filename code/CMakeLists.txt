cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# project name init
project(faceauth
		VERSION 1.0
		DESCRIPTION "FaceAuth - An automated face authetication system."
		LANGUAGES CXX)

# setting default build type
set(CMAKE_CXX_STANDARD 14)
set(default_buid_type "Release")
# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib) # for libraries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin) # for binaries 

# packages
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# finding all headers and sources 
file(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE HEADERS ${PROJECT_SOURCE_DIR}/include/*.hpp)

# include directories
include_directories(${PROJECT_SOURCE_DIR}/include/)

# building library/binary
# add_library(${PROJECT_NAME} STATIC ${SOURCES} ${HEADERS}) # for libraries
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})   # for binaries
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)



