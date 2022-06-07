cmake_minimum_required(VERSION 2.8.11)
project(openvino_detection)

set(CMAKE_CXX_STANDARD 14)

set(BIN_NAME "openvino_detection_unit_test")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} --coverage")

# Find Gtest
include_directories($ENV{GTEST_ROOT}/include)
link_directories($ENV{GTEST_ROOT}/lib)

# Test Src Module
find_package(OpenCV REQUIRED)
find_package(InferenceEngine)
find_package(ngraph REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../cpp ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/common/utils/include)
add_library(${CMAKE_PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/openvino_detection.cpp)
target_include_directories(${CMAKE_PROJECT_NAME} 
        PUBLIC ${OpenCV_INCLUDE_DIR}
        PUBLIC ${InferenceEngine_INCLUDE_DIRS}
        PUBLIC ${ngraph_INCLUDE_DIRS}
)
target_link_libraries(${CMAKE_PROJECT_NAME}
        ${OpenCV_LIBS}
        ${InferenceEngine_LIBRARIES}
        ${ngraph_LIBRARIES}
)


#Test Case Bin
enable_testing()
set(TEST_CASES ${CMAKE_CURRENT_SOURCE_DIR}/OpenvinoTests.cpp)
add_executable(${BIN_NAME} ${TEST_CASES})
target_link_libraries(${BIN_NAME} ${CMAKE_PROJECT_NAME} gtest gtest_main pthread)


add_custom_command(TARGET ${BIN_NAME}
                   POST_BUILD
                   COMMAND echo --Running unit tests
		           COMMAND rm -rf ../reports
		           COMMAND mkdir ../reports
                   COMMAND mkdir ../reports/unit_test_report
                   COMMAND ctest -V | tee ../reports/unit_test_report/report.txt
                   COMMAND echo Unit test reports written to reports/unit_test_report as report.txt
                   COMMAND echo --Running code coverage
                   COMMAND rm -rf code_coverage_report
                   COMMAND lcov --rc "lcov_branch_coverage=1" -o coverage_all.info -c -d .
		           COMMAND lcov --rc "lcov_branch_coverage=1" -e coverage_all.info "*/cpp/openvino*" -o coverage.info 
                   COMMAND genhtml coverage.info --branch-coverage --legend --output-directory code_coverage_report
                   COMMAND cp -r code_coverage_report ../reports/
                   COMMAND echo Coverage report written to reports/code_coverage_report as index.html
                   )

add_test(UnitTests ${BIN_NAME})
