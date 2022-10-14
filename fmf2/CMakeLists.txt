include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${Python3_NumPy_INCLUDE_DIRS} SYSTEM)

# Include the "serial" implementation which should always be available
add_library(serial src/serial.cpp)
# AVX2 should always be available as it is quite old and most all modern x86_64
# CPUs support it
target_compile_options(serial PUBLIC "-mavx2" "-mfma")
list(APPEND FMF2_LIBS serial)

# If we have OpenMP support in the compiler we add ensure our "serial" code is
# compiled with OpenMP support
if(OpenMP_CXX_FOUND)
	target_compile_options(serial PRIVATE ${OpenMP_CXX_FLAGS})
	target_link_libraries(serial ${OpenMP_CXX_LIBRARIES})
endif()

# Add SYCL backend if available compiler is found
if(HIPSYCL_SYCLCC)
	find_package(Boost REQUIRED COMPONENTS fiber)
	find_package(Threads REQUIRED)
	add_library(sycl src/sycl.cpp)
	add_sycl_to_target(TARGET sycl SOURCES src/sycl.cpp)
	list(APPEND FMF2_LIBS sycl Boost::fiber Threads::Threads)
	set(CYTHON_FLAGS "${CYTHON_FLAGS} -E FMF2_SYCL=True")
else()
	set(CYTHON_FLAGS "${CYTHON_FLAGS} -E FMF2_SYCL=False")
endif()

# Create Cython module
add_cython_target(_fmf2 CXX PY3)
add_library(_fmf2 MODULE ${_fmf2})
# Link against internal libraries
target_link_libraries(_fmf2 ${FMF2_LIBS} Python3::NumPy)
python_extension_module(_fmf2)

# Install Cython package
install(TARGETS _fmf2 LIBRARY DESTINATION fmf2)
install(FILES __init__.py DESTINATION fmf2)
