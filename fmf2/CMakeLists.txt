include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${Python3_NumPy_INCLUDE_DIRS} SYSTEM)

# Add NumPy to required libraries to link against
list(APPEND FMF2_LIBS Python3::NumPy)

# If the enable skip option is ON then we define an appropriate macro to enable
if (ENABLE_SKIP)
	message(STATUS "Enabling CPU skip on sparse weights")
	add_compile_definitions(FMF2_CPU_SKIP)
endif()

# Include the "serial" implementation which should always be available
add_library(serial src/serial.cpp)
# Compile the serial library with native instructions
target_compile_options(serial PRIVATE "-march=native")
list(APPEND FMF2_LIBS serial)

# If we have OpenMP support in the compiler we add ensure our "serial" code is
# compiled with OpenMP support
if(OpenMP_CXX_FOUND)
	target_compile_options(serial PRIVATE ${OpenMP_CXX_FLAGS})
	target_link_libraries(serial ${OpenMP_CXX_LIBRARIES})
endif()

# Add SYCL backend if available compiler is found
if(HIPSYCL_SYCLCC)
	# Build as a shared library so one isn't dependent on SYCL at runtime
	# to use the CPU backend
	add_library(sycl SHARED src/sycl.cpp)
	add_sycl_to_target(TARGET sycl SOURCES src/sycl.cpp)
	list(APPEND FMF2_LIBS sycl)
	set(CYTHON_FLAGS "${CYTHON_FLAGS} -E FMF2_SYCL=True")
	install(TARGETS sycl LIBRARY DESTINATION fmf2)
else()
	set(CYTHON_FLAGS "${CYTHON_FLAGS} -E FMF2_SYCL=False")
endif()

# Create Cython module
add_cython_target(_fmf2 CXX PY3)
add_library(_fmf2 MODULE ${_fmf2})
# Link against internal libraries
target_link_libraries(_fmf2 ${FMF2_LIBS})
python_extension_module(_fmf2)

# Need to manually set the RPATH for the Python library so that it can find the
# SYCL library above in a relative position next to itself
set_target_properties(_fmf2 PROPERTIES
	INSTALL_RPATH_USE_LINK_PATH TRUE
	INSTALL_RPATH "${PYTHON_SITE_PACKAGES_DIR}/fmf2")

# Install Cython package
install(TARGETS _fmf2 LIBRARY DESTINATION fmf2)
install(FILES __init__.py DESTINATION fmf2)
