#------------------------------------------------------------------------------
# 3rd Party Dependencies
#------------------------------------------------------------------------------

# Policy to use <PackageName>_ROOT variable in find_<Package> commands
# Policy added in 3.12+
if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

set(TPL_DEPS)

#------------------------------------------------------------------------------
# Create global variable to toggle between GPU targets
#------------------------------------------------------------------------------
if(USE_CUDA)
    set(dgl_device_depends cuda CACHE STRING "" FORCE)
    find_package(CUDAToolkit REQUIRED)
endif()
if(USE_HIP)
    set(dgl_device_depends blt::hip CACHE STRING "" FORCE)
endif()

if(USE_HIP)
    if (HIPSOLVER_DIR)
        if (NOT EXISTS "${HIPSOLVER_DIR}")
            message(FATAL_ERROR "Given HIPSOLVER_DIR does not exist: ${HIPSOLVER_DIR}")
        endif()

        if (NOT IS_DIRECTORY "${HIPSOLVER_DIR}")
            message(FATAL_ERROR "Given HIPSOLVER_DIR is not a directory: ${HIPSOLVER_DIR}")
        endif()

        find_package(HIPSOLVER REQUIRED PATHS ${HIPSOLVER_DIR} )

        message(STATUS "Checking for expected HIPSOLVER target 'roc::hipsolver'")
        if (NOT TARGET roc::hipsolver)
            message(FATAL_ERROR "HIPSOLVER failed to load: ${HIPSOLVER_DIR}")
        else()
            message(STATUS "hipsolver loaded: ${HIPSOLVER_DIR}")
            get_target_property(HIPSOLVER_INCLUDE_DIRS_DGL roc::hipsolver INTERFACE_INCLUDE_DIRECTORIES)
            message(STATUS "HIPSOLVER Includes: ${HIPSOLVER_INCLUDE_DIRS_DGL}")
            set_property(TARGET roc::hipsolver
                 APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                ${HIPSOLVER_INCLUDE_DIRS_DGL})
            set(HIPSOLVER_FOUND TRUE CACHE BOOL "")
        endif()
    else()
        message(STATUS "hipsolver support is OFF")
        set(HIPSOLVER_FOUND FALSE CACHE BOOL "")
    endif()

    if (HIPBLAS_DIR)
        if (NOT EXISTS "${HIPBLAS_DIR}")
            message(FATAL_ERROR "Given HIPBLAS_DIR does not exist: ${HIPBLAS_DIR}")
        endif()

        if (NOT IS_DIRECTORY "${HIPBLAS_DIR}")
            message(FATAL_ERROR "Given HIPBLAS_DIR is not a directory: ${HIPBLAS_DIR}")
        endif()

        find_package(hipblas REQUIRED PATHS ${HIPBLAS_DIR} )

        message(STATUS "Checking for expected hipblas target 'roc::hipblas'")
        if (NOT TARGET roc::hipblas)
            message(FATAL_ERROR "hipblas failed to load: ${HIPBLAS_DIR}")
        else()
            message(STATUS "hipblas loaded: ${HIPBLAS_DIR}")
            get_target_property(HIPBLAS_INCLUDE_DIRS_DGL roc::hipblas INTERFACE_INCLUDE_DIRECTORIES)
            message(STATUS "hipblas Includes: ${HIPBLAS_INCLUDE_DIRS_DGL}")
            set_property(TARGET roc::hipblas
                 APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                ${HIPBLAS_INCLUDE_DIRS_DGL})
            set(HIPBLAS_FOUND TRUE CACHE BOOL "")
        endif()
    else()
        message(STATUS "hipblas support is OFF")
        set(HIPBLAS_FOUND FALSE CACHE BOOL "")
    endif()

    if (HIPSPARSE_DIR)
        if (NOT EXISTS "${HIPSPARSE_DIR}")
            message(FATAL_ERROR "Given HIPSPARSE_DIR does not exist: ${HIPSPARSE_DIR}")
        endif()

        if (NOT IS_DIRECTORY "${HIPSPARSE_DIR}")
            message(FATAL_ERROR "Given HIPSPARSE_DIR is not a directory: ${HIPSPARSE_DIR}")
        endif()

        find_package(hipsparse REQUIRED PATHS ${HIPSPARSE_DIR} )

        message(STATUS "Checking for expected hipsparse target 'roc::hipsparse'")
        if (NOT TARGET roc::hipsparse)
            message(FATAL_ERROR "hipsparse failed to load: ${HIPSPARSE_DIR}")
        else()
            message(STATUS "hipsparse loaded: ${HIPSPARSE_DIR}")
            get_target_property(HIPSPARSE_INCLUDE_DIRS_DGL roc::hipsparse INTERFACE_INCLUDE_DIRECTORIES)
            message(STATUS "hipsparse Includes: ${HIPSPARSE_INCLUDE_DIRS_DGL}")
            set_property(TARGET roc::hipsparse
                 APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                ${HIPSPARSE_INCLUDE_DIRS_DGL})
            set(HIPSPARSE_FOUND TRUE CACHE BOOL "")
        endif()
    else()
        message(STATUS "hipsparse support is OFF")
        set(HIPSPARSE_FOUND FALSE CACHE BOOL "")
    endif()
endif()

