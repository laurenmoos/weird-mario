# Example usage:
#   find_package(CapnProto)
#   capnp_generate_cpp(CAPNP_SRCS CAPNP_HDRS schema.capnp)
#   include_directories(${CMAKE_CURRENT_BINARY_DIR})
#   add_executable(foo main.cpp ${CAPNP_SRCS})
#   target_link_libraries(foo CapnProto::capnp)
#
#  If you are using RPC features, use 'CapnProto::capnp-rpc'
#  in target_link_libraries call.
#

####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was CapnProtoConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

set(CapnProto_VERSION 0.6.1)

set(CAPNP_EXECUTABLE $<TARGET_FILE:CapnProto::capnp_tool>)
set(CAPNPC_CXX_EXECUTABLE $<TARGET_FILE:CapnProto::capnpc_cpp>)
set(CAPNP_INCLUDE_DIRECTORY "${PACKAGE_PREFIX_DIR}/include")

# work around http://public.kitware.com/Bug/view.php?id=15258
if(NOT _IMPORT_PREFIX)
  set(_IMPORT_PREFIX ${PACKAGE_PREFIX_DIR})
endif()



include("${CMAKE_CURRENT_LIST_DIR}/CapnProtoTargets.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/CapnProtoMacros.cmake")
