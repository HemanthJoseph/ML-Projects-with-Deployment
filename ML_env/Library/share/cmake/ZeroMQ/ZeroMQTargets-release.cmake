#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "libzmq" for configuration "Release"
set_property(TARGET libzmq APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(libzmq PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/libzmq-mt-4_3_4.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/libzmq-mt-4_3_4.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS libzmq )
list(APPEND _IMPORT_CHECK_FILES_FOR_libzmq "${_IMPORT_PREFIX}/lib/libzmq-mt-4_3_4.lib" "${_IMPORT_PREFIX}/bin/libzmq-mt-4_3_4.dll" )

# Import target "libzmq-static" for configuration "Release"
set_property(TARGET libzmq-static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(libzmq-static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C;CXX;RC"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libzmq-mt-s-4_3_4.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS libzmq-static )
list(APPEND _IMPORT_CHECK_FILES_FOR_libzmq-static "${_IMPORT_PREFIX}/lib/libzmq-mt-s-4_3_4.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
