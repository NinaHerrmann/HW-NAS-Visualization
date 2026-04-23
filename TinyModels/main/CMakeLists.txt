
list(APPEND requires esp-dl)

idf_build_get_property(component_targets __COMPONENT_TARGETS)
if ("___idf_espressif__esp-dl" IN_LIST component_targets)
   idf_component_get_property(espdl_dir espressif__esp-dl COMPONENT_DIR)
elseif("___idf_esp-dl" IN_LIST component_targets)
   idf_component_get_property(espdl_dir esp-dl COMPONENT_DIR)
endif()
set(cmake_dir ${espdl_dir}/fbs_loader/cmake)
include(${cmake_dir}/utilities.cmake)

file(GLOB embed_models models/s3/model3043.espdl)

file(GLOB embed_imgs "${CMAKE_CURRENT_SOURCE_DIR}/data/*")

set(embed_files ${embed_imgs} ${embed_models})
set(src_dirs        ./)

idf_component_register(SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires} PRIV_REQUIRES nvs_flash EMBED_FILES ${embed_files})