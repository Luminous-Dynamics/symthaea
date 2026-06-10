cmake_minimum_required(VERSION 3.22)
project(spore_jni C)

# JNI C glue — thin layer between Kotlin and the Rust cdylib
add_library(spore_jni SHARED spore_jni.c)

# Link against pre-built Rust library (cross-compiled per ABI)
add_library(symthaea_spore SHARED IMPORTED)
set_target_properties(symthaea_spore PROPERTIES
    IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libsymthaea_spore.so
)

target_link_libraries(spore_jni symthaea_spore log)
