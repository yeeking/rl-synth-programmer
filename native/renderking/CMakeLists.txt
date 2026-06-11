cmake_minimum_required(VERSION 3.22)

project(renderking VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG REQUIRED)

add_subdirectory(../../libs/JUCE JUCE)

pybind11_add_module(_renderking src/renderking.cpp)

set(RENDERKING_PYTHON_OUTPUT_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/rl_synth_programmer"
    CACHE PATH "Directory where the Python extension module is written")

set_target_properties(_renderking PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${RENDERKING_PYTHON_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${RENDERKING_PYTHON_OUTPUT_DIR}"
    OUTPUT_NAME "_renderking")

target_compile_definitions(_renderking
    PRIVATE
        JUCE_ALSA=1
        JUCE_DIRECTSOUND=1
        JUCE_DISABLE_CAUTIOUS_PARAMETER_ID_CHECKING=1
        JUCE_PLUGINHOST_VST3=1
        JUCE_PLUGINHOST_VST=0
        JUCE_PLUGINHOST_AU=0
        JUCE_PLUGINHOST_LADSPA=0
        JUCE_PLUGINHOST_LV2=0
        JUCE_PLUGINHOST_ARA=0
        JUCE_USE_CURL=0
        JUCE_WEB_BROWSER=0
        JUCE_USE_CAMERA=0
        JUCE_USE_CDBURNER=0
        JUCE_USE_CDREADER=0
        JUCE_MODAL_LOOPS_PERMITTED=0)

target_link_libraries(_renderking
    PRIVATE
        juce::juce_audio_utils
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags)
