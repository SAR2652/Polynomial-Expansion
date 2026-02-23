function(add_cutlass_test name use_gtest)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} inference_server_cutlass   # <-- link your library
    )
    if(${use_gtest})
        target_link_libraries(${name}
                              gtest
                              gtest_main)
    endif()
    add_test(NAME ${name} COMMAND ${name})
endfunction()

# add_cutlass_test(test_Embedding OFF test_layers/test_Embedding.cu)
# add_cutlass_test(test_LSTMCell OFF test_layers/test_LSTMCell.cu)
add_cutlass_test(test_WeightsMetadata ON test_WeightsMetadata.cu)
add_cutlass_test(test_Linear OFF test_layers/test_Linear.cu)
