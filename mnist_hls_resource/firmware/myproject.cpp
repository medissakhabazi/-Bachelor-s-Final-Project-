#include <iostream>

#include "myproject.h"
#include "parameters.h"


void myproject(
    input_t input_layer[28*28],
    result_t layer6_out[10]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=input_layer complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=input_layer,layer6_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<dense_weight_t, 100352>(w3, "w3.txt");
        nnet::load_weights_from_txt<dense_bias_t, 128>(b3, "b3.txt");
        nnet::load_weights_from_txt<dense_1_weight_t, 1280>(w6, "w6.txt");
        nnet::load_weights_from_txt<dense_1_bias_t, 10>(b6, "b6.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    auto& layer2_out = input_layer;
    dense_result_t layer3_out[128];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0

    layer4_t layer4_out[128];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0

    nnet::dense<input_t, dense_result_t, config3>(layer2_out, layer3_out, w3, b3); // dense

    nnet::relu<dense_result_t, layer4_t, relu_config4>(layer3_out, layer4_out); // dense_relu

    nnet::dense<layer4_t, result_t, config6>(layer4_out, layer6_out, w6, b6); // dense_1

}

