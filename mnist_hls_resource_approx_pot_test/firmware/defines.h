#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <array>
#include <cstddef>
#include <cstdio>
#include <tuple>
#include <tuple>


// hls-fpga-machine-learning insert numbers

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<43,23> dense_accum_t;
typedef ap_fixed<43,23> dense_result_t;
typedef ap_fixed<16,6> dense_weight_t;
typedef ap_fixed<16,6> dense_bias_t;
typedef ap_uint<1> layer3_index;
typedef ap_fixed<16,6> layer4_t;
typedef ap_fixed<18,8> dense_relu_table_t;
typedef ap_fixed<40,20> dense_1_accum_t;
typedef ap_fixed<40,20> result_t;
typedef ap_fixed<16,6> dense_1_weight_t;
typedef ap_fixed<16,6> dense_1_bias_t;
typedef ap_uint<1> layer6_index;

// hls-fpga-machine-learning insert emulator-defines


#endif
