#ifndef POT_MULT_H_
#define POT_MULT_H_

#include "ap_fixed.h"
#include "ap_int.h"

namespace nnet {
namespace product {

template <class x_T, class w_T>
class pot_mult {
public:

    typedef ap_fixed<40,20> result_t;

    static result_t product(x_T &a, w_T &w) {

        #pragma HLS INLINE

        if (!w.nonzero)
            return result_t(0);

        result_t x = static_cast<result_t>(a);
        result_t y;

        switch ((int)w.exponent) {

            case -3:
                y = x >> 3;
                break;

            case -2:
                y = x >> 2;
                break;

            case -1:
                y = x >> 1;
                break;

            

            default:
                y = 0;
                break;
        }

        if (!w.positive)
            y = -y;

        return y;
    }
};

} // namespace product
} // namespace nnet

#endif
