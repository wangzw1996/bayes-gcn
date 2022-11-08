#define node 2708
#define node1 651
#define node2 707
#define node3 583
#define node4 767
#define feature 1433
#define feature_block 45
#define input_size 512
#define input_block 3
#define edge 10556
#define hidden 256
#define cycle 6
#define block  3
#define avg_degree 4
#define drop_rate 0.125


#define AP_INT_MAX_W block*32
#include"ap_int.h"
#include"ap_fixed.h"

#include<hls_stream.h>

typedef ap_uint<1> data_bi;
typedef ap_uint<32> data_t;
typedef ap_uint<block*32> data_out;
typedef ap_uint<512> data_in;
typedef ap_fixed<32,8> data_tt;


void TopFun(data_in in[node][input_block], data_tt beta[node],data_t col[avg_degree][edge/avg_degree],data_t row[avg_degree][edge/avg_degree],data_t row1[avg_degree][edge/avg_degree],data_tt norm0[avg_degree][edge/avg_degree],data_tt norm1[node][2],data_out out[node][cycle]);
