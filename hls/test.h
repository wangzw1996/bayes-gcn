#define node 2708
#define node1 651
#define node2 707
#define node3 583
#define node4 767
#define feature 45
#define edge 10556
#define hidden 256
#define block  43
#define sample 5
#define avg_degree 4
#define drop_rate 0.5


#define AP_INT_MAX_W block*32
#include"ap_int.h"
#include"ap_fixed.h"

#include<hls_stream.h>

typedef ap_uint<1> data_bi;
typedef ap_uint<32> data_t;
typedef ap_uint<block*32> data_out;
typedef ap_uint<512> data_in;
typedef ap_fixed<32,8> data_tt;

void cc(hls::stream<data_in> stream_array[feature],data_t out[block][feature]);
void TopFun(int out12[node],data_in in[node][3], data_tt beta[node],data_t col[4][edge/4],data_t row[4][edge/4],data_t row1[4][edge/4],data_tt norm0[4][edge/4],data_tt norm1[node][2],data_out out[node][6]);
