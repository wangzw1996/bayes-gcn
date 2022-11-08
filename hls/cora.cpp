#include "cora.h"
#include "design.cpp"
#include "weight1.h"
#include <stdint.h>
#include <cmath>
#include<stdlib.h>
#include<iostream>
#include"ap_int.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>





void TopFun(data_in in[node][input_block], data_tt beta[node],data_t col[avg_degree][edge/avg_degree],data_t row[avg_degree][edge/avg_degree],data_t row1[avg_degree][edge/avg_degree],data_tt norm0[avg_degree][edge/avg_degree],data_tt norm1[node][2],data_out out[node][cycle])
{
#pragma HLS INTERFACE mode=s_axilite port=out storage_impl=uram
#pragma HLS INTERFACE mode=s_axilite port=row storage_impl=uram
#pragma HLS INTERFACE mode=s_axilite port=row1 storage_impl=uram
#pragma HLS INTERFACE mode=s_axilite port=norm0 storage_impl=uram
#pragma HLS INTERFACE mode=s_axilite port=col storage_impl=uram
#pragma HLS INTERFACE mode=s_axilite port=beta storage_impl=bram
#pragma HLS INTERFACE mode=s_axilite port=in storage_impl=uram

#pragma HLS INTERFACE mode=s_axilite port=return


#pragma HLS DATAFLOW
	data_t in_buf[node][ceil(feature/32)];
#pragma HLS BIND_STORAGE variable=in_buf type=ram_2p

	read_data(in,in_buf);



	data_t temp_weight[feature][block][cycle];
	data_tt temp_alpha[block][cycle];
	data_tt temp_bias[block][cycle];
    cc(weight,alpha,bias,temp_weight,temp_alpha,temp_bias);



    	data_tt out1[2*block]={0};
      	data_tt out2[2*block];

      	data_tt out3[2*block];
    	data_tt out4[2*block];


    static hls::stream<data_tt> inStream[block];
#pragma HLS BIND_STORAGE variable=inStream type=fifo impl=uram
#pragma HLS STREAM depth=999 type=fifo variable=inStream

    static hls::stream<data_tt> outStream1[2*block];
#pragma HLS STREAM depth=999 type=fifo variable=outStream1
    static hls::stream<data_tt> outStream1_2[block*2];
#pragma HLS STREAM depth=999 type=fifo variable=outStream1_2
    static hls::stream<data_tt> outStream2[2*block];
#pragma HLS STREAM depth=999 type=fifo variable=outStream2
    static hls::stream<data_tt> outStream2_2[block*2];
#pragma HLS STREAM depth=999 type=fifo variable=outStream2_2
    static hls::stream<data_tt> outStream3[2*block];
#pragma HLS STREAM depth=999 type=fifo variable=outStream3
    static hls::stream<data_tt> outStream3_2[2*block];
#pragma HLS STREAM depth=999 type=fifo variable=outStream3_2
    static hls::stream<data_tt> outStream4[2*block];
#pragma HLS STREAM depth=999 type=fifo variable=outStream4


    Extraction(in_buf,temp_weight,beta,temp_alpha,temp_bias,inStream);
    Aggregation(out1,out2,out3,out4,col,row,row1,norm0,norm1,inStream,outStream1,outStream1_2,outStream2,outStream2_2,outStream3,outStream3_2,outStream4);
    wr(outStream1,outStream1_2,outStream2,outStream2_2,outStream3,outStream3_2,outStream4,out);
}
