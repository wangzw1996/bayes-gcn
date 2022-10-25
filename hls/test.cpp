#include"test.h"
#include "weight1.h"
#include <stdint.h>
#include <cmath>
#include<stdlib.h>
#include<iostream>
#include"ap_int.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>



void read_data(data_bi in[node][1433],data_t out[node][feature])
{
	#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=out
	#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=in
	for (int j = 0; j < node; j++)
	{
		ap_uint<32> temp;

			for (int i = 0; i < 45; i++)
			{
#pragma HLS UNROLL
				for (int k = 0; k < 32; k++)
				{
#pragma HLS UNROLL
					if (i*32+k<1433)
				    {
						temp.range(k, k)  = in[j][i*32+k];
				    }
				    else
				    {
				    	temp.range(k,k) = 0;
				    }
				}
		    out[j][i]=temp;
		}
	}
}






void Extraction(data_t in[node][feature], data_t weight[feature][block][6],data_tt beta[node],data_tt alpha[block][6],data_tt bias[block][6],hls::stream<data_tt> inStream[block])
{
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=weight
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=weight
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=alpha
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=bias
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=in
	for (int m=0; m< 6 ;m++)
	{
	for(int l=0;l<node;l++)
    {
#pragma HLS PIPELINE
		for(int i=0;i<block;i++)
		{
#pragma HLS UNROLL
			data_t temp=0;
		    data_t bitcount1 =0;
		    for(int k=0;k<feature;k++)
		    {
#pragma HLS UNROLL
			    temp = in[l][k]^weight[k][i][m];
			    for(int p=0;p<32;p++)
			    {
#pragma HLS UNROLL
				    if (k*32+p < 1433)
				    {
					    data_bi temp2;
					    temp2=temp.range( (1* (p + 1) -1), p * 1);
				 	    bitcount1+=temp2;
				    }
			    }
		    }
		    data_tt temp1=0;
		    temp1 = alpha[i][m]*beta[l]* (1433-2*bitcount1);
		    inStream[i] << temp1 +bias[i][m];
	     }
    }
    }
}


void cc(data_t weight_in[feature][hidden],data_tt alpha_in[hidden],data_tt bias_in[hidden],data_t weight_out[feature][block][6],data_tt alpha_out[block][6],data_tt bias_out[block][6])
{
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=weight_in
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=weight_out
#pragma HLS ARRAY_PARTITION dim=3 type=complete variable=weight_out
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=alpha_in
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=alpha_out
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=alpha_out
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=bias_in
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=bias_out
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=bias_out
	for(int i=0;i<feature;i++)
	{
		for(int k=0;k<6; k++)
		{
#pragma HLS UNROLL
			for(int j=0;j<block; j++)
			{
#pragma HLS UNROLL
				if (j+block*k<hidden)
				{
					weight_out[i][j][k]=weight_in[i][j+block*k];
					alpha_out[j][k]=alpha_in[j+block*k];
				    bias_out[j][k]=bias_in[j+block*k];
				}
				else
				{
					weight_out[i][j][k]=0;
					alpha_out[j][k]=0;
				    bias_out[j][k]=0;
				}
			}
		}
	}
}












void Aggregation(data_tt out[block],data_tt out1[block],data_tt out2[block],data_tt out3[block],data_tt out4[block],data_t col[4][edge/4],data_t row[4][edge/4],data_t row1[4][edge/4],hls::stream<data_tt> inStream[block],hls::stream<data_tt> outStream1[block],hls::stream<data_tt> outStream2[block],hls::stream<data_tt> outStream3[block],hls::stream<data_tt> outStream4[block])
{
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=row
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=col

#pragma HLS DEPENDENCE variable=out1 inter false
#pragma HLS DEPENDENCE variable=out2 inter false
#pragma HLS DEPENDENCE variable=out3 inter false
#pragma HLS DEPENDENCE variable=out4 inter false

#pragma HLS ARRAY_PARTITION dim=0 type=complete variable=out1
#pragma HLS ARRAY_PARTITION dim=0 type=complete variable=out2
#pragma HLS ARRAY_PARTITION dim=0 type=complete variable=out3
#pragma HLS ARRAY_PARTITION dim=0 type=complete variable=out4

int temp1_row1=row[0][(edge/4)-1];
int temp2_row1=row[1][(edge/4)-1];
int temp3_row1=row[2][(edge/4)-1];



#pragma HLS BIND_STORAGE variable=temp type=ram_2p impl=uram
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=temp
#pragma HLS ARRAY_PARTITION dim=4 type=complete variable=temp
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=temp2
#pragma HLS ARRAY_PARTITION dim=4 type=complete variable=temp2

data_tt temp1[node][block][6];
	for (int m=0;m<6;m++)
	{
		data_tt temp[node][block][2];

		for (int i=0;i<node;i++)
		{
#pragma HLS PIPELINE
			for(int j=0;j<block;j++)
			{
#pragma HLS UNROLL
				data_tt temp_in[block];
				temp_in[j+k*block]=inStream[j].read();
				temp[i][j][0] = (temp_in[j]);
				temp[i][j][1] = (temp_in[j]);

				temp1[i][j][m] = (temp_in[j]);


			}
		}

		for (int i=0;i<edge/4;i++)
		{
			int temp1_row=row1[0][i];
			int temp1_col=col[0][i];
			int p_row1=0;
			if(i !=0)
			{
				p_row1=row[0][i-1];
			}

			int temp2_row=row1[1][i];
			int temp2_col=col[1][i];
	        int p_row2=0;
	        if(i !=0)
	        {
	    	   p_row2=row[1][i-1];
	        }


	       int temp3_row=row1[2][i];
	       int temp3_col=col[2][i];
	       int p_row3=0;
	       if(i !=0)
	       {
	          	p_row3=row[2][i-1];
	       }

	       int temp4_row=row1[3][i];
	       int temp4_col=col[3][i];
	       int p_row4=0;
	       if(i !=0)
	       {
	    	   p_row4=row[3][i-1];
           }

	       for (int k=0;k<block;k++)
	       {
#pragma HLS UNROLL
	    	   if(p_row1 == temp1_row)
	    	   {
	    		   out1[k] += temp[temp1_col][k][0];
	    	   }
	    	   else
	    	   {
	    		outStream1[k] << out1[k] + temp[p_row1][k][1];
	    		out1[k]=temp[temp1_col][k][0] ;

	    	   }

	    	   if(p_row2 == temp2_row)
	    	   {
	    		   out2[k] += temp[temp2_col][k][0];
	      	   }
	           else
	       	   {
	       		   outStream2[k] << out2[k] + temp[p_row2][k][1] ;
	       		   out2[k]=temp[temp2_col][k][0] ;
	       	   }

	    	   if(p_row3 == temp3_row)
	    	   {
	    		   out3[k] += temp[temp3_col][k][2];
	     	   }
	    	   else
	    	   {
	    		   outStream3[k] << out3[k] + temp[p_row3][k][3] ;
	    	   	   out3[k]=temp[temp3_col][k][2] ;
	    	   }

	    	   if(p_row4 == temp4_row)
	    	   {
	    	   	   out4[k] += temp[temp4_col][k][2];
	    	   }
	    	   else
	     	   {
 	    		   outStream4[k] << out4[k] + temp[p_row4][k][3] ;
	    	   	   out4[k]=temp[temp4_col][k][2] ;
	    	   }
	       }
		}
	}


	out[0]=temp1[1000][40][5];
	out[1]=temp1[281][40][5];
	out[2]=temp1[972][40][5];
	out[3]=temp1[1325][40][5];
	out[4]=temp1[2543][40][5];

}








ap_uint<256> sampler( ap_uint<256> seed, int load) {
  static ap_uint<256> mask;
  if (load ==1 )
    mask = seed;
  bool b_32 = mask.get_bit(256-32);
  bool b_104 = mask.get_bit(256-104);
  bool b_248 = mask.get_bit(256-248);
  bool b_1 = mask.get_bit(256-1);
  bool new_bit = b_32 ^ b_104 ^ b_248 ^ b_1;
  mask = mask >> 1;
  mask.set_bit(255, new_bit);

  return mask.to_uint();

}







void wr(hls::stream<data_tt> outStream1[block],hls::stream<data_tt> outStream2[block],hls::stream<data_tt> outStream3[block],hls::stream<data_tt> outStream4[block],data_out out[node][6])
{
	ap_uint<256> mask;
	for(int i=0;i<8;i++)
	{
		mask.range(32 * (i + 1) - 1, i * 32) = 10012903;
	}
	sampler(mask, 1);
	for (int m=0; m<6; m++)
	{
		for(int l=0;l<node1;l++)
		{
			ap_uint<256> mask;
			mask=sampler(mask,0);
			data_out  tmpOut1 = 0;
			for (int i = 0; i < block; i++)
			{
#pragma HLS UNROLL
				data_bi temp=mask.get_bit(i);
				tmpOut1.range(32 * (i + 1) - 1, i * 32) = (temp*(outStream1[i].read())).range(31, 0);
			}
			out[l][m]=tmpOut1;
		}

		for(int l=0;l<node2;l++)
		{
			ap_uint<256> mask;
			mask=sampler(mask,0);
			data_out  tmpOut2 = 0;
			for (int i = 0; i < block; i++)
			{
#pragma HLS UNROLL
				data_bi temp=mask.get_bit(i);
				tmpOut2.range(32 * (i + 1) - 1, i * 32) = (temp*(outStream2[i].read())).range(31, 0);
			}
			out[node1+l-1][m]=tmpOut2;
		}

		for(int l=0;l<node3;l++)
		{
			ap_uint<256> mask;
			mask=sampler(mask,0);
			data_out  tmpOut3 = 0;
			for (int i = 0; i < block; i++)
			{
#pragma HLS UNROLL
				data_bi temp=mask.get_bit(i);
				tmpOut3.range(32 * (i + 1) - 1, i * 32) = (temp*(outStream3[i].read())).range(31, 0);
			}
			out[node1+node2+l-1][m]=tmpOut3;
		}

	    for(int l=0;l<node4;l++)
		{
			ap_uint<256> mask;
			mask=sampler(mask,0);
			data_out  tmpOut4 = 0;
			for (int i = 0; i < block; i++)
			{
#pragma HLS UNROLL
				data_bi temp=mask.get_bit(i);
				tmpOut4.range(32 * (i + 1) - 1, i * 32) = (temp*(outStream4[i].read())).range(31, 0);
			}
			out[node1+node2+node3+l-1][m]=tmpOut4;
		}
	}
}










void TopFun(data_tt out11[block],data_bi in[node][1433], data_tt beta[node],data_t col[4][edge/4],data_t row[4][edge/4],data_t row1[4][edge/4],data_out out[node][6])
{
#pragma HLS INTERFACE mode=s_axilite port=out storage_impl=uram
#pragma HLS INTERFACE mode=s_axilite port=row storage_impl=uram
#pragma HLS INTERFACE mode=s_axilite port=col storage_impl=uram
#pragma HLS INTERFACE mode=s_axilite port=beta storage_impl=bram
#pragma HLS INTERFACE mode=s_axilite port=in storage_impl=bram

#pragma HLS INTERFACE mode=s_axilite port=return



	data_t in_buf[node][feature];
#pragma HLS BIND_STORAGE variable=in_buf type=ram_2p

	read_data(in,in_buf);



	data_t temp_weight[feature][block][6];
	data_tt temp_alpha[block][6];
	data_tt temp_bias[block][6];
    cc(weight,alpha,bias,temp_weight,temp_alpha,temp_bias);

#pragma HLS DATAFLOW

    	data_tt out1[block]={0};
      	data_tt out2[block];

      	data_tt out3[block];
    	data_tt out4[block];


    static hls::stream<data_tt> inStream[block];
#pragma HLS BIND_STORAGE variable=inStream type=fifo impl=uram
#pragma HLS STREAM depth=99 type=fifo variable=inStream

    static hls::stream<data_tt> outStream1[block];

#pragma HLS STREAM depth=99 type=fifo variable=outStream1
    static hls::stream<data_tt> outStream2[block];

#pragma HLS STREAM depth=99 type=fifo variable=outStream2
    static hls::stream<data_tt> outStream3[block];
#pragma HLS STREAM depth=99 type=fifo variable=outStream3
    static hls::stream<data_tt> outStream4[block];
#pragma HLS STREAM depth=99 type=fifo variable=outStream4


    Extraction(in_buf,temp_weight,beta,temp_alpha,temp_bias,inStream);
    Aggregation(out11,out1,out2,out3,out4,col,row,row1,inStream,outStream1,outStream2,outStream3,outStream4);
    wr(outStream1,outStream2,outStream3,outStream4,out);
}


