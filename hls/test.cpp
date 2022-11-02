#include "8.h"
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

void read_data1(data_in in[node][3],data_t out[node][feature])
{
	#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=out
	#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=in
	for (int j = 0; j < node; j++)
	{
		ap_uint<32> temp;

		for(int m=0; m<3; m++)
		{
#pragma HLS UNROLL
			for (int i = 0; i < 512/32; i++)
			{
#pragma HLS UNROLL
				for (int k = 0; k < 32; k++)
				{
#pragma HLS UNROLL
					if (m*512+i*32+k<1433)
				    {
						temp.range(k, k)  = in[j][m].range(i*32+k,i*32+k);
				    }
				    else
				    {
				    	temp.range(k,k) = 0;
				    }
				}
			    if (m*512/32+i< feature)
			    {
			    	out[j][m*512/32+i]=temp;
			    }
			}
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

	for (int m=0; m< 3 ;m++)
    {
		for(int j=0;j<2;j++)
		{
		for (int l=0;l<node;l++)
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
			    temp = in[l][k]^weight[k][i][m*2+j];
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
		    temp1 = alpha[i][m*2+j]*beta[l]* (1433-2*bitcount1);
		    inStream[i] << temp1 +bias[i][m*2+j];
	     }
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











void Aggregation(data_tt out1[2*block],data_tt out2[2*block],data_tt out3[2*block],data_tt out4[2*block],data_t col[avg_degree][edge/avg_degree],data_t row[avg_degree][edge/avg_degree],data_t row1[avg_degree][edge/avg_degree],data_tt norm0[avg_degree][edge/avg_degree],data_tt norm1[node][avg_degree/2],hls::stream<data_tt> inStream[block],hls::stream<data_tt> outStream1[2*block],hls::stream<data_tt> outStream1_2[block*2],hls::stream<data_tt> outStream2[2*block],hls::stream<data_tt> outStream2_2[block*2],hls::stream<data_tt> outStream3[2*block],hls::stream<data_tt> outStream3_2[block*2],hls::stream<data_tt> outStream4[2*block])
{
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=row
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=norm0
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=norm1
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=row1
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=col

#pragma HLS DEPENDENCE variable=out1 inter false
#pragma HLS DEPENDENCE variable=out2 inter false
#pragma HLS DEPENDENCE variable=out3 inter false
#pragma HLS DEPENDENCE variable=out4 inter false

#pragma HLS ARRAY_PARTITION dim=0 type=complete variable=out1
#pragma HLS ARRAY_PARTITION dim=0 type=complete variable=out2
#pragma HLS ARRAY_PARTITION dim=0 type=complete variable=out3
#pragma HLS ARRAY_PARTITION dim=0 type=complete variable=out4

int temp1_row1=row[0][(edge/avg_degree)-1];
int temp2_row1=row[1][(edge/avg_degree)-1];
int temp3_row1=row[2][(edge/avg_degree)-1];



data_tt temp[node][block*2][avg_degree];
#pragma HLS BIND_STORAGE variable=temp type=fifo impl=uram
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=temp
#pragma HLS ARRAY_PARTITION dim=3 type=complete variable=temp
	for (int m=0;m<3;m++)
	{
		for(int k=0;k<2;k++)
		{
		for (int i=0 ; i<node; i++)
		{
#pragma HLS PIPELINE
			 for(int j=0;j<block;j++)
			 {
				data_tt temp_in[block][2];
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=temp_in
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=temp_in

				temp_in[j][k]=inStream[j].read();
				for (int d=0;d<avg_degree;d++)
				{
				temp[i][k*block+j][d] = temp_in[j][k];
				}
			 }
		}
		}

		for (int i=0 ; i<edge/avg_degree ; i++)
		{
#pragma HLS PIPELINE
			int temp1_row=row1[0][i];
			int temp1_col=col[0][i];
			int p_row1=0;
			if(i !=0)
			{
				p_row1=row[0][i-1];
			}

			int temp2_row=row1[1][i];
			int temp2_col=col[1][i];
	        int p_row2=row[1][0];
	        if(i !=0)
	        {
	    	   p_row2=row[1][i-1];
	        }


	       int temp3_row=row1[2][i];
	       int temp3_col=col[2][i];
	       int p_row3=row[2][0];
	       if(i !=0)
	       {
	          	p_row3=row[2][i-1];
	       }

	       int temp4_row=row1[3][i];
	       int temp4_col=col[3][i];
	       int p_row4=row[3][0];
	       if(i !=0)
	       {
	    	   p_row4=row[3][i-1];
           }

	       for (int k=0;k<2*block;k++)
	       {
	    	   if(p_row1 == temp1_row)
	    	   {
	     		   out1[k] += temp[temp1_col][k][0]*norm0[0][i];
	    	   }
	    	   else
	     	   {
	    	   	  	outStream1[k] << out1[k] + temp[p_row1][k][1]*norm1[p_row1][0];
	    	   		out1[k]=temp[temp1_col][k][0] * norm0[0][i];
	     	   }

	    	   if(p_row2 == temp2_row)
	    	   {
	    		   out2[k] += temp[temp2_col][k][0]*norm0[1][i];
	      	   }
	    	   else
	       	   {
	       		   outStream2[k] << out2[k] + temp[p_row2][k][1] *norm1[p_row2][0] ;
	       		   out2[k]=temp[temp2_col][k][0] * norm0[1][i];

	       	   }

	    	   if(p_row3 == temp3_row)
	    	   {
	    		   out3[k] += temp[temp3_col][k][2] * norm0[2][i];
	     	   }
	    	   else
	    	   {
	    		   outStream3[k] << out3[k] + temp[p_row3][k][3] *norm1[p_row3][1] ;
	    	   	   out3[k]=temp[temp3_col][k][2]* norm0[2][i] ;
	    	   }

	    	   if(p_row4 == temp4_row)
	    	   {
	    	   	   out4[k] += temp[temp4_col][k][2] * norm0[3][i] ;
	    	   }
	    	   else
	     	   {
 	    		   outStream4[k] << out4[k] + temp[p_row4][k][3] *norm1[p_row4][1] ;
	    	   	   out4[k]=temp[temp4_col][k][2] * norm0[3][i] ;
	    	   }
	       }
		}
		for (int k=0;k<block*2;k++)
	    {
#pragma HLS UNROLL
			outStream4[k] << out4[k] + temp[node-1][k][3] * norm1[node-1][0] ;
	    	if ((temp1_row1 != row[1][0]))
	    	{
	    		outStream1[k] << out1[k] + temp[temp1_row1][k][0] * norm1[temp1_row1][0];
	    	   	outStream1_2[k]  << 0;
	    	   	out1[k]=0;
	    	}
	    	else
	    	{
	    		outStream1_2[k]  << out1[k];
	    		out1[k]=0;
	    	}
	    	if  (temp2_row1 != row[2][0])
	    	{
	    		outStream2[k]  << out2[k]  + temp[temp2_row1][k][1] * norm1[temp2_row1][1];
          	    outStream2_2[k]  << 0;
          	    out2[k]=0;
	    	}
	    	else
	    	{
	    		outStream2_2[k]  << out2[k];
	    		out2[k]=0;
          	}
	    	if (temp3_row1 != row[3][0])
	    	{
           		outStream3[k]  << out3[k] + temp[temp3_row1][k][2] * norm1[temp3_row1][1];
          	    outStream3_2[k]  << 0;
          	    out3[k]=0;
	    	}
	    	else
	    	{
	    		outStream3_2[k]  << out3[k];
	    		out3[k]=0;
	    	}
		}

	}
}








ap_uint<256> sampler( ap_uint<256> seed, int load) {
  static ap_uint<256> mask;
  if (load ==1 )
    mask = seed;
  bool b_32 = mask.get_bit(256-32);
  bool b_104 = mask.get_bit(256-104);
  bool b_248 = mask.get_bit(256-248);
  bool b_1 = mask.get_bit(256-1);
  bool new_bit = b_32^b_104^b_248^b_1 ;
  //mask.range(254,0)=mask.range(255,1);
  //mask.range(511,254)=new_bit;
  mask = mask >> 1;
  mask.set_bit(255, new_bit);

  //return mask.to_uint();
  return mask;

}










void wr(int out12[node],hls::stream<data_tt> outStream1[2*block],hls::stream<data_tt> outStream1_2[2*block],hls::stream<data_tt> outStream2[2*block],hls::stream<data_tt> outStream2_2[2*block],hls::stream<data_tt> outStream3[2*block],hls::stream<data_tt> outStream3_2[2*block],hls::stream<data_tt> outStream4[2*block],data_out out[node][6])
{
	ap_uint<256> mask;
	for(int i=0;i<8;i++)
	{
		mask.range(32 * (i + 1) - 1, i * 32) = 18023893251;

	}

	sampler(mask, 1);
	for (int m=0; m<3; m++)
	{

		for(int l=0;l<node1;l++)
		{
			ap_uint<256> mask;

			mask=sampler(mask,0);

			data_out  tmpOut1_1 = 0;
			data_out  tmpOut1_2 = 0;
			for (int i = 0; i < 2 * block; i++)
			{
#pragma HLS UNROLL
				data_bi mask0=mask.get_bit(i);
				int temp;
				temp=mask0/(1-drop_rate);
				if (i<block)
				{
					tmpOut1_1.range(32 * (i + 1) - 1, i * 32) = (temp*(outStream1[i].read())).range(31, 0);
				}
				else
				{
					tmpOut1_2.range(32 * (i-block + 1) - 1, (i-block) * 32) = (temp*(outStream1[i].read())).range(31, 0);
				}
			}
			out[l][2*m].write(tmpOut1_1);
			out[l][2*m+1].write(tmpOut1_2);
		}


		for(int l=0;l < node2;l++)
		{
			ap_uint<256> mask;
			mask=sampler(mask,0);
			data_out  tmpOut2_1= 0;
			data_out  tmpOut2_2 = 0;
			for (int i = 0; i < 2 * block; i++)
			{
#pragma HLS UNROLL
				data_bi mask0=mask.get_bit(i);
				int temp;
								temp=mask0/(1-drop_rate);
				if (i<block)
			    {
					if (l==0)
					{
						tmpOut2_1.range(32 * (i + 1) - 1, i * 32)=(temp*(outStream2[i].read() +  outStream1_2[i].read())).range(31, 0);
					}
					else
					{
						tmpOut2_1.range(32 * (i + 1) - 1, i * 32) = (temp*outStream2[i].read()).range(31, 0);
					}
			    }
				else
				{
					if (l==0)
					{
						tmpOut2_2.range(32 * (i-block + 1) - 1, (i-block) * 32) = (temp*(outStream2[i].read()+  outStream1_2[i].read())) .range(31, 0);
					}
					else
					{
						tmpOut2_2.range(32 * (i-block + 1) - 1, (i-block) * 32) = (temp*outStream2[i].read()).range(31, 0);
					}

				}
			}
			out[l+node1][2*m].write(tmpOut2_1);
			out[l+node1][2*m+1].write(tmpOut2_2);
		}
		for(int l=0;l<node3;l++)
		{
			ap_uint<256> mask;
			mask=sampler(mask,0);
			data_out  tmpOut3_1= 0;
			data_out  tmpOut3_2 = 0;
			for (int i = 0; i < 2* block; i++)
			{
#pragma HLS UNROLL
				data_bi mask0=mask.get_bit(i);
				int temp;
								temp=mask0/(1-drop_rate);
				if (i<block)
			    {
					if (l==0)
					{
						tmpOut3_1.range(32 * (i + 1) - 1, i * 32)=(temp*(outStream3[i].read() +  outStream2_2[i].read())).range(31, 0);
					}
					else
					{
						tmpOut3_1.range(32 * (i + 1) - 1, i * 32) = (temp*outStream3[i].read()).range(31, 0);
					}
			    }
				else
				{
					if (l==0)
					{
						tmpOut3_2.range(32 * (i-block + 1) - 1, (i-block) * 32) = (temp*(outStream3[i].read()+  outStream2_2[i].read())) .range(31, 0);
					}
					else
					{
						tmpOut3_2.range(32 * (i-block + 1) - 1, (i-block) * 32) =(temp*outStream3[i].read()).range(31, 0);
					}
				}
			}
		    out[l+node1+node2][2*m].write(tmpOut3_1);
		    out[l+node1+node2][2*m+1].write(tmpOut3_2);
		}

	    for(int l=0;l<node4;l++)
	    {
			ap_uint<256> mask;
			mask=sampler(mask,0);
			data_out  tmpOut4_1= 0;
			data_out  tmpOut4_2 = 0;
			for (int i = 0; i < 2* block; i++)
			{
#pragma HLS UNROLL
				data_bi mask0=mask.get_bit(i);
				int temp=mask0/(1-drop_rate);

				if (i<block)
				{
					if (l==0)
					{
						tmpOut4_1.range(32 * (i + 1) - 1, i * 32)=((outStream4[i].read()+ outStream3_2[i].read())).range(31, 0);
					}
					else
					{

						tmpOut4_1.range(32 * (i + 1) - 1, i * 32) = (temp*(outStream4[i].read())).range(31, 0);
					}
				}
				else
				{
					if (l==0)
					{
						tmpOut4_2.range(32 * (i-block + 1) - 1, (i-block) * 32) = (temp*(outStream4[i].read()+ outStream3_2[i].read())) .range(31, 0);
					}
					else
					{

						tmpOut4_2.range(32 * (i-block + 1) - 1, (i-block) * 32) = (temp*(outStream4[i].read())).range(31, 0);
					}
				}
			}
		    out[l+node1+node2+node3][2*m].write(tmpOut4_1);
		    out[l+node1+node2+node3][2*m+1].write(tmpOut4_2);
		}
	}
}







void TopFun(int out12[node],data_in in[node][3], data_tt beta[node],data_t col[4][edge/4],data_t row[4][edge/4],data_t row1[4][edge/4],data_tt norm0[4][edge/4],data_tt norm1[node][2],data_out out[node][6])
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
	data_t in_buf[node][feature];
#pragma HLS BIND_STORAGE variable=in_buf type=ram_2p

	read_data1(in,in_buf);



	data_t temp_weight[feature][block][6];
	data_tt temp_alpha[block][6];
	data_tt temp_bias[block][6];
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
    wr(out12,outStream1,outStream1_2,outStream2,outStream2_2,outStream3,outStream3_2,outStream4,out);
}


