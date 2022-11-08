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

#ifndef DESIGN_H
#define  DESIGN_H



void read_data(data_in in[node][input_block],data_t out[node][feature_block])
{
	#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=out
	#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=in
	for (int j = 0; j < node; j++)
	{
		ap_uint<32> temp;

		for(int m=0; m < input_block; m++)
		{
#pragma HLS UNROLL
			for (int i = 0; i < feature_block; i++)
			{
#pragma HLS UNROLL
				for (int k = 0; k < 32; k++)
				{
#pragma HLS UNROLL
					if (m*feature_block+k<feature)
				    {
						temp.range(k, k)  = in[j][m].range(i*32+k,i*32+k);
				    }
				    else
				    {
				    	temp.range(k,k) = 0;
				    }
				}
			    if (m*feature_block+i< feature)
			    {
			    	out[j][(m*feature_block)+i]=temp;
			    }
			}
		}
	}
}




void Extraction(data_t in[node][feature_block], data_t weight[feature_block][block][cycle],data_tt beta[node],data_tt alpha[block][cycle],data_tt bias[block][cycle],hls::stream<data_tt> inStream[block])
{
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=weight
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=weight
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=alpha
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=bias
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=in

	for (int m=0; m< cycle/2 ;m++)
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
		    for(int k=0;k<feature_block;k++)
		    {
#pragma HLS UNROLL
			    temp = in[l][k]^weight[k][i][m*2+j];
			    for(int p=0;p<32;p++)
			    {
#pragma HLS UNROLL
				    if (k*32+p < feature)
				    {
					    data_bi temp2;
					    temp2=temp.range( (1* (p + 1) -1), p * 1);
				 	    bitcount1+=temp2;
				    }
			    }
		    }
		    data_tt temp1=0;
		    temp1 = alpha[i][m*2+j]*beta[l]* (feature-2*bitcount1);
		    inStream[i] << temp1 +bias[i][m*2+j];
	     }
		}
    }
	}

}


void cc(data_t weight_in[feature_block][hidden],data_tt alpha_in[hidden],data_tt bias_in[hidden],data_t weight_out[feature_block][block][cycle],data_tt alpha_out[block][cycle],data_tt bias_out[block][cycle])
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
	for(int i=0;i<feature_block;i++)
	{
		for(int k=0;k<cycle; k++)
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
#pragma HLS BIND_STORAGE variable=temp type=ram_2p impl=uram
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=temp
#pragma HLS ARRAY_PARTITION dim=3 type=complete variable=temp
	for (int m=0;m< cycle/2;m++)
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








ap_uint<256> sampler1( ap_uint<256> seed, int load)
{
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

ap_uint<256> sampler2( ap_uint<256> seed, int load)
{
  static ap_uint<256> mask;
  if (load ==1 )
    mask = seed;
  bool b_48 = mask.get_bit(256-48);
  bool b_108 = mask.get_bit(256-108);
  bool b_182 = mask.get_bit(256-183);
  bool b_207 = mask.get_bit(256-207);
  bool new_bit = b_48^b_108^b_182^b_207 ;
  //mask.range(254,0)=mask.range(255,1);
  //mask.range(511,254)=new_bit;
  mask = mask >> 1;
  mask.set_bit(255, new_bit);

  //return mask.to_uint();
  return mask;
}


ap_uint<256> sampler3( ap_uint<256> seed, int load)
{
  static ap_uint<256> mask;
  if (load ==1 )
    mask = seed;
  bool b_28 = mask.get_bit(256-28);
  bool b_164 = mask.get_bit(256-164);
  bool b_207 = mask.get_bit(256-207);
  bool b_88 = mask.get_bit(256-88);
  bool new_bit = b_28^b_164^b_207^b_88 ;
  //mask.range(254,0)=mask.range(255,1);
  //mask.range(511,254)=new_bit;
  mask = mask >> 1;
  mask.set_bit(255, new_bit);

  //return mask.to_uint();
  return mask;
}








void wr(hls::stream<data_tt> outStream1[2*block],hls::stream<data_tt> outStream1_2[2*block],hls::stream<data_tt> outStream2[2*block],hls::stream<data_tt> outStream2_2[2*block],hls::stream<data_tt> outStream3[2*block],hls::stream<data_tt> outStream3_2[2*block],hls::stream<data_tt> outStream4[2*block],data_out out[node][cycle])
{
	ap_uint<256> mask1;
	ap_uint<256> mask2;
	ap_uint<256> mask3;
	for(int i=0;i< 8;i++)
	{
		mask1.range(32 * (i + 1) - 1, i * 32) = 18023893251;
		mask2.range(32 * (i + 1) - 1, i * 32) = 18023893251;
		mask3.range(32 * (i + 1) - 1, i * 32) = 18023893251;

	}
	sampler1(mask1, 1);
	sampler2(mask2, 1);
	sampler3(mask3, 1);
	for (int m=0; m<cycle/2; m++)
	{

		for(int l=0;l<node1;l++)
		{
			ap_uint<256> mask;

			mask1=sampler1(mask1,0);
			mask2=sampler2(mask2,0);
			mask3=sampler3(mask2,0);

			data_out  tmpOut1_1 = 0;
			data_out  tmpOut1_2 = 0;
			data_tt temp_scale= 1/(1-drop_rate);
			for (int i = 0; i < 2 * block; i++)
			{
#pragma HLS UNROLL
				data_bi mask0;
				if (drop_rate ==1)
				{
					mask0=mask1.get_bit(i);
				}
				else
				{
					if (drop_rate == 2)
					{
						mask0=mask1.get_bit(i)||mask2.get_bit(i);
					}
					else
					{
						mask0=mask1.get_bit(i)||mask2.get_bit(i)||mask3.get_bit(i);
					}
				}
				data_tt mask_scale=mask0*temp_scale;
				if (i<block)
				{
					data_tt temp1=(outStream1[i].read())*mask_scale;
					tmpOut1_1.range(32 * (i + 1) - 1, i * 32) = temp1.range(31, 0);
				}
				else
				{
					data_tt temp1=(outStream1[i].read())*mask_scale;
					tmpOut1_2.range(32 * (i-block + 1) - 1, (i-block) * 32) = temp1.range(31, 0);
				}
			}
			out[l][2*m].write(tmpOut1_1);
			out[l][2*m+1].write(tmpOut1_2);
		}


		for(int l=0;l < node2;l++)
		{
			ap_uint<256> mask;
			mask1=sampler1(mask1,0);
			mask2=sampler2(mask2,0);
			mask3=sampler2(mask3,0);
			data_out  tmpOut2_1= 0;
			data_out  tmpOut2_2 = 0;
			data_tt temp_scale= 1/(1-drop_rate);
			for (int i = 0; i < 2 * block; i++)
			{
#pragma HLS UNROLL
				data_bi mask0;
				if (drop_rate ==1)
				{
					mask0=mask1.get_bit(i);
				}
				else
				{
					if (drop_rate ==2)
					{
						mask0=mask1.get_bit(i)||mask2.get_bit(i);
					}
					else
					{
						mask0=mask1.get_bit(i)||mask2.get_bit(i)||mask3.get_bit(i);
					}
				}
				data_tt mask_scale=mask0*temp_scale;
				if (i<block)
			    {
					if (l==0)
					{
						data_tt temp1=(outStream2[i].read()+outStream1_2[i].read())*mask_scale;
						tmpOut2_1.range(32 * (i + 1) - 1, i * 32)=temp1.range(31, 0);
					}
					else
					{
						data_tt temp1=(outStream2[i].read())*mask_scale;
						tmpOut2_1.range(32 * (i + 1) - 1, i * 32) = temp1.range(31, 0);
					}
			    }
				else
				{
					if (l==0)
					{
						data_tt temp1=(outStream2[i].read()+outStream1_2[i].read())*mask_scale;
						tmpOut2_2.range(32 * (i-block + 1) - 1, (i-block) * 32) = temp1.range(31, 0);
					}
					else
					{
						data_tt temp1=(outStream2[i].read())*mask_scale;
						tmpOut2_2.range(32 * (i-block + 1) - 1, (i-block) * 32) = temp1.range(31, 0);
					}

				}
			}
			out[l+node1][2*m].write(tmpOut2_1);
			out[l+node1][2*m+1].write(tmpOut2_2);
		}
		for(int l=0;l<node3;l++)
		{
			ap_uint<256> mask;
			mask1=sampler1(mask1,0);
			mask2=sampler2(mask2,0);
			mask3=sampler2(mask3,0);
			data_out  tmpOut3_1= 0;
			data_out  tmpOut3_2 = 0;
			data_tt temp_scale= 1/(1-drop_rate);
			for (int i = 0; i < 2* block; i++)
			{
#pragma HLS UNROLL
				data_bi mask0;
				if (drop_rate ==1)
				{
					mask0=mask1.get_bit(i);
				}
				else
				{
					if (drop_rate ==2)
					{
						 mask0=mask1.get_bit(i)||mask2.get_bit(i);
					}
									else
									{
										 mask0=mask1.get_bit(i)||mask2.get_bit(i)||mask3.get_bit(i);
									}
								}
				data_tt mask_scale=mask0*temp_scale;
				if (i<block)
			    {
					if (l==0)
					{
						data_tt temp1=(outStream3[i].read()+outStream2_2[i].read())*mask_scale;
						tmpOut3_1.range(32 * (i + 1) - 1, i * 32)=temp1.range(31, 0);
					}
					else
					{
						data_tt temp1=(outStream3[i].read())*mask_scale;
						tmpOut3_1.range(32 * (i + 1) - 1, i * 32) = temp1.range(31, 0);
					}
			    }
				else
				{
					if (l==0)
					{
						data_tt temp1=(outStream3[i].read()+outStream2_2[i].read())*mask_scale;
						tmpOut3_2.range(32 * (i-block + 1) - 1, (i-block) * 32) = temp1.range(31, 0);
					}
					else
					{
						data_tt temp1=(outStream3[i].read())*mask_scale;
						tmpOut3_2.range(32 * (i-block + 1) - 1, (i-block) * 32) =temp1.range(31, 0);
					}
				}
			}
		    out[l+node1+node2][2*m].write(tmpOut3_1);
		    out[l+node1+node2][2*m+1].write(tmpOut3_2);
		}

	    for(int l=0;l<node4;l++)
	    {
			ap_uint<256> mask;
			mask1=sampler1(mask1,0);
			mask2=sampler2(mask2,0);
			mask3=sampler2(mask3,0);
			data_out  tmpOut4_1= 0;
			data_out  tmpOut4_2 = 0;
			data_tt temp_scale= 1/(1-drop_rate);

			for (int i = 0; i < 2* block; i++)
			{
#pragma HLS UNROLL
				data_bi mask0;
				if (drop_rate ==0.5)
				{
					 mask0=mask1.get_bit(i);
				}
				else
				{
					if (drop_rate ==0.25)
					{
						 mask0=mask1.get_bit(i)||mask2.get_bit(i);
					}
					else
					{
						 mask0=mask1.get_bit(i)||mask2.get_bit(i)||mask3.get_bit(i);
					}
				}
				data_tt mask_scale=mask0*temp_scale;

				if (i<block)
				{
					if (l==0)
					{

						data_tt temp1=(outStream4[i].read()+outStream3_2[i].read())*mask_scale;
						tmpOut4_1.range(32 * (i + 1) - 1, i * 32)=temp1.range(31, 0);
					}
					else
					{
						data_tt temp1= mask_scale * outStream4[i].read();
						tmpOut4_1.range(32 * (i + 1) - 1, i * 32) = temp1.range(31, 0);
					}
				}
				else
				{
					if (l==0)
					{
						data_tt temp1=(outStream4[i].read()+outStream3_2[i].read())*mask_scale;
						tmpOut4_2.range(32 * (i-block + 1) - 1, (i-block) * 32) = temp1.range(31, 0);
					}
					else
					{
						data_tt temp1=mask_scale * outStream4[i].read();
						tmpOut4_2.range(32 * (i-block + 1) - 1, (i-block) * 32) = temp1.range(31, 0);
					}
				}
			}
		    out[l+node1+node2+node3][2*m].write(tmpOut4_1);
		    out[l+node1+node2+node3][2*m+1].write(tmpOut4_2);
		}
	}
}

#endif
