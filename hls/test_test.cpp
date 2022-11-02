#include "8.h"
#include<hls_stream.h>
#include <stdint.h>
#include <cmath>
#include<stdlib.h>
#include<iostream>
#include"ap_int.h"
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include"ap_fixed.h"

using namespace std;
string Trim(string& str)
{
str.erase(0,str.find_first_not_of(" \t\r\n"));
str.erase(str.find_last_not_of(" \t\r\n") + 1);
return str;
}



data_tt out10[node];
data_tt out11[node];
data_in data[2708][3];
data_tt beta[2708];
data_tt norm1[2708][2];

data_tt out1[node1][block];
data_tt out2[node2][block];
data_tt out3[node3][block];
data_tt out4[node4][block];

data_t row[4][2639];
data_t row1[4][2639];
data_t col[4][2639];
data_tt norm0[4][2639];
data_out out0[node][6];
data_tt temp_out0[node][8];
data_tt temp_out1[node][block][8];
data_tt out[node][hidden];
int out12[node];


int main()
{
	string line;
	int i_data=0;
	ifstream fin_data("/mnt/ccnas2/bdp/zw4520/data2/cora.csv");
	ifstream fin_beta("/mnt/ccnas2/bdp/zw4520/data2/beta0(1).csv");
	ifstream fin_norm1("/mnt/ccnas2/bdp/zw4520/data2/cora_norm1.csv");
	ifstream fin_row("/mnt/ccnas2/bdp/zw4520/data1/cora_row.csv");
	ifstream fin_norm0("/mnt/ccnas2/bdp/zw4520/data2/cora_norm0.csv");
	ifstream fin_col("/mnt/ccnas2/bdp/zw4520/data1/cora_col.csv");

	data_bi temp_in[1536]={0};
	while (getline(fin_data, line))
		{
			istringstream sin(line);
			vector<string> fields;
			string field;
			while (getline(sin, field, ','))
			{
				fields.push_back(field);
			}
			for(int j=0;j<1433;j++)
			{
				string temp1 = Trim(fields[j]);
				float temp2=atof(temp1.c_str());
				if (temp2 < 0)
					temp_in[j]=0;
				else
					temp_in[j]=1;
			}
			data_in temp=0;
			for(int i=0;i<3;i++)
			{
				for(int k=0;k<512;k++)
				{
					temp.range(k,k)=temp_in[i*512+k];
				}
			data[i_data][i]=temp;
			}

			i_data=i_data+1;
		}

		cout <<data[2707][1]<< endl;
	int i_beta=0;
	while (getline(fin_beta, line))
	{
		stringstream sin(line);
		vector<string> fields;
		string field;
		while (getline(sin, field, ','))
		{
			fields.push_back(field);
		}
		string temp1 = Trim(fields[0]);
		beta[i_beta]=abs(atof(temp1.c_str()));
		i_beta=i_beta+1;
}

	int i_norm1=0;
		while (getline(fin_norm1, line))
		{
			stringstream sin(line);
			vector<string> fields;
			string field;
			while (getline(sin, field, ','))
			{
				fields.push_back(field);
			}
			string temp1 = Trim(fields[0]);
			norm1[i_norm1][0]=abs(atof(temp1.c_str()));
			norm1[i_norm1][1]=abs(atof(temp1.c_str()));
			i_norm1=i_norm1+1;
	}


	int i_row=0;
	while (getline(fin_row, line))
	{
		stringstream sin(line);
		vector<string> fields;
		string field;
		while (getline(sin, field, ','))
		{
			fields.push_back(field);
		}
		string temp1 = Trim(fields[0]);
		if (i_row<2639)
			row[0][i_row]=abs(atof(temp1.c_str()));
		    row1[0][i_row]=abs(atof(temp1.c_str()));
		if (2638 <i_row < 5278)
			row[1][i_row-2639]=abs(atof(temp1.c_str()));
		    row1[1][i_row-2639]=abs(atof(temp1.c_str()));
		if (5277 <i_row < 7917)
			row[2][i_row-5278]=abs(atof(temp1.c_str()));
		    row1[2][i_row-5278]=abs(atof(temp1.c_str()));
		if (7916 < i_row)
			row[3][i_row-7917]=abs(atof(temp1.c_str()));
		    row1[3][i_row-7917]=abs(atof(temp1.c_str()));
		i_row=i_row+1;
	}
	cout <<row[3][2638]<< endl;

	int i_norm0=0;
		while (getline(fin_norm0, line))
		{
			stringstream sin(line);
			vector<string> fields;
			string field;
			while (getline(sin, field, ','))
			{
				fields.push_back(field);
			}
			string temp1 = Trim(fields[0]);
			if (i_norm0<2639)
				norm0[0][i_norm0]=abs(atof(temp1.c_str()));
			if (2638 <i_norm0 < 5278)
				norm0[1][i_norm0-2639]=abs(atof(temp1.c_str()));
			if (5277 <i_norm0 < 7917)
				norm0[2][i_norm0-5278]=abs(atof(temp1.c_str()));
			if (7916 < i_norm0)
			    norm0[3][i_norm0-7917]=abs(atof(temp1.c_str()));
			i_norm0=i_norm0+1;
		}

	int i_col=0;
	while (getline(fin_col, line))
	{
		stringstream sin(line);
		vector<string> fields;
		string field;
		while (getline(sin, field, ','))
		{
			fields.push_back(field);
		}
		string temp1 = Trim(fields[0]);
		if (i_col<2639)
			col[0][i_col]=abs(atof(temp1.c_str()));
		if (2638 <i_col < 5278)
			col[1][i_col-2639]=abs(atof(temp1.c_str()));
		if (5277 <i_col < 7917)
			col[2][i_col-5278]=abs(atof(temp1.c_str()));
		if (7916 < i_col)
			col[3][i_col-7917]=abs(atof(temp1.c_str()));
		i_col=i_col+1;
	}
	cout <<col[3][2638]<< endl;
	TopFun(out12,data,beta,col,row,row1,norm0,norm1,out0);

	for (int i=0; i<node;i++)
	{
		for(int j=0;j<block;j++)
		{
			for(int k=0;k<6;k++)
			{
				temp_out1[i][j][k].range(31,0)=out0[i][k].range(32 * (j + 1) - 1, j * 32);
			}
		}
	}

	for (int i=0; i<node;i++)
		{
			for(int j=0;j<block;j++)
			{
				for(int k=0;k<6;k++)
				{
					if (k*block +j<hidden)
					{

					out[i][k*block +j]=temp_out1[i][j][k];
					}
				}
			}
		}

	cout << out12[0]<< endl;
	cout <<temp_out1[7][2][5]<< endl;
			cout <<temp_out1[7][1][5]<< endl;
			cout <<temp_out1[7][0][5]<< endl;
			cout <<temp_out1[8][2][5]<< endl;
			cout <<temp_out1[8][1][5]<< endl;
			cout <<temp_out1[8][0][5]<< endl;
			cout <<temp_out1[9][2][5]<< endl;
			cout <<temp_out1[9][1][5]<< endl;
			cout <<temp_out1[9][0][5]<< endl;
			cout <<temp_out1[8][0][2]<< endl;
				cout <<temp_out1[9][2][3]<< endl;
				cout <<temp_out1[9][1][4]<< endl;
				cout <<temp_out1[9][0][1]<< endl;
				cout <<temp_out1[0][0][0]<< endl;
				cout <<temp_out1[1][0][0]<< endl;
				cout <<temp_out1[1][1][0]<< endl;

				cout <<temp_out1[649][0][0]<< endl;
				cout <<temp_out1[649][42][3]<< endl;
				cout <<temp_out1[651][42][3]<< endl;
				cout <<temp_out1[652][42][3]<< endl;
				cout <<temp_out1[700][42][3]<< endl;
				cout <<temp_out1[650][42][3]<< endl;
				cout <<temp_out1[651][0][0]<< endl;
			    cout <<temp_out1[650][0][0]<< endl;

			    cout <<temp_out1[1358][42][4]<< endl;
			    cout <<temp_out1[1358][42][3]<< endl;
			    cout <<temp_out1[1358][0][0]<< endl;
			    cout <<temp_out1[1502][0][0]<< endl;
			    cout <<temp_out1[1502][42][3]<< endl;

			    cout <<temp_out1[1940][42][4]<< endl;
			    cout <<temp_out1[1940][42][3]<< endl;
			    cout <<temp_out1[1940][0][0]<< endl;
			    cout <<temp_out1[1941][0][0]<< endl;
			    cout <<temp_out1[1941][42][4]<< endl;
			    cout <<temp_out1[2000][0][0]<< endl;

				cout <<temp_out1[2707][0][0]<< endl;
				cout <<temp_out1[2707][40][3]<< endl;
				cout <<temp_out1[2707][41][2]<< endl;
				cout <<temp_out1[2707][42][4]<< endl;
				cout <<temp_out1[2][0][0]<< endl;
				cout <<temp_out1[1000][40][5]<< endl;
				cout <<temp_out1[653][0][0]<< endl;

                ofstream dataFile;
				dataFile.open("/mnt/ccnas2/bdp/zw4520/data2/out2.csv");

				for(int j=0;j< hidden;j++)
					if (j==255)
				       {
													dataFile << 0 ;
												}
												else
												{
													dataFile << 0 << ",";
												}
				dataFile <<  endl;

				for (int i=0; i<node;i++)
					{

						for(int j=0;j< hidden;j++)
						{
							if (j==255)
							{
								dataFile << out[i][j] ;
							}
							else
							{
								dataFile << out[i][j] << ",";
							}


						}
						dataFile <<  endl;
					}
				dataFile.close();


}
