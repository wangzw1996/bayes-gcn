#include "test.h"
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




data_bi data[2708][1433];
data_tt beta[2708];
data_t out2[2708];
data_tt out[2708];
data_tt out11[block];
data_out out12[node][6];

data_tt out10[node];

data_t row[4][2639];
data_t row1[4][2639];
data_t col[4][2639];
data_out out0[node][6];
data_tt temp_out0[node1][block];
data_tt temp_out1[node][block][6];


int main()
{
	string line;
	int i_data=0;
	ifstream fin_data("/mnt/ccnas2/bdp/zw4520/data2/cora.csv");
	ifstream fin_beta("/mnt/ccnas2/bdp/zw4520/data2/beta0(1).csv");
	ifstream fin_row("/mnt/ccnas2/bdp/zw4520/data1/cora_row.csv");
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
				{
					data[i_data][j]=0;
				}
				else
				{
					data[i_data][j]=1;
				}
			}

			i_data=i_data+1;
		}


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
		if (7916 < i_row)
			row[3][i_row-7917]=abs(atof(temp1.c_str()));
		row1[3][i_row-7917]=abs(atof(temp1.c_str()));
		i_row=i_row+1;
	}
	cout <<row[3][2638]<< endl;
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

	cout <<col[0][0]<< endl;
	cout <<col[0][1]<< endl;
	cout <<col[0][2]<< endl;
	cout <<col[0][3]<< endl;

	cout <<row[0][0]<< endl;
		cout <<row[0][1]<< endl;
		cout <<row[0][2]<< endl;
		cout <<row[0][3]<< endl;









	cout <<out12[0]<< endl;
	cout <<out12[1]<< endl;
	cout <<out12[2]<< endl;


	TopFun(out11,data,beta,col,row,row1,out0);
	//TopFun1(out10,data,beta,out0);





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
			cout <<temp_out1[2707][39][5]<< endl;
			cout <<temp_out1[2707][40][5]<< endl;
			cout <<temp_out1[2707][41][5]<< endl;
			cout <<temp_out1[2707][42][5]<< endl;
			cout <<out10[0]<< endl;
			cout <<out11[0]<< endl;
			cout <<out11[1]<< endl;
			cout <<out11[2]<< endl;
			cout <<out11[3]<< endl;
			cout <<out11[4]<< endl;
			cout <<out11[5]<< endl;
			cout <<temp_out1[2][0][0]<< endl;
			cout <<temp_out1[1000][40][5]<< endl;


}
