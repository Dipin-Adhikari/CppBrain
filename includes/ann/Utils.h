#pragma once
#include<iostream>
#include<vector>

using namespace std;
class Utils
{
public:
	static vector<vector<vector<double>>> trainTestSplit(vector<vector<double>>& inputs, double trainSize, int randomState=42);
	static vector<vector<double>> readCSV(const string& filename);
	static vector<vector<vector<double>>> separateInputsOutputs(vector<vector<double>>& inputs, int index);
	static vector<vector<double>> convertCategoricalToOneHot(vector<vector<double>>& output);
};

