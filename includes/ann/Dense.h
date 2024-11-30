#pragma once
#include<vector>
#include<memory>
#include<string>
#include"../autodiff/Variable.h"

class Dense
{
private:
	vector<Variable> bias;
	vector<vector<Variable>> weights;
	vector<vector<double>> weightsHistory1;
	vector<double> biasHistory1;
	vector<vector<double>> weightsHistory2;
	vector<double> biasHistory2;
	int inputShape, noOfNeurons;
	string activationFunctionName;

public:
	Dense(int inputShape, int noOfNeurons, const string& activationFunctionName);
	void initialize();
	vector<vector<Variable>> forwardPass(vector<vector<Variable>>& inputs);
	vector<vector<Variable>> activationFunction(vector<vector<Variable>>& z);
	void updateWeightsAndBiases(double learningRate, string& optimizationName, int epoch);
	void getWeights();
	void getBiases();
	int getInputShape();
	int getNoOfNeurons();
};

