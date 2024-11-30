#pragma once

#include<vector>
#include<string>
#include"../autodiff/Variable.h"
#include"Dense.h"

class NeuralNetwork
{
private:
	vector<Dense> layers;
	double learningRate;
	string lossFunctionName;
	string optimizationName;
	bool isInitialized;
public:
	NeuralNetwork();
	void addLayers(Dense layer);
	void clearComputationalGraph(vector<vector<Variable>>& variables);
	void initializeLayers();
	void compile(string lossFunctionName, string optimizationName, double learningRate=1);
	vector<vector<Variable>> predict(vector<vector<double>>& inputs);
	Variable lossFunction(vector<vector<Variable>>& yPred, vector<vector<double>>& y);
	vector<double> evaluate(vector<vector<double>>& inputs, vector<vector<double>>& outputs);
	void fit(vector<vector<double>>& trainingInputs, vector<vector<double>>& trainingOutputs, int epochs, vector<vector<double>>& validationInputs, vector<vector<double>>& validationOutputs, int batchSize=8);

};

