#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include"includes/ann/Dense.h"
#include"includes/ann/NeuralNetwork.h"
#include"includes/ann/Utils.h"
using namespace std;


int main() {
	
	string filename = "D:/ML in C++/ANN/iris.csv";
	vector<vector<double>> inputs = Utils::readCSV(filename);

	vector<vector<vector<double>>> output = Utils::trainTestSplit(inputs, 0.7, 42);
	vector<vector<double>> trainingInputs = output[0];
	vector<vector<double>> testingInputs = output[1];

	output = Utils::trainTestSplit(testingInputs, 0.5);
	testingInputs= output[0];
	vector<vector<double>> validationInputs = output[1];

	output = Utils::separateInputsOutputs(trainingInputs, 4);
	trainingInputs = output[0];
	vector<vector<double>> trainingOutputs = output[1];
	output = Utils::separateInputsOutputs(testingInputs, 4);
	testingInputs = output[0];
	vector<vector<double>> testingOutputs = output[1];
	output = Utils::separateInputsOutputs(validationInputs, 4);
	validationInputs = output[0];
	vector<vector<double>> validationOutputs = output[1];

	trainingOutputs = Utils::convertCategoricalToOneHot(trainingOutputs);
	testingOutputs = Utils::convertCategoricalToOneHot(testingOutputs);
	validationOutputs = Utils::convertCategoricalToOneHot(validationOutputs);
	


	NeuralNetwork nn;
	nn.addLayers(Dense(4, 10, "relu"));
	nn.addLayers(Dense(10, 7, "relu"));
	nn.addLayers(Dense(7, 3, "softmax"));

	nn.compile("categorical-cross-entropy", "adam", 0.02);

	nn.fit(trainingInputs, trainingOutputs,50, validationInputs, validationOutputs, 16);

	vector<double> metrics = nn.evaluate(testingInputs, testingOutputs);
	cout << endl << "Accuracy: " << metrics[0] << "\tLoss: " << metrics[1] << endl;


	return 0;
}