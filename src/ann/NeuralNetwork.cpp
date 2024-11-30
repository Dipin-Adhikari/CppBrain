#include "../../includes/ann/NeuralNetwork.h"
#include<chrono>
#include<string>
#include<iostream>
#include<algorithm>

using namespace std;

NeuralNetwork::NeuralNetwork() {
	isInitialized = false;
};

void NeuralNetwork::addLayers(Dense layer) {
	this->layers.push_back(layer);
}

void NeuralNetwork::clearComputationalGraph(vector<vector<Variable>>& variables) {
	for (auto& row : variables) {
		for (auto& var : row) {
			var.clearGraph();
		}
	}
}

void NeuralNetwork::initializeLayers() {
	for (auto& layer : layers) {
		layer.initialize();
	}
	isInitialized = true;
}


void NeuralNetwork::compile(string lossFunctionName, string optimizationName, double learningRate) {
	this->lossFunctionName = lossFunctionName;
	this->optimizationName = optimizationName;
	this->learningRate = learningRate;
}


vector<vector<Variable>> NeuralNetwork::predict(vector<vector<double>>& inputs) {
	vector<vector<Variable>> variableInputs(inputs.size(), vector<Variable>(inputs[0].size()));
	for (size_t i = 0; i < inputs.size(); ++i) {
		for (size_t j = 0; j < inputs[0].size(); ++j) {
			variableInputs[i][j] = Variable(inputs[i][j]);
		}
	}

	vector<vector<Variable>> output = variableInputs;
	for (auto& layer : layers) {
		output = layer.forwardPass(output);
	}
	return output;
}

Variable NeuralNetwork::lossFunction(vector<vector<Variable>>& yPred, vector<vector<double>>& y) {

	if (lossFunctionName == "mse") {
		Variable loss(0.0);
		for (size_t i = 0; i < yPred.size(); ++i) {
			Variable error = Variable(y[i][0]) - yPred[i][0];
			loss = loss + error * error;
		}
		return loss / Variable(yPred.size());
	}

	if (lossFunctionName == "categorical-cross-entropy") {
		Variable loss(0.0);
		for (size_t i = 0; i < yPred.size(); ++i) {
			for (size_t j = 0; j < yPred[0].size(); ++j) {
				Variable term = Variable(y[i][j]) * (yPred[i][j]).logarithm();

				loss = loss - term;
			}
		}

		loss = loss / Variable(yPred.size());
		return loss;
	}

	if (lossFunctionName == "binary-cross-entropy") {
		Variable loss(0.0);
		for (size_t i = 0; i < yPred.size(); ++i) {
			for (size_t j = 0; j < yPred[0].size(); ++j) {
				loss = loss - (Variable(y[i][j]) * yPred[i][j].logarithm() + (1 - Variable(y[i][j])) * (1 - yPred[i][j]).logarithm());
			}
		}
		return loss / Variable(yPred.size());
	}


}

vector<double> NeuralNetwork::evaluate(vector<vector<double>>& inputs, vector<vector<double>>& outputs) {

	vector<vector<Variable>> predictedOutputsV = predict(inputs);

	vector<vector<double>> predictedOutputs(predictedOutputsV.size(), vector<double>(predictedOutputsV[0].size()));
	int noOfCorrectPrediction = 0;
	for (size_t i = 0; i < predictedOutputsV.size(); ++i) {
		for (size_t j = 0; j < predictedOutputsV[0].size(); ++j) {
			predictedOutputs[i][j] = predictedOutputsV[i][j].getValue();
		}
		auto maxElement = *max_element(predictedOutputs[i].begin(), predictedOutputs[i].end());
		int maxIndex = 0;
		for (size_t j = 0; j < predictedOutputs[0].size(); ++j) {
			if (maxElement == predictedOutputs[i][j]) {
				maxIndex = j;
			}
		}

		if (outputs[i][maxIndex] == 1) {
			noOfCorrectPrediction++;
		}
	}
	double accuracy = (double)noOfCorrectPrediction / inputs.size();
	Variable loss = lossFunction(predictedOutputsV, outputs);
	

	return {accuracy, (double)loss.getValue()};
}


void NeuralNetwork::fit(vector<vector<double>>& trainingInputs, vector<vector<double>>& trainingOutputs, int epochs, vector<vector<double>>& validationInputs, vector<vector<double>>& validationOutputs, int batchSize) {


	if (!isInitialized) {
		initializeLayers();
	}

	for (size_t epoch = 0; epoch < epochs; ++epoch) {
		auto startTime = chrono::high_resolution_clock::now();
		Variable trainingLoss(0.0);

		size_t loopIterationTrain = trainingInputs.size() / batchSize;
		for (size_t batch = 0; batch < loopIterationTrain; ++batch) {

			vector<vector<double>> batchTrainingInputs;
			vector<vector<double>> batchTrainingOutputs;

			if (batch == (trainingInputs.size() / batchSize)-1) {
				batchTrainingInputs = vector<vector<double>>(trainingInputs.begin() + batch * batchSize, trainingInputs.end());
				batchTrainingOutputs = vector<vector<double>>(trainingOutputs.begin() + batch * batchSize, trainingOutputs.end());
			}
			else {
				batchTrainingInputs = vector<vector<double>>(trainingInputs.begin() + batch * batchSize, trainingInputs.begin() + batch * batchSize + batchSize);
				batchTrainingOutputs = vector<vector<double>>(trainingOutputs.begin() + batch * batchSize, trainingOutputs.begin() + batch * batchSize + batchSize);
			}

			vector<vector<Variable>> yTrainPred = predict(batchTrainingInputs);
			trainingLoss = lossFunction(yTrainPred, batchTrainingOutputs);
			trainingLoss.backward();

			for (auto& layer : layers) {
				layer.updateWeightsAndBiases(learningRate, optimizationName, epoch);
			}

			clearComputationalGraph(yTrainPred);
		}

		Variable validationLoss(0.0);

		size_t loopIterationVal = validationInputs.size() / batchSize;
		for (size_t batch = 0; batch < loopIterationVal; ++batch) {
			vector<vector<double>> batchValidationInputs;
			vector<vector<double>> batchValidationOutputs;

			if (batch == (validationInputs.size() / batchSize)-1) {
				batchValidationInputs = vector<vector<double>>(validationInputs.begin() + batch * batchSize, validationInputs.end());
				batchValidationOutputs = vector<vector<double>>(validationOutputs.begin() + batch * batchSize, validationOutputs.end());
			}
			else {
				batchValidationInputs = vector<vector<double>>(validationInputs.begin() + batch * batchSize, validationInputs.begin() + batch * batchSize + batchSize);
				batchValidationOutputs = vector<vector<double>>(validationOutputs.begin() + batch * batchSize, validationOutputs.begin() + batch * batchSize + batchSize);
			}

			vector<vector<Variable>> yValidationPred = predict(batchValidationInputs);
			validationLoss = lossFunction(yValidationPred, batchValidationOutputs);
			validationLoss.backward();

			clearComputationalGraph(yValidationPred);
		}

		auto endTime = chrono::high_resolution_clock::now();
		chrono::duration<double> elapsedTime = endTime - startTime;

		cout << endl << "Epoch: " << epoch << "/" << epochs << endl << "-  " << elapsedTime.count() << "s  -  " << "Loss: " << trainingLoss.getValue() << "   -   Validation Loss: " << validationLoss.getValue() << endl;
	}
}