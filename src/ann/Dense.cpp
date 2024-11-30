#include "../../includes/ann/Dense.h"
#include"../../includes/autodiff/Variable.h"
#include<vector>
#include<random>
#include<iostream>

using namespace std;

Dense::Dense(int inputShape, int noOfNeurons, const string& activationFunctionName) {
	this->inputShape = inputShape;
	this->noOfNeurons = noOfNeurons;
	this->activationFunctionName = activationFunctionName;
}

void Dense::initialize() {
	weights.resize(inputShape, vector<Variable>(noOfNeurons));
	bias.resize(noOfNeurons);
	weightsHistory1.resize(inputShape, vector<double>(noOfNeurons, 0.0));
	biasHistory1.resize(noOfNeurons);
	weightsHistory2.resize(inputShape, vector<double>(noOfNeurons, 0.0));
	biasHistory2.resize(noOfNeurons);

	int fanIn = inputShape;
	int fanOut = noOfNeurons;
	double limit;
	if (activationFunctionName == "relu") {
		limit = sqrt(2.0 / (fanIn));
	}
	if (activationFunctionName == "sigmoid" || activationFunctionName == "softmax" || activationFunctionName == "tanh") {
		limit = sqrt(2.0 / (fanIn + fanOut));
	}

	mt19937 gen(static_cast<unsigned int>(42));
	uniform_real_distribution<double> distrib(-limit, limit);

	for (size_t i = 0; i < inputShape; ++i) {
		for (size_t j = 0; j < noOfNeurons; ++j) {
			weights[i][j] = Variable(distrib(gen));
		}
	}

	for (size_t i = 0; i < noOfNeurons; ++i) {
		bias[i] = Variable(distrib(gen));
	}
}

vector<vector<Variable>> Dense::forwardPass(vector<vector<Variable>>& inputs) {
	vector<vector<Variable>> sumArray(inputs.size(), vector<Variable>(noOfNeurons));

	for (size_t i = 0; i < inputs.size(); ++i) {
		for (size_t j = 0; j < noOfNeurons; ++j) {
			sumArray[i][j] = bias[j];

			for (size_t k = 0; k < inputShape; ++k) {
				Variable prod = inputs[i][k] * weights[k][j];
				sumArray[i][j] = sumArray[i][j] + prod;
			}
		}
	}

	return activationFunction(sumArray);
}

vector<vector<Variable>> Dense::activationFunction(vector<vector<Variable>>& z) {
	vector<vector<Variable>> activatedZ(z.size(), vector<Variable>(z[0].size()));
	if (activationFunctionName == "sigmoid") {
		for (size_t i = 0; i < z.size(); ++i) {
			for (size_t j = 0; j < z[0].size(); ++j) {
				activatedZ[i][j] = 1 / (1 + (-z[i][j]).exponential());
			}
		}
		return activatedZ;
	}

	else if (activationFunctionName == "tanh") {
		for (size_t i = 0; i < z.size(); ++i) {
			for (size_t j = 0; j < z[0].size(); ++j) {
				activatedZ[i][j] = (z[i][j].exponential() - (-z[i][j]).exponential()) / (z[i][j].exponential() + (-z[i][j]).exponential());
			}
		}
		return activatedZ;
	}

	else if (activationFunctionName == "relu") {
		for (size_t i = 0; i < z.size(); ++i) {
			for (size_t j = 0; j < z[0].size(); ++j) {
				if (z[i][j].getValue() > 0) {
					activatedZ[i][j] = z[i][j];
				}
				else {
					activatedZ[i][j] = 0;
				}
			}
		}
		return activatedZ;
	}

	else if (activationFunctionName == "softmax") {
		for (size_t i = 0; i < z.size(); ++i) {

			double maxNum = z[i][0].getValue();
			for (size_t j = 0; j < z[0].size(); ++j) {
				if (maxNum < z[i][j].getValue()) {
					maxNum = z[i][j].getValue();
				}
			}

			Variable sum = 0.0;
			vector<Variable> expValues;

			for (size_t j = 0; j < z[0].size(); ++j) {
				Variable expVal = (z[i][j] - maxNum).exponential();
				expValues.push_back(expVal);
				sum = sum + expVal;
			}

			for (size_t j = 0; j < z[0].size(); ++j) {
				activatedZ[i][j] = expValues[j] / sum;
			}
		}
		return activatedZ;
	}
	else {
		return z;
	}
}

void Dense::updateWeightsAndBiases(double learningRate, string& optimizationName, int epoch = 0) {
	if (optimizationName == "momentum") {
		double beta = 0.9;
		for (size_t i = 0; i < inputShape; ++i) {
			for (size_t j = 0; j < noOfNeurons; ++j) {
				double history = learningRate * weights[i][j].getGrad();
				history = history + beta * weightsHistory1[i][j];
				weights[i][j] = weights[i][j] - history;
				weightsHistory1[i][j] = history;
			}
		}
		for (size_t i = 0; i < noOfNeurons; ++i) {
			double history = learningRate * bias[i].getGrad();
			history = history + beta * biasHistory1[i];
			bias[i] = bias[i] - history;
			biasHistory1[i] = history;
		}
	}
	if (optimizationName == "adam") {
		double beta1 = 0.9;
		double beta2 = 0.999;
		double epsilon = 1e-08;
		for (size_t i = 0; i < inputShape; ++i) {
			for (size_t j = 0; j < noOfNeurons; ++j) {
				double history1 = (1 - beta1) * weights[i][j].getGrad();
				history1 = history1 + beta1 * weightsHistory1[i][j];

				double history2 = (1 - beta2) * pow(weights[i][j].getGrad(), 2);
				history2 = history2 + beta2 * weightsHistory2[i][j];

				double historyHat1 = history1 / (1 - pow(beta1, epoch + 1));
				double historyHat2 = history2 / (1 - pow(beta2, epoch + 1));

				weights[i][j] = weights[i][j] - Variable((learningRate * historyHat1) / sqrt(historyHat2 + epsilon));

				weightsHistory1[i][j] = history1;
				weightsHistory2[i][j] = history2;
			}
		}

		for (size_t i = 0; i < noOfNeurons; ++i) {
			double history1 = (1 - beta1) * bias[i].getGrad();
			history1 = history1 + beta1 * biasHistory1[i];

			double history2 = (1 - beta2) * pow(bias[i].getGrad(), 2);
			history2 = history2 + beta2 * biasHistory2[i];

			double historyHat1 = history1 / (1 - pow(beta1, epoch + 1));
			double historyHat2 = history2 / (1 - pow(beta2, epoch + 1));

			bias[i] = bias[i] - Variable((learningRate * historyHat1) / sqrt(historyHat2 + epsilon));

			biasHistory1[i] = history1;
			biasHistory2[i] = history2;
		}

	}
	else {
		for (size_t i = 0; i < inputShape; ++i) {
			for (size_t j = 0; j < noOfNeurons; ++j) {
				weights[i][j] = weights[i][j] - Variable(learningRate * weights[i][j].getGrad());
			}
		}
		for (size_t i = 0; i < noOfNeurons; ++i) {
			bias[i] = bias[i] - Variable(learningRate * bias[i].getGrad());
		}
	}
}


void Dense::getWeights() {
	for (size_t i = 0; i < inputShape; ++i) {
		cout << "Weights of " << inputShape << " layer:  ";
		for (size_t j = 0; j < noOfNeurons; ++j) {
			cout << weights[i][j].getValue() << "\t";
		}
		cout << endl;
	}
}

void Dense::getBiases() {
	for (size_t i = 0; i < noOfNeurons; ++i) {
		cout << "Biases " << bias[i].getValue() << "\t";
	}
	cout << endl;
}

int Dense::getInputShape() {
	return inputShape;
}

int Dense::getNoOfNeurons() {
	return noOfNeurons;
}