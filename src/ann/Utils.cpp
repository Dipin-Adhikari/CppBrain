#include<iostream>
#include<vector>
#include<fstream>
#include<sstream>
#include<string>
#include<stdexcept>
#include<algorithm>
#include<random>

#include"../../includes/ann/Utils.h"
using namespace std;

vector<vector<vector<double>>> Utils::trainTestSplit(vector<vector<double>>& inputs, double trainSize, int randomState) {
    vector<vector<double>> trainingInputs;
    vector<vector<double>> testingInputs;

    int trainingInputsSize = inputs.size() * trainSize;
    int testingInputsSize = inputs.size() - trainingInputsSize;

    random_device rd;
    mt19937 rng(randomState);
    shuffle(inputs.begin(), inputs.end(), rng);

    trainingInputs = vector<vector<double>>(inputs.begin(), inputs.begin() + trainingInputsSize);
    testingInputs = vector<vector<double>>(inputs.begin() + trainingInputsSize, inputs.end());

    vector<vector<vector<double>>> output = { trainingInputs, testingInputs };
    return output;

}

vector<vector<double>> Utils::readCSV(const string& filename) {
    vector<vector<double>> data;
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "\033[31m";
        cerr << "Error: Could not open file " << filename << endl;
        cerr << "\033[0m";
        return data;
    }

    string line;

    if (getline(file, line)) {
        cout << "Skipping header: " << line << endl;
    }

    while (getline(file, line)) {
        vector<double> row;
        stringstream ss(line);
        string value;

        while (getline(ss, value, ',')) {
            try {
                if (!value.empty()) {
                    row.push_back(stod(value));
                }
            }
            catch (const invalid_argument& e) {
                cerr << "\033[31m";
                cerr << "Error: Invalid value '" << value << "' in file." << endl;
                cerr << "\033[0m";
            }
            catch (const out_of_range& e) {
                cerr << "\033[31m";
                cerr << "Error: Value '" << value << "' is out of range." << endl;
                cerr << "\033[0m";
            }
        }

        if (!row.empty()) {
            data.push_back(row);
        }
    }

    file.close();
    return data;
}

vector<vector<vector<double>>> Utils::separateInputsOutputs(vector<vector<double>>& inputs, int index) {
    vector<vector<double>> finalInputs(inputs.size(), vector<double>(inputs[0].size() - 1));
    vector<vector<double>> outputs(inputs.size(), vector<double>(1));

    for (size_t i = 0; i < inputs.size(); ++i) {
        outputs[i][0] = inputs[i][index];
        for (size_t j = 0; j < finalInputs[0].size(); ++j) {
            finalInputs[i][j] = inputs[i][j];
        }

    }

    vector<vector<vector<double>>> result = { finalInputs, outputs };
    return result;
}


vector<vector<double>> Utils::convertCategoricalToOneHot(vector<vector<double>>& output) {
    vector<double> flattenOutput(output.size());

    for (size_t i = 0; i < output.size(); ++i) {
        flattenOutput[i] = output[i][0];
    }
    int maxIndex = *max_element(flattenOutput.begin(), flattenOutput.end());

    vector<vector<double>> finalOutput(output.size(), vector<double>(maxIndex + 1, 0));
    for (size_t i = 0; i < output.size(); ++i) {
        finalOutput[i][flattenOutput[i]] = 1;
    }
    return finalOutput;
}