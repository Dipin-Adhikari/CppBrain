
# CppBrain

CppBrain is a lightweight C++ library for building and training neural networks, powered by a custom-built automatic differentiation system. Designed for simplicity, CppBrain is ideal for machine learning enthusiasts who prefer a C++-based solution.

## Key Features

- Custom Autodiff: A lightweight and efficient automatic differentiation engine built from scratch for seamless backpropagation.
- Artificial Neural Networks (ANN):
    - Easy-to-define architectures with support for Dense layers.
    - Activation functions like ReLU, Sigmoid, Tanh, and Softmax.
    - Optimizers like Gradient-Descent, Momentum, and Adam.
    - Loss functions such as mean-squared-error, categorical cross-entropy, and binary-cross-entropy.
- Utilities for Data Preprocessing:
    - Read CSV datasets effortlessly.
    - Train-test-validation splitting with random seeding.
    - One-hot encoding for categorical outputs.
- Extensible Design: Built with a modular architecture to accommodate future extensions.




## Installation

If you are using **Visual Studio Code**, follow these instructions to set up the configuration:

Refer to the official guide for setting up C++ in Visual Studio Code:  
[Visual Studio Code C++ Setup Guide](https://code.visualstudio.com/docs/languages/cpp)

### Steps to Get Started:

1. **Clone the Repository**  
   Run the following commands to clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/Dipin-Adhikari/CppBrain.git
   cd CppBrain
   ```

2. **Include Source Files**  
   Ensure all the source files are included in the args section of the tasks.json file.

3. **Run the main.cpp**


## Example Usage of CppBrain: Iris Dataset Classification

This example demonstrates how to use the **CppBrain** library to classify data from the Iris dataset. The process involves data preprocessing, building a neural network, training it, and evaluating its performance.

### Code Explanation

```cpp
#include <iostream>
#include <vector>
#include <string>
#include "includes/ann/Dense.h"
#include "includes/ann/NeuralNetwork.h"
#include "includes/ann/Utils.h"

using namespace std;

int main() {
    string filename = "iris.csv";
    vector<vector<double>> inputs = Utils::readCSV(filename);

    auto output = Utils::trainTestSplit(inputs, 0.7, 42);
    vector<vector<double>> trainingInputs = output[0];
    vector<vector<double>> testingInputs = output[1];

    output = Utils::trainTestSplit(testingInputs, 0.5);
    testingInputs = output[0];
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
    nn.fit(trainingInputs, trainingOutputs, 50, validationInputs, validationOutputs, 16);

    auto metrics = nn.evaluate(testingInputs, testingOutputs);
    cout << "Accuracy: " << metrics[0] << " Loss: " << metrics[1] << endl;

    return 0;
}
```
## Step-by-Step Explanation

### Reading the Dataset
The Iris dataset is loaded using the `Utils::readCSV` method, which reads the `iris.csv` file into a vector of vectors for processing.

### Splitting the Dataset
The dataset is split into training (70%), testing (15%), and validation (15%) sets using the `Utils::trainTestSplit` function. The random seed is set to `42` for reproducibility.

### Separating Inputs and Outputs
The `Utils::separateInputsOutputs` function splits each dataset (training, testing, and validation) into:
- **Inputs**: Features (first 4 columns of the dataset).
- **Outputs**: Labels (last column of the dataset).

### One-Hot Encoding Outputs
The categorical labels (e.g., class indices) are converted into one-hot encoded vectors using `Utils::convertCategoricalToOneHot`. This is required for the neural network to handle multi-class classification.

### Building the Neural Network
A neural network is constructed with three layers:
- **Input Layer**: 4 input neurons with ReLU activation.
- **Hidden Layer**: 10 neurons with ReLU activation.
- **Output Layer**: 3 neurons with softmax activation for multi-class classification.

### Compiling the Neural Network
The network is compiled with:
- **Loss function**: `categorical-cross-entropy` for multi-class classification.
- **Optimizer**: `adam` for efficient training.
- **Learning rate**: `0.02`.

### Training the Neural Network
The `fit` method trains the model on the training data for 50 epochs with a batch size of 16. Validation data is used to monitor performance during training.

### Evaluating the Neural Network
The `evaluate` method calculates:
- **Accuracy**: Percentage of correct predictions on the test dataset.
- **Loss**: Quantifies the error between predictions and true labels.

### Output Results
The accuracy and loss values are printed to the console, showcasing the model's performance.

## Future Plans

CppBrain is a growing project! Here are some planned features for upcoming releases:

- Convolutional Neural Networks (CNNs).
- Support for Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks.
- Advanced optimizers and custom loss functions.
- Visualization tools for training metrics.
- Export and deployment options for trained models.
## Contributions

Contributions are welcome! If you have suggestions, bug fixes, or feature requests, feel free to open an issue or submit a pull request.

