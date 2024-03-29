double* layerWeights[] = {new double[98]{-0.533740,0.067835,-0.327872,0.330057,0.090073,-0.024202,8846.032831,0.423302,0.345131,0.263632,-0.348393,0.383727,0.225035,0.014469,-0.209620,-0.518511,-0.436809,-0.144907,-0.377098,-0.357369,19615.066083,-0.058057,-0.407217,-0.529531,-0.524996,-0.130552,0.033849,0.076099,0.108206,0.114445,-0.356811,0.174303,-0.053289,-0.160354,61638.448725,0.115120,0.302880,0.323500,0.021256,-0.211724,0.401932,0.242327,0.486028,0.454660,0.042071,-0.382356,-0.042792,-0.290459,-71927.533356,-0.310449,0.298965,0.367381,0.531097,0.534196,0.119198,-0.114989,-0.250055,-0.216738,0.363629,-0.509140,-0.132853,-0.435998,13069.536482,-0.474426,-0.525126,0.447706,-0.239587,-0.242784,0.093978,0.204383,0.363153,0.242665,-0.016101,-0.314984,0.264002,-0.022260,28311.285638,0.480168,0.261315,-0.418767,0.105887,-0.122689,0.251234,0.116490,0.077827,-0.148145,-0.372503,-0.293876,-0.079511,0.325472,-41000.632798,0.523821,0.268917,-0.165102,-0.353874,0.168169,-0.008662,-0.466596},new double[98]{9482.685376,0.005215,-0.376748,0.480774,-0.383101,0.419090,0.214857,-0.208090,-0.135927,-0.566595,0.443653,0.046150,-0.457435,0.375640,129634.431830,0.027546,-0.340946,-0.362513,0.388355,0.038282,-0.359347,0.060805,-0.099052,-0.314488,-0.168830,-0.202983,0.318370,0.230824,103459.861965,-0.274185,-0.372661,-0.138704,0.625566,-0.473631,0.014164,0.079457,0.588355,0.254885,0.441322,1.081458,-0.555560,-0.232660,-155741.937651,0.331704,0.238987,-0.245216,-0.549281,-0.331110,0.518935,-0.216178,0.265286,0.077066,-0.326702,0.279944,0.373398,-0.076202,-6878.190724,0.417103,-0.505159,0.528780,0.077600,-0.480527,0.033490,-0.327056,0.366910,0.135609,0.168495,-0.323010,0.366280,-0.401023,70938.087515,0.259912,-0.198772,0.471522,-0.228690,-0.174988,-0.384575,0.249178,0.357458,0.222275,0.107159,0.264284,-0.264799,-0.381573,-15953.934156,-0.469304,0.327382,0.376973,-0.309405,-0.410936,0.056883,-0.519286,-0.412894,-0.048629,0.269635,0.199000,0.046443,-0.455536},new double[14]{0.235677,-0.038554,0.465839,-0.254850,-0.043097,-1.337342,-0.010720,0.289033,0.119521,-0.554905,0.006753,0.756145,0.319676,0.693699}}; 
double* layerBiases[] = {new double[14]{-0.000054,-0.000003,-0.000045,-0.000088,-0.000002,0.000185,0.000014,-0.000001,-0.000045,-0.000013,-0.000002,0.000366,0.000006,-0.000002},new double[7]{-0.003309,0.000589,0.007521,0.007867,-0.003844,-0.017729,0.000395},new double[2]{0.491339,-0.343778}}; 
//The weights and biases are above (comment must be below due to how the update code function works)

//Standard libraries only
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <array>

using namespace std;

//Toggle to switch between training and testing
bool training = true;
string file = training ? "train.csv" : "test.csv";

//Variables to control how the training is performed
int batchSize = 100, trainingIterations = 10;
double learningRate = 0.01;

//The amount of nodes in each layer (modifiying the number of layers does result in requiring a couple modifications in the code)
int layerNodeCounts[] = {7, 14, 7, 2};
int layerCount = sizeof(layerNodeCounts)/sizeof(*layerNodeCounts);

//Updates the code file's hard coded weights and biases
void UpdateCode(string fileName) {
    string weightsLine = "double* layerWeights[] = {", biasesLine = "double* layerBiases[] = {";
    for (int layer = 1; layer < layerCount; layer++) { //Translates the variable values to lines of c++ code
        weightsLine += "new double[" + to_string(layerNodeCounts[layer - 1] * layerNodeCounts[layer]) + "]{";
        biasesLine += "new double[" + to_string(layerNodeCounts[layer]) + "]{";

        weightsLine += to_string(layerWeights[layer - 1][0]);
        for (int weight = 1; weight < layerNodeCounts[layer - 1] * layerNodeCounts[layer]; weight++) {
            weightsLine += "," + to_string(layerWeights[layer - 1][weight]);
        }

        biasesLine += to_string(layerBiases[layer - 1][0]);
        for (int bias = 1; bias < layerNodeCounts[layer]; bias++) {
            biasesLine += "," + to_string(layerBiases[layer - 1][bias]);
        }

        weightsLine += "},";
        biasesLine += "},";
    }
    weightsLine.pop_back(); //Removes the final comma meant to seperate the layers which is not wanted
    biasesLine.pop_back();

    weightsLine += "}; \n";
    biasesLine += "}; \n";

    fstream codeFile(fileName);
    string code = weightsLine + biasesLine; //Stores the entire code including the new lines for weights and biases
    string line = ""; //Temporarily stores a given line before being switched for the next

    getline(codeFile, line); //Remove the existing lines defining the weights and biases
    getline(codeFile, line);

    while(getline(codeFile, line)) { //Read all the existing code to a string
        code += line + "\n";
    }

    codeFile.close();
    codeFile.open(fileName, fstream::out | fstream::trunc); //Clear the code file and open it
    codeFile << code; //Add back the stored code with the new weights and biases
    codeFile.close();
}

void Initiate() { //Initiates all weights and biases using the Normalized Xavier Weight Initialization
    for (int layer = 0; layer < layerCount - 1; layer++) {
        double range = sqrt(6 / (layerNodeCounts[layer] + layerNodeCounts[layer + 1])); //Range of the random numbers
        
        for (int weight = 0; weight < layerNodeCounts[layer] * layerNodeCounts[layer + 1]; weight++) {
            double randomNumber = (double)rand() / RAND_MAX; //Gets a random number from 0 to 1
            layerWeights[layer][weight] = randomNumber * range * 2 - range; //Constrains the random number into the given range
        }

        for (int bias = 0; bias < layerNodeCounts[layer + 1]; bias++) {
           layerBiases[layer][bias] = 0; //For biases, they don't suffer the exploding or vanishing gradient problem and can be initiated at 0
        }
    }
    return;
}

//Sigmoid activation function
double ActivationFunction(double weightedInput) { 
    return 1.0 / (1.0 + exp(-weightedInput));
}

//The derivative of the Sigmoid activation function
double ActivationFunctionDerivative(double weightedInput) { 
    double output = ActivationFunction(weightedInput);
    return output * (1 - output);
}

//The cost function used to evaluate how well the network performed
double Cost(vector<double> outputs, vector<double> expectedOutputs) {
    double cost = 0;
    for (int i = 0; i < outputs.size(); i++)
        cost += (outputs[i] - expectedOutputs[i])*(outputs[i] - expectedOutputs[i]);

    return cost;
}

//The derivative of the cost function for use in the back propogation (training)
double CostDerivative(double output, double expectedOutput) {
    return 2 * (output - expectedOutput);
}

//Helper function to retrieve a weight value
double Weight(int layer, int inputNode, int outputNode) {
    return layerWeights[layer - 1][inputNode + outputNode * layerNodeCounts[layer]];
}

//Helper function to retrieve a bias value
double Bias(int layer, int outputNode) {
    return layerBiases[layer - 1][outputNode];
}

//Forward propogates through the neural network to estimate the output
vector<double> Evalute(vector<int> data, vector<vector<double>> &weightedInputs) { //Evaluates if the given passenger would've survived or not
    vector<double> inputs;
    for (int i = 0; i < data.size(); i++) {
        inputs.push_back((double)data[i]);
    }
    
    //Input nodes are where the input is coming from and output nodes is the next layer
    for (int layer = 1; layer < layerCount; layer++) {
        vector<double> newInputs;
        
        for (int outputNode = 0; outputNode < layerNodeCounts[layer]; outputNode++) {
            double input = Bias(layer, outputNode);
            for (int inputNode = 0; inputNode < layerNodeCounts[layer - 1]; inputNode++) {
                double weightedInput = inputs[inputNode] * Weight(layer, inputNode, outputNode);
                input += weightedInput;
            }
            newInputs.push_back(ActivationFunction(input));
            weightedInputs[layer - 1][outputNode] = input;
        }
        inputs = newInputs;
    }
    return inputs;
}

//Finds the consequentual partial derivatives for any weights and biases that aren't in the last layer
double NodeGradient(int layer, int inputNode, int outputNode, vector<vector<double>> weightedInputs, vector<double> expectedOutputs) {
    double za = Weight(layer, inputNode, outputNode);
    double az = ActivationFunctionDerivative(weightedInputs[layer - 1][outputNode]);
    
    if (layer == layerCount - 1) {
        return za * az * CostDerivative(ActivationFunction(weightedInputs[layer - 1][outputNode]), expectedOutputs[outputNode]);
    }

    else {
        double combinedSum = 0;
        for (int node = 0; node < layerNodeCounts[layer + 1]; node++)
            combinedSum += NodeGradient(layer + 1, outputNode, node, weightedInputs, expectedOutputs);
        
        return za * az * combinedSum;
    }
}

//The main function which reads all the data, formats it and then either trains or tests the network based on the just read data
int main() {
    //Initiate(); //Used only for first run of the network to initialize weights and biases before training

    int numCorrectAnswers = 0, numPassengers = 0;
    vector<array<vector<int>, 2>> allData;

    ifstream dataFile; //Create the file object and open the given file into it
    dataFile.open(file); //The given file will be hard coded to be the training or the testing data

    string line;
    getline(dataFile, line);
    while (getline(dataFile, line)) { //Read all the lines of data and reformat them for use
        numPassengers++;
        vector<int> data;
        vector<int> nonInputData;
        int dataCount = 0;
        string splitString = "";
        bool skip = false;
        for (int i = 0; i < line.length(); i++) {
            if (line[i] == ',' || line[i] == ' ') {
                if (skip) skip = false; //skip the double commas

                else if (dataCount++ < (training ? 2 : 1)) //Seperatly record the non input data
                    nonInputData.push_back(stoi(splitString)); 

                else if (splitString == "male") { //Reformate the gender
                    data.push_back(0);
                    skip = true;
                }
                
                else if (splitString == "female") {
                    data.push_back(1);
                    skip = true;
                }
                
                else if (isdigit(splitString[0])) //Record all other data as normal
                    data.push_back(stoi(splitString));
                splitString = "";
            }
            else splitString += line[i]; //If it isn't a comma, add the character to the string representing the given data point
        }
        if (splitString == "S") //Reformat the embarking location
            data.push_back(0);
        else if (splitString == "C")
            data.push_back(1);
        else if (splitString == "Q")
            data.push_back(2);

        allData.push_back(array<vector<int>, 2>{data, nonInputData});
    }
    dataFile.close(); //Close the file

    if (training) {
        auto rd = random_device {}; //Sets the random generation for later use in shuffling
        auto rng = default_random_engine { rd() };

        for (int i = 0; i < trainingIterations; i++) { //Repeats the training a given amount of times, saving the weights and biases each time to prevent data loss
            cout << "started training round " << i + 1<< endl;

            //After each training iteration, shuffle the data so the batches are random
            shuffle(begin(allData), end(allData), rng);

            //Data is batched for increased training speed 
            for (int batch = 0; batch < allData.size() / batchSize; batch++) {               
                //Allows for differnt node counts among the layers but will need to be modified if the number of hidden layers changes
                //Stores the sums of the gradients for each weight and biases according to each data sample
                double* layerWeightGradients[] = {new double[layerNodeCounts[0] * layerNodeCounts[1]], new double[layerNodeCounts[1] * layerNodeCounts[2]], new double[layerNodeCounts[2] * layerNodeCounts[3]]};
                double* layerBiasGradients[] = {new double[layerNodeCounts[1]], new double[layerNodeCounts[2]], new double[layerNodeCounts[3]]};
                
                double totalCurrentCost = 0;

                //Finds the gradients of each weight and bias and averages it out across the different samples in the batch
                for (int sample = 0; sample < batchSize; sample++) {
                    array<vector<int>,2> combinedData = allData[batch * batchSize + sample];
                    vector<vector<double>> weightedInputs;
                    weightedInputs.resize(layerCount - 1);
                    for (int layer = 0; layer < layerCount - 1; layer++)
                        weightedInputs[layer].resize(layerNodeCounts[layer + 1]);

                    vector<double> outputs = Evalute(combinedData[0], weightedInputs);
                    vector<double> expectedOutputs (2, 0);
                    expectedOutputs[combinedData[1][1]] = 1;

                    totalCurrentCost += Cost(outputs, expectedOutputs);

                    //Input nodes are where the input is coming from and output nodes is the next layer
                    //Find the partial derivates of each weight and bias to use for gradient descent
                    for (int layer = 1; layer < layerCount; layer++) {                       
                        for (int outputNode = 0; outputNode < layerNodeCounts[layer]; outputNode++) {
                            double az = ActivationFunctionDerivative(weightedInputs[layer - 1][outputNode]);
                            if (layer == 3)
                                layerBiasGradients[layer - 1][outputNode] += az * CostDerivative(ActivationFunction(weightedInputs[layer - 1][outputNode]), expectedOutputs[outputNode]);

                            else {
                                double combinedSum = 0;
                                for (int node = 0; node < layerNodeCounts[layer + 1]; node++)
                                    combinedSum += NodeGradient(layer + 1, outputNode, node, weightedInputs, expectedOutputs);
                                
                                layerBiasGradients[layer - 1][outputNode] += az * az * combinedSum;
                            }

                            for (int inputNode = 0; inputNode < layerNodeCounts[layer - 1]; inputNode++) {
                                double zw = layer == 1 ? combinedData[0][inputNode] : ActivationFunction(weightedInputs[layer - 2][inputNode]);
                                
                                if (layer == 3) 
                                    layerWeightGradients[layer - 1][inputNode + outputNode * layerNodeCounts[layer]] += zw * az * CostDerivative(ActivationFunction(weightedInputs[layer - 1][outputNode]), expectedOutputs[outputNode]);
                                
                                else {
                                    double combinedSum = 0;
                                    for (int node = 0; node < layerNodeCounts[layer + 1]; node++)
                                        combinedSum += NodeGradient(layer + 1, outputNode, node, weightedInputs, expectedOutputs);
                                    
                                    layerWeightGradients[layer - 1][inputNode + outputNode * layerNodeCounts[layer]] += zw * az * combinedSum;
                                }
                            }
                        }
                    }
                }

                std::cout << totalCurrentCost / batchSize << endl;

                //Update all weights using gradient descent based on the previously found weight and bias gradients
                for (int layer = 1; layer < layerCount; layer++) {
                    for (int outputNode = 0; outputNode < layerNodeCounts[layer]; outputNode++) {
                        layerBiases[layer - 1][outputNode] -= layerBiasGradients[layer - 1][outputNode] * learningRate;

                        for (int inputNode = 0; inputNode < layerNodeCounts[layer - 1]; inputNode++) {
                            layerWeights[layer - 1][inputNode + outputNode * layerNodeCounts[layer]] -= layerWeightGradients[layer - 1][inputNode + outputNode * layerNodeCounts[layer]] * learningRate;              
                        }
                    }
                }
                /*
                for (int i = 0; i < layerCount - 1; i++) {
                    delete[] layerWeightGradients[i];
                    delete[] layerBiasGradients[i];
                }
                */
            }
        }
    }

    else { //If the network is being tested, then find the average cost across all testing data
        double totalCost = 0;
        cout << allData.size() << endl;
        for (int i = 0; i < allData.size(); i++) {
            array<vector<int>,2> combinedData = allData[i];
            vector<vector<double>> weightedInputs;
            weightedInputs.resize(layerCount - 1);
            for (int layer = 0; layer < layerCount - 1; layer++)
                weightedInputs[layer].resize(layerNodeCounts[layer + 1]);

            vector<double> outputs = Evalute(combinedData[0], weightedInputs);
            vector<double> expectedOutputs (2, 0);
            expectedOutputs[combinedData[1][1]] = 1;

            totalCost += Cost(outputs, expectedOutputs);
        }
        cout << totalCost << endl;
    }

    UpdateCode("titanic.cpp"); //Update the weights and biases in the code

    return 0;
}
