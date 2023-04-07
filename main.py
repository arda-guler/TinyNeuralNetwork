import random
import math

class NeuralNet:
    def __init__(self, N_in, N_mid, N_out):

        # set layer neuron numbers
        self.N_in = N_in
        self.N_mid = N_mid
        self.N_out = N_out

        # initialize weights
        self.weights_im = []
        for idx1 in range(N_in):
            self.weights_im.append([])
            for idx2 in range(N_mid):
                self.weights_im[idx1].append(random.uniform(-1, 1))
                
        self.weights_mo = []
        for idx1 in range(N_mid):
            self.weights_mo.append([])
            for idx2 in range(N_out):
                self.weights_mo[idx1].append(random.uniform(-1, 1))

        # initialize biases
        self.biases_in = []
        for idx1 in range(N_mid):
            self.biases_in.append(random.uniform(-1, 1))

        self.biases_mid = []
        for idx1 in range(N_out):
            self.biases_mid.append(random.uniform(-1, 1))

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def forward(self, inputs):
        layer_mid = [0] * self.N_mid
        
        for j in range(self.N_mid):
            csum = 0
            
            for i in range(self.N_in):
                csum += inputs[i] * self.weights_im[i][j]
                
            csum += self.biases_in[j]
            layer_mid[j] = self.sigmoid(csum)

        layer_out = [0] * self.N_out
        
        for j in range(self.N_out):
            csum = 0
            for i in range(self.N_mid):
                csum += layer_mid[i] * self.weights_mo[i][j]
                
            csum += self.biases_mid[j]
            layer_out[j] = self.sigmoid(csum)

        return layer_out

    def train(self, inputs, targets, delta_rate):
        layer_mid = [0] * self.N_mid

        for j in range(self.N_mid):
            csum = 0
            for i in range(self.N_in):
                csum += inputs[i] * self.weights_im[i][j]

            csum += self.biases_in[j]
            layer_mid[j] = self.sigmoid(csum)

        layer_out = [0] * self.N_out
        for j in range(self.N_out):
            csum = 0
            for i in range(self.N_mid):
                csum += layer_mid[i] * self.weights_mo[i][j]

            csum += self.biases_mid[j]
            layer_out[j] = self.sigmoid(csum)

        error_out = []
        for i in range(self.N_out):
            error_out.append(targets[i] - layer_out[i])

        error_mid = [0] * self.N_mid
        for i in range(self.N_mid):
            for j in range(self.N_out):
                error_mid[i] += error_out[j] * self.weights_mo[i][j]

            error_mid[i] *= layer_mid[i] * (1 - layer_mid[i])

        for i in range(self.N_in):
            for j in range(self.N_mid):
                self.weights_im[i][j] += delta_rate * error_mid[j] * inputs[i]

        for i in range(self.N_mid):
            for j in range(self.N_out):
                self.weights_mo[i][j] += delta_rate * error_out[j] * layer_mid[i]

        for i in range(self.N_mid):
            self.biases_in[i] += delta_rate * error_mid[i]

        for i in range(self.N_out):
            self.biases_mid[i] += delta_rate * error_out[i]

    def autotrain(self, inputset, targetset, rate):
        for idx in range(len(inputset)):
            self.train(inputset[idx], targetset[idx], rate)

    def predict(self, input):
        return self.forward(input)

# example case: check which input is larger
def generate_training_data(n=1000):
    inputs = []
    targets = []

    for i in range(n):
        a1 = random.uniform(0, 10)
        a2 = random.uniform(0, 10)
        inputs.append([a1, a2])

        if a1 > a2:
            targets.append([1])
        else:
            targets.append([0])

    return inputs, targets

# initialize AI
KSI = NeuralNet(2, 20, 1)

# try to predict with untrained network for sake of demonstration
print("\nUntrained predictions:")
print(KSI.predict([3, 5]), "Expected: 0")
print(KSI.predict([8, 5]), "Expected: 1")
print(KSI.predict([1, 10]), "Expected: 0")
print(KSI.predict([2, 0]), "Expected: 1")
print(KSI.predict([3, 3.5]), "Expected: 0")

# generate a dataset of 10000 cases
inputs, targets = generate_training_data(10000)

# go over generated dataset with delta rate 1
KSI.autotrain(inputs, targets, 1)

# try to predict with trained network
print("\nTrained predictions:")
print(KSI.predict([3, 5]), "Expected: 0")
print(KSI.predict([8, 5]), "Expected: 1")
print(KSI.predict([1, 10]), "Expected: 0")
print(KSI.predict([2, 0]), "Expected: 1")
print(KSI.predict([3, 3.5]), "Expected: 0")
