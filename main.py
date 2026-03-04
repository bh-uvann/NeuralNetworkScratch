import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data Preparation
train = pd.read_csv('train.csv')
train_ans = train['label'].to_numpy()
train = train.drop('label', axis='columns').to_numpy()
train = train / 255.0  # Normalize


test = pd.read_csv('test.csv').to_numpy()
test = test / 255.0  # Normalize

class DigitRecognizer():
    def __init__(self):
        self.lr = 0.1 # learning Rate
        self.input = None
        # Initialisation of random weights
        self.layer1_w = np.random.randn(784, 50) * 0.01
        self.layer1_b = np.zeros((1, 50))
        self.layer2_w = np.random.randn(50, 10) * 0.01
        self.layer2_b = np.zeros((1, 10))
        
        self.z1 = None
        self.layer1_out = None
        self.layer2_out = None

    def forward(self, x):
        self.input = x.reshape(1, -1) # Ensure 2D shape (1, 784)
        
        # Layer 1: Linear -> ReLU
        self.z1 = np.dot(self.input, self.layer1_w) + self.layer1_b
        self.layer1_out = np.maximum(0, self.z1) 
        
        # Layer 2: Linear -> Softmax
        z2 = np.dot(self.layer1_out, self.layer2_w) + self.layer2_b
        # Numerical stability for softmax
        exp_z = np.exp(z2 - np.max(z2)) 
        self.layer2_out = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return self.layer2_out

    def backward(self, y_true):
        # 1. Error at Output (Softmax + Cross-Entropy derivative)
        # dz2 = (prediction - target)
        dz2 = self.layer2_out - y_true 
        
        dw2 = np.dot(self.layer1_out.T, dz2)
        db2 = dz2
        
        # 2. Error at Hidden Layer (Backprop through ReLU)
        # dz1 = (dz2 * W2.T) * ReLU_derivative
        dz1 = np.dot(dz2, self.layer2_w.T) * (self.z1 > 0)
        
        dw1 = np.dot(self.input.T, dz1)
        db1 = dz1
        
        # 3. Parameter Updates
        self.layer1_w -= self.lr * dw1
        self.layer1_b -= self.lr * db1
        self.layer2_w -= self.lr * dw2
        self.layer2_b -= self.lr * db2

# Hyperparameters and Training
train_x = train[:1000] # Increased sample size for better observation
train_y = train_ans[:1000]
model = DigitRecognizer()

for i in range(len(train_x)):
    # One hot encoding
    y_true = np.zeros((1, 10))
    y_true[0, train_y[i]] = 1
    
    # Forward pass
    prediction = model.forward(train_x[i])
    
    # Backward pass
    model.backward(y_true)
    
    if i % 100 == 0:
        loss = -np.sum(y_true * np.log(prediction + 1e-8))
        # print(f"Iteration {i}, Loss: {loss:.4f}")

ImageId = []
Label = []

for x in range(28000):
    result = model.forward(test[x])
    result = int(np.argmax(result))
    ImageId.append(x+1)
    Label.append(result)

df = pd.DataFrame({"ImageId":ImageId,"Label":Label})
df.to_csv('Submission.csv',index=False)