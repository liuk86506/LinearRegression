import pandas as pd
import numpy as np
import MultipleLinearRegression as mlr

# Example using multiple linear regression
# Getting data
data = pd.read_csv("multiple_linear_regression_dataset.csv")
data = data.to_numpy()

# Adjusting data
inputs = []
outputs = []
for i in range(data.shape[0]):
    if data[i][2] == 'Yes':
        data[i][2] = 1.0
    else:
        data[i][2] = 0.0

    inputs.append(data[i][:5])
    outputs.append(data[i][5:])

# Converting data
inputs = np.array(inputs)
outputs = np.array(outputs)

# Preparing data
train_inputs = inputs[0 : int(inputs.shape[0] * 0.75)]
train_outputs = outputs[0 : int(outputs.shape[0] * 0.75)]

test_inputs = inputs[int(inputs.shape[0] * 0.75) :]
test_outputs = outputs[int(outputs.shape[0] * 0.75) :]

# Testing multiple linear regression model
model = mlr.MultipleLinearRegression(train_inputs, train_outputs)
b, weights, iterations, loss = model.batch_grad_desc(lr = 0.0001, max_iters = 7500, show_iterations = False)
print(model.error(test_inputs, test_outputs))