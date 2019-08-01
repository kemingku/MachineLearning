import numpy as np
import tensorflow as tf
from tqdm import tqdm_notebook
import pandas as pd
from keras.utils import np_utils
import matplotlib.pyplot as plt


def fizzbuzz(n):
    # Enter input n, when n can be divided by 15, output 'FizzBuzz'
    #                when n can be divided by 3, output 'Fizz'
    #                when n can be divided by 5, output 'Buzz'
    #                output 'Other' in any other situation

    if n % 3 == 0 and n % 5 == 0:
        return 'FizzBuzz'
    elif n % 3 == 0:
        return 'Fizz'
    elif n % 5 == 0:
        return 'Buzz'
    else:
        return 'Other'


def createInputCSV(start, end, filename):
    # Why list in Python?
    # We need 2 list of value: inputData and outputData. In this exercise, we use List instead of Tuples because
    # List has more builtin function, and it makes it easier to use them in later coding. Moreover, List has more
    # variable length and mutable nature.

    inputData = []
    outputData = []

    # Why do we need training Data?
    # Since we are using supervisor in MachineLearning, we need to give the training set from 0 to 100 as the initial
    # training data.

    for i in range(start, end):
        inputData.append(i)
        outputData.append(fizzbuzz(i))

    # Why Dataframe?
    # Because we need to organize the our data into two-dimensional array-like structure in which each column
    # contains values of one variable and each row contains one set of values from each column.
    dataset = {}
    dataset["input"] = inputData
    dataset["label"] = outputData

    # Writing to csv
    pd.DataFrame(dataset).to_csv(filename)

    print(filename, "Created!")


def processData(dataset):
    # Why do we have to process?
    # It gets the data value and label value, and call the encodeData to process them into binary data.
    data = dataset['input'].values
    labels = dataset['label'].values

    processedData = encodeData(data)
    processedLabel = encodeLabel(labels)

    return processedData, processedLabel


def encodeData(data):
    processedData = []

    for dataInstance in data:
        # Why do we have number 10?
        # Because we want to append the data to a list 10 times.
        processedData.append([dataInstance >> d & 1 for d in range(10)])

    return np.array(processedData)

def encodeLabel(labels):
    processedLabel = []

    for labelInstance in labels:
        if (labelInstance == "FizzBuzz"):
            # Fizzbuzz
            processedLabel.append([3])
        elif (labelInstance == "Fizz"):
            # Fizz
            processedLabel.append([1])
        elif (labelInstance == "Buzz"):
            # Buzz
            processedLabel.append([2])
        else:
            # Other
            processedLabel.append([0])

    return np_utils.to_categorical(np.array(processedLabel), 4)

# Create datafiles
createInputCSV(101,1001,'training.csv')
createInputCSV(1,101,'testing.csv')

# Read Dataset
trainingData = pd.read_csv('training.csv')
testingData  = pd.read_csv('testing.csv')

# Process Dataset
processedTrainingData, processedTrainingLabel = processData(trainingData)
processedTestingData, processedTestingLabel   = processData(testingData)

# Defining Placeholder
inputTensor  = tf.placeholder(tf.float32, [None, 10])
outputTensor = tf.placeholder(tf.float32, [None, 4])


NUM_HIDDEN_NEURONS_LAYER_1 = 100
LEARNING_RATE = 0.05

# Initializing the weights to Normal Distribution
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.01))

# Initializing the input to hidden layer weights
input_hidden_weights  = init_weights([10, NUM_HIDDEN_NEURONS_LAYER_1])
# Initializing the hidden to output layer weights
hidden_output_weights = init_weights([NUM_HIDDEN_NEURONS_LAYER_1, 4])

# Computing values at the hidden layer
hidden_layer = tf.nn.relu(tf.matmul(inputTensor, input_hidden_weights))
# Computing values at the output layer
output_layer = tf.matmul(hidden_layer, hidden_output_weights)

# Defining Error Function
error_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=outputTensor))

# Defining Learning Algorithm and Training Parameters
training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(error_function)

# Prediction Function
prediction = tf.argmax(output_layer, 1)

NUM_OF_EPOCHS = 5000
BATCH_SIZE = 128

training_accuracy = []

with tf.Session() as sess:
    # Set Global Variables ?
    # Because tf.global_variables_initializer() is a shortcut to initialize all global variables. And we are using this
    # function to initialize used variables.
    tf.global_variables_initializer().run()

    for epoch in tqdm_notebook(range(NUM_OF_EPOCHS)):

        # Shuffle the Training Dataset at each epoch
        p = np.random.permutation(range(len(processedTrainingData)))
        processedTrainingData = processedTrainingData[p]
        processedTrainingLabel = processedTrainingLabel[p]

        # Start batch training
        for start in range(0, len(processedTrainingData), BATCH_SIZE):
            end = start + BATCH_SIZE
            sess.run(training, feed_dict={inputTensor: processedTrainingData[start:end],
                                          outputTensor: processedTrainingLabel[start:end]})
        # Training accuracy for an epoch
        training_accuracy.append(np.mean(np.argmax(processedTrainingLabel, axis=1) ==
                                         sess.run(prediction, feed_dict={inputTensor: processedTrainingData,
                                                                         outputTensor: processedTrainingLabel})))
    # Testing
    predictedTestLabel = sess.run(prediction, feed_dict={inputTensor: processedTestingData})

df = pd.DataFrame()
df['acc'] = training_accuracy
df.plot(grid=True)
plt.show()

def decodeLabel(encodedLabel):
    if encodedLabel == 0:
        return "Other"
    elif encodedLabel == 1:
        return "Fizz"
    elif encodedLabel == 2:
        return "Buzz"
    elif encodedLabel == 3:
        return "FizzBuzz"


wrong = 0
right = 0

predictedTestLabelList = []

for i, j in zip(processedTestingLabel, predictedTestLabel):
    predictedTestLabelList.append(decodeLabel(j))

    if np.argmax(i) == j:
        right = right + 1
    else:
        wrong = wrong + 1

print("Errors: " + str(wrong), " Correct :" + str(right))

print("Testing Accuracy: " + str(right / (right + wrong) * 100))

# Please input your UBID and personNumber
testDataInput = testingData['input'].tolist()
testDataLabel = testingData['label'].tolist()

testDataInput.insert(0, "UBID")
testDataLabel.insert(0, "kemingku")

testDataInput.insert(1, "personNumber")
testDataLabel.insert(1, "50161776")

predictedTestLabelList.insert(0, "")
predictedTestLabelList.insert(1, "")

output = {}
output["input"] = testDataInput
output["label"] = testDataLabel

output["predicted_label"] = predictedTestLabelList

opdf = pd.DataFrame(output)
opdf.to_csv('output.csv')

