import numpy as np

# Creating a variable that contains the training data for email filter
training_spam = np.loadtxt(open("venv/data/training_spam.csv"), delimiter=",")


# input = data set with binary response variable (0s and 1s) in the left-most column
# output = numpy array containing the logs of the class priors for the spam & ham classes
def estimate_log_class_priors(data):
    # Calculating the number of spam and ham emails
    spam_count = 0
    for i in range(len(data)):
        if data[i][0] == 1:
            spam_count += 1
    ham_count = len(data) - spam_count

    # Calculating the probabilities of spam and ham emails
    prob_of_spam = (spam_count / len(data))
    prob_of_ham = 1 - prob_of_spam

    log_class_priors = np.array([np.log(prob_of_ham), np.log(prob_of_spam)])

    return log_class_priors


log_class_priors = estimate_log_class_priors(training_spam)


# input =  data set with binary response variable (0s and 1s) in the left-most column
# output = 2D numpy-array containing logs of the class-conditional likelihoods for all words in the spam & ham classes
def estimate_log_class_conditional_likelihoods(data, alpha=1.0):
    no_of_columns = len(data[0])
    theta = np.array([[], []])  # 2d array to return
    total_spamw = 0
    total_hamw = 0

    for i in range(no_of_columns - 1):  # this loops along the columns
        spamw_count = 0
        hamw_count = 0
        for j in range(len(data)):  # this loops along the rows
            if data[j][0] == 1:  # checks for for features present in spam email
                if data[j][i + 1] == 1:
                    spamw_count += 1
                    total_spamw += 1
            else:  # checks for features present in ham email
                if data[j][i + 1] == 1:
                    hamw_count += 1
                    total_hamw += 1
        theta = np.append(theta, [[(spamw_count + alpha)], [(hamw_count + alpha)]], axis=1)

    for i in range(len(theta[0])):
        theta[0][i] = np.log(((theta[0][i]) / (total_spamw + ((no_of_columns - 1) * alpha))))
        theta[1][i] = np.log(((theta[1][i]) / (total_hamw + ((no_of_columns - 1) * alpha))))

    return theta


log_class_conditional_likelihoods = estimate_log_class_conditional_likelihoods(training_spam, alpha=1.0)


# input = a data set of new emails
# output = numpy array containing prediction of whether emails are spam(1) or ham(0)
def predict(new_data, log_class_priors, log_class_conditional_likelihoods):
    class_predictions = np.array([])

    for i in range(len(new_data)):
        sum_of_spamWordLogs = log_class_priors[1]
        sum_of_hamWordLogs = log_class_priors[0]
        for j in range(len(log_class_conditional_likelihoods[0])):
            sum_of_spamWordLogs += (new_data[i][j] * log_class_conditional_likelihoods[0][j])
            sum_of_hamWordLogs += (new_data[i][j] * log_class_conditional_likelihoods[1][j])
        if (sum_of_spamWordLogs >= sum_of_hamWordLogs):
            class_predictions = np.append(class_predictions, 1)
        else:
            class_predictions = np.append(class_predictions, 0)
    return class_predictions


# determine accuracy based on the proportion of true predictions made by the classifier.
def accuracy(y_predictions, y_true):
    count = 0
    for i in range(len(y_true)):
        if y_true[i] == y_predictions[i]:
            count += 1
    acc = count / len(y_true)
    return acc


# Creating a variable that contains the training data for email filter
testing_spam = np.loadtxt(open("venv/data/testing_spam.csv"), delimiter=",")
class_predictions = predict(testing_spam[:, 1:], log_class_priors, log_class_conditional_likelihoods)
true_classes = testing_spam[:, 0]
testing_set_accuracy = accuracy(class_predictions, true_classes)

# The predictions made on the testing data and the accuracy of the models predictions
print(class_predictions)
print("\n")
print(testing_set_accuracy)
