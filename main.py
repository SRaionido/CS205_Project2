import numpy as np
import random
import math
import csv

# input_file = 'CS205_small_Data__33.txt'

# Custom function to load custom data for CS205 Project 2: Breast Cancer Wisconsin (Diagnostic) Data Set
def load_data_custom(filename):
    data = []
    with open(filename, 'r') as f:
        for row in csv.reader(f):
            if not row:
                continue
            label = 1 if row[1] == 'M' else 0
            features = list(map(float, row[2:]))
            data.append([label] + features)
    return np.array(data)

def Feature_Search_Forward(data):
    # IN PROGRESS
    num_features = data.shape[1] - 1
    num_samples = data.shape[0]

    print ("\nThis dataset has", num_features, "features (not including the class attribute), with", num_samples, "instances.\n")

    set_with_all_features = []
    for i in range(num_features):
        set_with_all_features.append(i+1)

    print("Running nearest neighbor with all " + str(num_features) + " features, using leave-one-out evaluation, I get an accuracy of", round(Cross_Validate(data, set_with_all_features, 1)*100,1), "%\n")
    print("Beginning Search\n")

    current_feature_set = []
    best_total_acc = 0
    best_feature_set = []

    for i in range(num_features):
        # print("In level " + str(i+1) + " of tree")
        feature_to_add = -1
        best_acc_so_far = 0
        for k in range(num_features):
            if k+1 in current_feature_set:
                continue
            # print("- - - Considering feature", k)
            acc = Cross_Validate(data, current_feature_set, k+1)
            set_just_tested = current_feature_set.copy()
            set_just_tested.append(k+1)
            print ("     Using features", set_just_tested, "accuracy is", round(acc*100,1), "%")

            if acc > best_acc_so_far or feature_to_add == -1:
                best_acc_so_far = acc
                feature_to_add = k+1

        if best_acc_so_far > best_total_acc:
            best_total_acc = best_acc_so_far
            current_feature_set.append(feature_to_add)
            best_feature_set = current_feature_set.copy()
            print("\nFeature set", current_feature_set, "was best, accuracy is", round(best_acc_so_far*100,1), "%\n")
        else:
            current_feature_set.append(feature_to_add)
            print("\n(Warning, accuracy has decreased! Continuing search in case of local maxima)")
            print("Feature set", current_feature_set, "was best, accuracy is", round(best_acc_so_far*100,1), "%\n")

    print("\nFinished search, best feature set is", best_feature_set, "with accuracy", round(best_total_acc*100,1), "%")
    return

def Feature_Search_Backward(data):
    # IN PROGRESS
    num_features = data.shape[1] - 1
    num_samples = data.shape[0]

    print ("\nThis dataset has", num_features, "features (not including the class attribute), with", num_samples, "instances.\n")

    set_with_all_features = []
    for i in range(num_features):
        set_with_all_features.append(i+1)

    print("Running nearest neighbor with all " + str(num_features) + " features, using leave-one-out evaluation, I get an accuracy of", round(Cross_Validate(data, set_with_all_features, 1)*100,1), "%\n")
    print("Beginning Search\n")

    current_feature_set = set_with_all_features.copy()
    best_total_acc = Cross_Validate(data, current_feature_set, 1)
    best_feature_set = current_feature_set.copy()

    for i in range(num_features-1):
        # print("In level " + str(i+1) + " of tree")
        feature_to_remove = -1
        best_acc_so_far = 0
        for k in range(num_features):
            if k+1 not in current_feature_set:
                continue
            # print("- - - Considering feature", k)
            acc = Cross_Validate_Leave_One_Out(data, current_feature_set, k+1)
            set_just_tested = current_feature_set.copy()
            set_just_tested.remove(k+1)
            print ("     Using features", set_just_tested, "accuracy is", round(acc*100,1), "%")

            if acc > best_acc_so_far or feature_to_remove == -1:
                best_acc_so_far = acc
                feature_to_remove = k

        if best_acc_so_far > best_total_acc:
            best_total_acc = best_acc_so_far
            current_feature_set.remove(feature_to_remove+1)
            best_feature_set = current_feature_set.copy()
            print("\nFeature set", current_feature_set, "was best, accuracy is", round(best_acc_so_far*100,1), "%\n")
        else:
            current_feature_set.remove(feature_to_remove+1)
            print("\n(Warning, accuracy has decreased! Continuing search in case of local maxima)")
            print("Feature set", current_feature_set, "was best, accuracy is", round(best_acc_so_far*100,1), "%\n")

    print("\nFinished search, best feature set is", best_feature_set, "with accuracy", round(best_total_acc*100,1), "%")
    return

def Cross_Validate(data, current_set, feature_to_add):
    num_features = data.shape[1] - 1
    num_samples = data.shape[0]
    # To Be Added    
    # This function should return the accuracy of the model with the current feature set and the new feature added via leave-one-out cross-validation
    

    set_to_test = current_set.copy()
    set_to_test.append(feature_to_add)

    labels = data[:, 0]
    features = data[:, set_to_test]
    # print("Features to test are ", features)
    # print("Labels to test are ", labels)

    # Here you would implement the logic for leave-one-out cross-validation
    # Do leave one out and find nearest neighbor by euclidean distance

    accuracy = 0

    for i in range(num_samples):
        # Leave one out
        test_sample = features[i]
        train_samples = np.delete(features, i, axis=0)
        train_labels = np.delete(labels, i, axis=0)
        nearest_label = None
        nearest_distance = float('inf')
        # Find nearest neighbor
        # Calculate the distance between the test sample and all training samples by euclidean distance
        for j in range(len(train_samples)):
            distance = math.sqrt(np.sum((test_sample - train_samples[j]) ** 2))
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_label = train_labels[j]

        # Check if the predicted label is correct
        if nearest_label == labels[i]:
            accuracy += 1

    # Calculate the accuracy
    accuracy /= num_samples
    # print("Accuracy is ", accuracy)
    # return random.random()
    return accuracy

def Cross_Validate_Leave_One_Out(data, current_set, feature_to_remove):
    # This function should return the accuracy of the model with the current feature set and the new feature removed via leave-one-out cross-validation
    num_features = data.shape[1] - 1
    num_samples = data.shape[0]

    set_to_test = current_set.copy()
    set_to_test.remove(feature_to_remove)
    labels = data[:, 0]
    features = data[:, set_to_test]

    accuracy = 0

    for i in range(num_samples):
        # Leave one out
        test_sample = features[i]
        train_samples = np.delete(features, i, axis=0)
        train_labels = np.delete(labels, i, axis=0)
        nearest_label = None
        nearest_distance = float('inf')
        # Find nearest neighbor
        # Calculate the distance between the test sample and all training samples by euclidean distance
        for j in range(len(train_samples)):
            distance = math.sqrt(np.sum((test_sample - train_samples[j]) ** 2))
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_label = train_labels[j]

        # Check if the predicted label is correct
        if nearest_label == labels[i]:
            accuracy += 1

    # Calculate the accuracy
    accuracy /= num_samples
    return accuracy
    

def main():
    # IN PROGRESS
    print("Welcome to Steven Ryan Leonido's Feature Search Algorithm.")
    print("Type the name of the file to test: ")
    input_file = input()
    if input_file == 'wdbc.txt':
        data = load_data_custom(input_file)
        # print(data[:5])  # Print first 5 rows for verification
    else:
        data = []
        f = open(input_file, 'r')
        for line in f:
            line = line.strip()
            columns = line.split()
            data.append([float(x) for x in columns])

        f.close()
        data = np.array(data)
    print("Type the number of the algorithm you want to run.\n\n   1.Forward Selection\n   2.Backward Elimination\n    ")
    choice = input()
    if choice == '1':
        Feature_Search_Forward(data)
    elif choice == '2':
        Feature_Search_Backward(data)
    else:
        print("Invalid choice, exiting.")
        return

    # Testing the Cross_Validate function
    # current_set = [1, 2]
    # feature_to_add = 3
    # acc = Cross_Validate(data, current_set, feature_to_add)
    # print("Accuracy of the model with features", current_set, "and feature", feature_to_add, "is", acc)
    return

if __name__ == "__main__":
    main()