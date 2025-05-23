import numpy as np
import random

input_file = 'CS205_small_Data__33.txt'

f = open(input_file, 'r')

data = []

for line in f:
    line = line.strip()
    columns = line.split()
    data.append([float(x) for x in columns])

f.close()
data = np.array(data)
# print(data)
print(data.shape)
print(data[0:5, :])

num_features = data.shape[1] - 1
num_samples = data.shape[0]

def Feature_Search(data):
    # IN PROGRESS
    print("Beginning Search\n")

    current_feature_set = []
    best_total_acc = 0
    best_feature_set = []

    for i in range(num_features):
        print("In level " + str(i+1) + " of tree")
        feature_to_add = -1
        best_acc_so_far = 0
        for k in range(num_features):
            if k in current_feature_set:
                continue
            print("- - - Considering feature", k)
            acc = Cross_Validate(data, current_feature_set, k)

            if acc > best_acc_so_far or feature_to_add == -1:
                best_acc_so_far = acc
                feature_to_add = k

        if best_acc_so_far > best_total_acc:
            best_total_acc = best_acc_so_far
            current_feature_set.append(feature_to_add)
            best_feature_set = current_feature_set.copy()
            print("Best feature set so far in the whole test is ", best_feature_set)
            print("Best accuracy so far in the whole test is ", best_total_acc)
        else:
            current_feature_set.append(feature_to_add)
            print("This level does not improve the accuracy")
            print("Best feature to add is", feature_to_add)
            print("Best accuracy so far is", best_acc_so_far)

    print("\n\nBest feature set is ", best_feature_set)
    print("Best accuracy is ", best_total_acc)

    pass

def Cross_Validate(data, current_set, feature_to_add):
    # To Be Added    
    # This function should return the accuracy of the model with the current feature set and the new feature added via leave-one-out cross-validation
    

    set_to_test = current_set.copy()
    set_to_test.append(feature_to_add)

    labels = data[:, 0]
    features = data[:, set_to_test]
    print("Features to test are ", features)
    print("Labels to test are ", labels)

    # Here you would implement the logic for leave-one-out cross-validation
    # Do leave one out and find nearest neighbor

    

    # return random.random()
    # return

def main():
    # IN PROGRESS
    # Feature_Search(data)

    # Testing the Cross_Validate function
    current_set = [1, 2]
    feature_to_add = 3
    acc = Cross_Validate(data, current_set, feature_to_add)
    print("Accuracy of the model with features", current_set, "and feature", feature_to_add, "is", acc)
    return

if __name__ == "__main__":
    main()