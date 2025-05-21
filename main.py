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
    print("Beginning Search\n\n")

    for i in range(num_features):
        print("In level " + str(i+1) + " of tree")
        for k in range(num_features):
            print("- - - Considering feature", k)


    pass

def Cross_Validate(data, current_set, feature_to_add):
    # To Be Added
    # Placeholder for cross-validation logic
    return random.random()

def main():
    # IN PROGRESS
    Feature_Search(data)
    return

if __name__ == "__main__":
    main()