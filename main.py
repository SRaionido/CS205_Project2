import numpy as np

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

def Feature_Search(data):
    # To Be Added
    pass

def Cross_Validate(data, current_set, feature_to_add):
    # To Be Added
    pass

def main():
    # To Be Added
    return

if __name__ == "__main__":
    main()