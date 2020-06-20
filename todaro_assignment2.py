#Author: James Todaro
#Date: 06/15/2020
#Purpose: To implement a knn algorithm

# Datasets:
#   ecoli.data
#   forestfires.data
#   machine.data
#   irissegmentation.data


import sys
import csv
import math
import copy
import random
 

"""This function accepts a filename of a file which should be in csv format, and returns it as a list
"""
def read_data(filename):
    with open(filename, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        data = []
        for row in csv_reader:
            data.append([str(v) for v in row])
    return data
def write_data(filename, data):
    f = open(filename, "w+")
    for record in data:
        for index in range(len(record) - 1):
            f.write(str(record[index]) + ",")
        f.write(str(record[-1]) + "\n")
    f.close()

def get_test_data(data):
    test_data = []
    validation_data = []
    index_list = [index for index in range(len(data))] #create a list of all valid indices
    counter = 0
    while (len(index_list) > 0): 
        random_index = 0 if len(index_list) == 1 else random.randrange(0,len(index_list)-1)
        if counter % 2 == 0: test_data.append(data[index_list[random_index]])
        else: validation_data.append(data[index_list[random_index]])
        index_list.remove(index_list[random_index]) #remove the index from the index list after it has been used
        counter += 1 
        counter %= 3 #reset the counter when it gets to 3
    return [test_data, validation_data]

""" This function will take a dataset and return a list of lists with each inner list containing the same class
"""
def get_list_of_classes(raw_data):
    class_list =[]
    for record in raw_data:
        class_found = False
        for classes in class_list:
            if classes[0][-1] == record[-1]:
                classes.append(record)
                class_found = True

        if not(class_found):
            class_list.append([record])
    return class_list

""" This function will split the records into 8 seperate groups according to class ranges so that the same proportions can be 
    maintained when creating the folds.
"""
def get_list_of_classes_for_machine_data(raw_data):
    class_list = [[],[],[],[],[],[],[],[]]
    for record in raw_data:
        if record[-1] <= 20:
            class_list[0].append(record)
        elif record[-1] > 20 & record[-1] <= 100:
            class_list[1].append(record)
        elif record[-1] > 100 & record[-1] <= 200:
            class_list[2].append(record)
        elif record[-1] > 200 & record[-1] <= 300:
            class_list[3].append(record)
        elif record[-1] > 300 & record[-1] <= 400:
            class_list[4].append(record)
        elif record[-1] > 400 & record[-1] <= 500:
            class_list[5].append(record)
        elif record[-1] > 500 & record[-1] <= 600:
            class_list[6].append(record)
        elif record[-1] > 600:
            class_list[7].append(record)
        else:
            print("ERROR")
    return class_list

""" normalize data in a dataset so all values are between 0 and 1.  Min and max values for each attribute will be calculated, then
    the min value will be subtracted from each point and that will be divided by the max - min value for each attribute.
"""
def normalize_data_set(data):
    min_values = data[0][0:-1]
    max_values = data[0][0:-1]
    for record_index in range(1, len(data)):
        for attribute_index in range(len(data[0]) - 1):
            if data[record_index][attribute_index] < min_values[attribute_index]:
                min_values[attribute_index] = data[record_index][attribute_index]
            if data[record_index][attribute_index] > max_values[attribute_index]:
                max_values[attribute_index] = data[record_index][attribute_index]
    for record in data:
        for attribute_index in range(len(record) - 1):
            record[attribute_index] = (record[attribute_index]- min_values[attribute_index])/(max_values[attribute_index] - min_values[attribute_index])
            print(record)

""" Process the ecoli data set.  Delete index 0, and change the number data types from string to float
"""
def process_ecoli_data(list_of_records):
    new_list = []
    for record in list_of_records:
        new_list.append(record[0].split()[1:])
    for record_num in range(len(new_list)):
        for attribute_num in range(len(new_list[record_num]) - 1):
            new_list[record_num][attribute_num] = float(new_list[record_num][attribute_num])
    return new_list

""" Process the forestfire data set.  Change months names to numbers (1 - 12), and days of the week to numbers (1 - 7)
    Change the classifier to a nubmer from 0 - 3 using log(10, classifier) as suggested by forestfire.names
"""
def process_forestfire_data(list_of_records):
    processed_data = []
    first_record = True
    for record in list_of_records:
        if first_record: first_record = False # the first record contains the headers
        else:
            new_record = []
            for index in range(len(record)):
                if index == 2:
                    if record[index] == 'jan': new_record.append(1)
                    elif record[index] == 'feb': new_record.append(2)
                    elif record[index] == 'mar': new_record.append(3)
                    elif record[index] == 'apr': new_record.append(4)
                    elif record[index] == 'may': new_record.append(5)
                    elif record[index] == 'jun': new_record.append(6)
                    elif record[index] == 'jul': new_record.append(7)
                    elif record[index] == 'aug': new_record.append(8)
                    elif record[index] == 'sep': new_record.append(9)
                    elif record[index] == 'oct': new_record.append(10)
                    elif record[index] == 'nov': new_record.append(11)
                    elif record[index] == 'dec': new_record.append(12)
                    else: print("MONTH NOT FOUND")
                elif index == 3:
                    if record[index] == 'mon': new_record.append(1)
                    elif record[index] == 'tue': new_record.append(2)
                    elif record[index] == 'wed': new_record.append(3)
                    elif record[index] == 'thu': new_record.append(4)
                    elif record[index] == 'fri': new_record.append(5)
                    elif record[index] == 'sat': new_record.append(6)
                    elif record[index] == 'sun': new_record.append(7)
                    else: print("MONTH NOT FOUND")
                else:
                    new_record.append(float(record[index]))
            processed_data.append(new_record)
    return processed_data

""" process machine.data data set.  Exlude the vendor name and model name and only include the attributes which have a numerical value.
    The classifier is continuous value.
"""
def process_machine_data(list_of_records):
    processed_data = []
    for record in list_of_records:
        new_record = []
        for attribute_num in  range(2, len(record) - 1):
            new_record.append(int(record[attribute_num]))
        print(new_record)
        processed_data.append(new_record)
    return processed_data

""" process segmentation.data data set. 
    
"""
def process_segmentation_data(list_of_records):
    processed_data = []
    for record in list_of_records:
        new_record = []
        for attribute_num in  range(1, len(record)):
            new_record.append(float(record[attribute_num]))
        new_record.append(record[0])
        processed_data.append(new_record)
    return processed_data

""" This functions assumes that format of data is such that contains lists of records separated by class.
    This function will create n folds of data with roughly the same ratio of data in each fold.
    Returns n folds of the original data.
"""

def get_folds(data, n):
    folds = []
    num_instances = []

    for index in range(n): folds.append([])
    for index in range(len(data)): 
        num_instances.append(int(len(data[index]) / n)) # this is the minimum number of each class that should appear in each fold

    data_copy = copy.deepcopy(data) # make a copy of the data
    
    for class_list_num in range(len(data_copy)):
        for fold_index in range(n):
            for record_num in range(num_instances[class_list_num]): # randomly select a number of elements from each class to maintain the origianl ratio of classes
                if (len(data_copy[class_list_num]) > 0):
                    folds[fold_index].append(data_copy[class_list_num].pop(random.randint(0, len(data_copy[class_list_num]) - 1)))

    fold_index = n - 1
    while(len(data_copy) > 0): # randomly select from the remaining records to populate the last few records
        class_index = random.randint(0, len(data_copy) - 1)
        if (len(data_copy[class_index]) == 0): data_copy.pop(class_index)
        else: 
            folds[fold_index].append(data_copy[class_index].pop(random.randint(0, len(data_copy[class_index]) - 1)))
            fold_index = fold_index - 1 if fold_index - 1 >= 0 else n - 1

    return folds

"""
    returns all data from all data except those in folds[i]
"""
def get_training_data(folds, i):
    training_data = []
    for idx in range(len(folds)):
        if idx != i:
            for data_point in folds[idx]:
                training_data.append(data_point)
    return training_data

""" This functions uses the k nearest neighbors algorithm to classify a test set of data given a training set of data, and a number of points k.
    It returns the percent classified correctly.
"""
def knn(training_data, validation_data, k, class_list, fill_to_k = False):
    total_correct = 0
    for record in validation_data:
        if classify(record, training_data, k, class_list, fill_to_k) == record[-1]:
            total_correct += 1
    return total_correct / len(validation_data)

""" This function returns the mean squared error of the classification of the validation data.
"""
def knn_with_regression(training_data, validation_data, k, is_forestfire_data = False):
    MSE = 0
    for record in validation_data:
        MSE += (record[-1] - classify_with_regression(record, training_data, k, is_forestfire_data))**2
    return MSE/len(validation_data)

""" Edited knn algorithm.  This algorithm will attempt to classify each element in the training set.  Reach element that is 
    INCORRECTLY classified will be removed from the training set
"""
def edited_knn(training_data, validation_data, k, class_list):
    training_copy = copy.deepcopy(training_data)
    num_edited = 1 # inistilialize variable to enter loop
    print(len(training_copy))
    while (num_edited > 0):
        num_edited = 0 # initializ variable at the beginning of each loop
        for example in training_copy:
            c = classify(example, training_copy, k, class_list)
            print(c, "\t", example[-1])
            if c != example[-1]:
                training_copy.remove(example)
                num_edited += 1
        print("removed", num_edited, "examples")
    print(len(training_copy))
    total_correct = 0
    for record in validation_data:
        if classify(record, training_copy, k, class_list) == record[-1]:
            total_correct += 1
    return total_correct / len(validation_data)   

""" Condensed knn algorithm.  This algorithm will begin with an empty set of training data.
"""
def condensed_knn(training_data, validation_data, k, class_list):
    Z = []
    X = copy.deepcopy(training_data)
    print("Original Size:", len(X))
    while(True):
        
        size_of_Z = len(Z)
        if size_of_Z == 0:
            Z.append(X.pop(random.randint(0, len(X) - 1 ))) # intialize Z to a random element from X
        else:
            for x in X: # for every x in the training set X
                distances = [] # initialize distances
                for z in Z: # for every z in the new condensed set Z
                    distances.append(euclidean_distance(x, z)) # find all the distances from x to every element in Z
                    index_of_min = distances.index(min(distances)) #find the element
                if x[-1] != Z[index_of_min][-1]:
                    Z.append(X.pop(X.index(x)))
        
        if size_of_Z == len(Z):
            break
    print("Condensed Size:", len(Z))
    total_correct = 0
    for record in validation_data:
        if classify(record, Z, k, class_list) == record[-1]:
            total_correct += 1
    return total_correct / len(validation_data)


""" This function returns the classifier with least minimum of k points from the classifier
    If there are not at least k points of a class that classifier is ignored.
"""
def classify(data_point, training_data, k, class_list, fill_to_k = False):
    distances = []
    min_distances = []
    min_classifier_index = -1
    min_sum = 0
    for classifier_number in range(len(class_list)): # get the distances to all points in the training set
        distances.append([])
        for record in training_data:
            if record[-1] == class_list[classifier_number]:
                distances[classifier_number].append(euclidean_distance(data_point, record, True))

    for classifier_number in range(len(class_list)): # find the index of the class with the minimum sum of the least k distances
        if len(distances[classifier_number]) >= k:
            distances[classifier_number].sort()
            min_distances.append(distances[classifier_number][0:k])

            if min_classifier_index == -1:
                min_classifier_index = classifier_number
                min_sum = sum(min_distances[classifier_number])
            elif sum(min_distances[classifier_number]) < min_sum:
                min_sum = sum(min_distances[classifier_number])
                min_classifier_index = classifier_number
        elif len(distances[classifier_number]) > 0 & fill_to_k: #if there arent enough instances of a class, use the largest distance and fill to k records
            min_distances.append(distances[classifier_number])
            while len(min_distances[classifier_number]) < k:
                min_distances[classifier_number].append(distances[classifier_number][-1])
            if min_classifier_index == -1:
                min_classifier_index = classifier_number
                min_sum = sum(min_distances[classifier_number])
            elif sum(min_distances[classifier_number]) < min_sum:
                min_sum = sum(min_distances[classifier_number])
                min_classifier_index = classifier_number
        else:
            min_distances.append([])
    return class_list[min_classifier_index]

""" This function will take a data point, and training set and find the k nearest points to that data point.
    It will then return the average of the k nearest points classifiers.
"""

def classify_with_regression(data_point, training_data, k, is_forestfire_data = False):
    distances = []
    training_data_copy = copy.deepcopy(training_data)
    for record in training_data_copy:
        distances.append(euclidean_distance(data_point, record, is_forestfire_data))
    k_nearest_neighbors = []
    for k_neighbor_number in range(k): #search k times
        min_index = 0 #initialize min_index to first record
        min_distance = distances[0] #initialize min_distance to first distance
        for index in range (1, len(distances)):  #search through all the distances
            if distances[index] < min_distance:  #if you find a new distance that is shorter
                min_index = index   #save the index
                min_distance = distances[index]  #save the distance
        k_nearest_neighbors.append(training_data_copy.pop(min_index))    # pop the record that corresponds with the shortest distance
        distances.pop(min_index) #pop the shortest distance
    sum = 0
    for k_record in k_nearest_neighbors:
        sum += k_record[-1]
    return sum/k


""" this function returns the eucliean distance between two vectors x and y.
    This function assumes that the classifier is in the last index for each vector.
"""
def euclidean_distance(x, y, is_forestfire_data = False):
    if len(x) != len(y): print("ERROR")
    sum = 0
    for index in range(len(x) - 1):
        if is_forestfire_data & (index == 2): #to determine distance for months add together and mod 12
            sum += ((x[index] + y[index]) % 12)**2
        elif is_forestfire_data & (index == 3): # to determine distance for days add together and mod 7
            sum += ((x[index] + y[index]) % 7)**2
        else: 
            sum += (x[index] - y[index])**2
    return sum**(0.5)

""" this function returns the eucliean distance between two vectors x and y.
    This function assumes that the classifier is in the last index for each vector.
"""
def manhattan_distance(x,y):
    if len(x) != len(y): print("ERROR")
    sum = 0
    for index in range(len(x) - 1):
        sum += abs(x[index] - y[index])
    return sum

""" this function returns the eucliean distance between two vectors x and y.
    This function assumes that the classifier is in the last index for each vector.
"""
def max_coordinate_distance(x,y):
    if len(x) != len(y): print("ERROR")
    max_distance = x[0] - y[0]
    for index in range(len(x) - 1):
        if abs(x[index] - y[index]) > max_distance:
            max_distance = abs(x[index] - y[index])         
    return max_distance


if __name__ == "__main__":
    #Data set 1: ecoli.data
    
    print("ECOLI DATASET")
    data_to_graph = []
    
    #tune k
    best_ks = []
    global_best = 0
    best_trial = 0
    for trial in range (10):
        ecoli_raw_data = process_ecoli_data(read_data("ecoli.data")) # read the raw data
        ecoli_data_by_class = get_list_of_classes(ecoli_raw_data)
        ecoli_folded_data = get_folds(ecoli_data_by_class, 5)
        most_accurate = 0
        best_k = 1
        data_to_graph.append([])
        for k in range(1, 16):
            accuracy_sum = 0
            for fold_num in range(len(ecoli_folded_data)):
                accuracy_sum += knn(get_training_data(ecoli_folded_data, fold_num), ecoli_folded_data[fold_num], k, ['cp', 'im', 'pp', 'imU', 'om'])
            data_to_graph[trial].append([k, accuracy_sum/len(ecoli_folded_data)])
            if accuracy_sum/len(ecoli_folded_data) > most_accurate:
                most_accurate = accuracy_sum/len(ecoli_folded_data)
                best_k = k
            if accuracy_sum/len(ecoli_folded_data) > global_best:
                global_best = accuracy_sum/len(ecoli_folded_data)
                best_trial = trial
    sums = []
    for k in range(len(data_to_graph[0])):
        sums.append([0])
    for trial in range(len(data_to_graph)):
        for k in range(len(data_to_graph[0])):
            print(sums[k][0], data_to_graph[trial][k][1])
            sums[k][0] += data_to_graph[trial][k][1]
    for k in range(len(sums)):
            print(str(k + 1), sums[k][0]/len(data_to_graph))
            sums[k][0] /= len(data_to_graph)
            sums[k].insert(0, k)
    print (sums)
    write_data("k_values_ecoli.csv", data_to_graph[best_trial])
    write_data("best_k_for_ecoli_data.csv", sums)    
    
    #Data set 2: forestfires.data
    """
    data_to_graph = []   
    print("FOREST FIRES DATASET")
    forestfires_data = read_data("forestfires.data")
    forestfires_processed_data = process_forestfire_data(forestfires_data)
    #normalize_data_set(forestfires_processed_data)
    forestfire_data_by_class = get_list_of_classes(forestfires_processed_data)
    forestfire_folded_data = get_folds(forestfire_data_by_class, 5)
    for k in range(1, 16):
        MSE_sum = 0
        for fold_num in range(len(forestfire_folded_data)):
            MSE_sum += knn_with_regression(get_training_data(forestfire_folded_data, fold_num), forestfire_folded_data[fold_num], k, True)
        data_to_graph.append([k, MSE_sum/len(forestfire_folded_data)])
    write_data("k_values_forestfires_MSE_dcalc.csv", data_to_graph)
    """
    
    #Data set 3: machine.data
    """
    data_to_graph = [] 
    print("MACHINE DATASET")
    machine_raw_data = read_data("machine.data")
    machine_processed_data = process_machine_data(machine_raw_data)
    #normalize_data_set(machine_processed_data)
    machine_data_by_class = get_list_of_classes_for_machine_data(machine_processed_data)
    machine_folded_data = get_folds(machine_data_by_class, 5)
    for k in range(1, 15):
        MSE_sum = 0
        for fold_num in range(len(machine_folded_data)):
            MSE_sum += knn_with_regression(get_training_data(machine_folded_data, fold_num), machine_folded_data[fold_num], k)
        data_to_graph.append([k, MSE_sum/len(machine_folded_data)])
    write_data("k_values_machine.csv", data_to_graph)
    """
    """
    #Data set 4: segmentation.data
    print("SEGMENTATION DATASET")
    data_to_graph = []
    
    #tune k
    best_ks = []
    global_best = 0
    best_trial = 0
    for trial in range (3):
        segmentation_raw_data = read_data("segmentation.data")
        segmentation_processed_data = process_segmentation_data(segmentation_raw_data)
        #normalize_data_set(segmentation_processed_data)
        segmentation_data_by_class = get_list_of_classes(segmentation_processed_data)
        segmentation_folded_data = get_folds(segmentation_data_by_class, 5)
        most_accurate = 0
        best_k = 1
        data_to_graph.append([])
        for k in range(1, 16):
            accuracy_sum = 0
            for fold_num in range(len(segmentation_folded_data)):
                accuracy_sum += knn(get_training_data(segmentation_folded_data, fold_num), segmentation_folded_data[fold_num], k, ["BRICKFACE", "SKY", "FOLIAGE", "CEMENT", "WINDOW", "PATH", "GRASS"])
            data_to_graph[trial].append([k, accuracy_sum/len(segmentation_folded_data)])
            if accuracy_sum/len(segmentation_folded_data) > most_accurate:
                most_accurate = accuracy_sum/len(segmentation_folded_data)
                best_k = k
            if accuracy_sum/len(segmentation_folded_data) > global_best:
                global_best = accuracy_sum/len(segmentation_folded_data)
                best_trial = trial
    sums = []
    for k in range(len(data_to_graph[0])):
        sums.append([0])
    for trial in range(len(data_to_graph)):
        for k in range(len(data_to_graph[0])):
            print(sums[k][0], data_to_graph[trial][k][1])
            sums[k][0] += data_to_graph[trial][k][1]
    for k in range(len(sums)):
            print(str(k + 1), sums[k][0]/len(data_to_graph))
            sums[k][0] /= len(data_to_graph)
            sums[k].insert(0, k)
    print (sums)
    write_data("k_values_segmentation.csv", data_to_graph[best_trial])
    write_data("best_k_for_machine_data.csv", sums)
    """
    #EDITED KNN ECOLI
    """
    print("ECOLI DATASET")
    ecoli_raw_data = process_ecoli_data(read_data("ecoli.data")) # read the raw data
    ecoli_data_by_class = get_list_of_classes(ecoli_raw_data)
    ecoli_folded_data = get_folds(ecoli_data_by_class, 5)
    
    data_to_graph = []
    for k in range(1, 30):
        accuracy_sum = 0
        edited_sum = 0
        for fold_num in range(len(ecoli_folded_data)):
            accuracy_sum += knn(get_training_data(ecoli_folded_data, fold_num), ecoli_folded_data[fold_num], k, ['cp', 'im', 'pp', 'imU', 'om'])
            edited_sum += edited_knn(get_training_data(ecoli_folded_data, fold_num), ecoli_folded_data[fold_num], k, ['cp', 'im', 'pp', 'imU', 'om'])
        data_to_graph.append([k, accuracy_sum/len(ecoli_folded_data), edited_sum/len(ecoli_folded_data)])
    write_data("edited_knn_comp.csv", data_to_graph)
    """
    #CONDENSED KNN ECOLI
    print("ECOLI DATASET")
    ecoli_raw_data = process_ecoli_data(read_data("ecoli.data")) # read the raw data
    ecoli_data_by_class = get_list_of_classes(ecoli_raw_data)
    ecoli_folded_data = get_folds(ecoli_data_by_class, 5)
    
    data_to_graph = []
    for k in range(1, 10):
        accuracy_sum = 0
        edited_sum = 0
        for fold_num in range(len(ecoli_folded_data)):
            accuracy_sum += knn(get_training_data(ecoli_folded_data, fold_num), ecoli_folded_data[fold_num], k, ['cp', 'im', 'pp', 'imU', 'om'])
            edited_sum += condensed_knn(get_training_data(ecoli_folded_data, fold_num), ecoli_folded_data[fold_num], k, ['cp', 'im', 'pp', 'imU', 'om'])
        data_to_graph.append([k, accuracy_sum/len(ecoli_folded_data), edited_sum/len(ecoli_folded_data)])
    write_data("condensed_knn_comp.csv", data_to_graph)