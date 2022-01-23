with open("class_names.txt", "r") as f:

    data = f.readlines()

    for line in data:
        data_array = line.split("\t")
        print(data_array[0], ": u\'" + data_array[1] + "\'" +",")



    # print(data, data_array)