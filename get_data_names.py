import os

def get_names(data_path=""):    # to extract dataset names and store them in a LIST
    dat_names = [] 
    for path in os.listdir(data_path):
        dat_names.append(data_path+path+"/")
    return dat_names

# main function only for testing
if __name__=="__main__": 
    data_path = "/home/rohit/Documents/Research/data_prep/HDD_data/GenHW/" # for testing
    x = get_names(data_path)
    print(len(x))
    print(x)