import pandas as pd 
import pdb
import os

training_dir = '../../datasets/SasankYadati-Guns-Dataset-0eb7329/Images/'


elems = []
val_elems = []
counter = 0

for elem in os.listdir(training_dir):
    if elem.endswith('.jpeg'):
        elem_name = elem.split('.jpeg')[0]
        counter += 1
        if counter % 5 == 0:
            val_elems.append([elem, elem_name+'.txt'])
        else:
            elems.append([elem, elem_name+'.txt'])

df = pd.DataFrame(elems)
df.to_csv('../../datasets/SasankYadati-Guns-Dataset-0eb7329/train_split_file.csv', sep=' ', header=False, index=False)
df2 = pd.DataFrame(val_elems)
df2.to_csv('../../datasets/SasankYadati-Guns-Dataset-0eb7329/val_split_file.csv', sep=' ', header=False, index=False)
pdb.set_trace()


