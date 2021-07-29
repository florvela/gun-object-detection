import os
import pickle
import pdb
from matplotlib import pyplot as plt

path = "../output/ssd_train_output_32_batches/"
log_files = [f for f in os.listdir(path) if f.startswith("losses")]
h5_files = [f for f in os.listdir(path) if not f.startswith("losses")]

epochs_saved = {}

for file in h5_files:
    temp = file.strip(".h5").split("_")
    epochs_saved[temp[2]] = temp[4]

losses_32 = []

for file in log_files:
    losses_temp = pickle.load(open(path + file, "rb"))
    max_index = (len(losses_temp)//10)*10
    losses_temp = losses_temp[:max_index]
    losses_32 += losses_temp

path = "../output/ssd_train_output_64_batches/"
log_files = [f for f in os.listdir(path) if f.startswith("losses")]
h5_files = [f for f in os.listdir(path) if not f.startswith("losses")]

epochs_saved = {}

for file in h5_files:
    temp = file.strip(".h5").split("_")
    epochs_saved[temp[2]] = temp[4]

losses_64 = []

for file in log_files:
    losses_temp = pickle.load(open(path + file, "rb"))
    max_index = (len(losses_temp)//10)*10
    losses_temp = losses_temp[:max_index]
    losses_64 += losses_temp


plt.plot(range(1,len(losses_32)+1), losses_32)
plt.plot(range(1,len(losses_64)+1), losses_64)
plt.show()
