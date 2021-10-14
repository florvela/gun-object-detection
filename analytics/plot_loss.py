import os
import operator
import matplotlib.pyplot as plt

# TODO: leer esto https://towardsdatascience.com/implementing-single-shot-detector-ssd-in-keras-part-vi-model-evaluation-c519852588d1

ssd_dir = "../training/SSD/output/"
ssd_experiments = ['ssd_train_output_32_batches','ssd_train_output_32_batches_augmented', 'ssd_train_output_64_batches']

fig, ax = plt.subplots()
for exp in ssd_experiments:
    x_points = []
    y_points = []
    files = [f for f in os.listdir(ssd_dir+exp) if f.startswith("cp_ep_")]
    for filename in files:
        splitted = filename.strip(".h5").split("_")
        x_points.append(splitted[2])
        y_points.append(splitted[4])
    x_points = list(map(int, x_points))
    y_points = list(map(float, y_points))

    x_sorted = sorted(x_points)

    zipped = zip(x_points,y_points)
    res = sorted(zipped, key = operator.itemgetter(0))
    print(res)
    x_sorted = [elem[0] for elem in res]
    y_sorted = [elem[1] for elem in res]

    ax.plot(x_sorted,y_sorted)
plt.show()

yolo_dir = "../training/YOLO/darknet/output/"
f = open(yolo_dir+"yolov4222.log")

lines = [line.rstrip("\n") for line in f.readlines()]

numbers = {'1', '2', '3', '4', '5', '6', '7', '8', '9'}

iters = []
loss = []
_map = []

fig, ax = plt.subplots()

prev_line = ""
for i, line in enumerate(lines):
    args = line.split(' ')
    if len(args) > 1:
        if args[1][-1] == ':':
            if args[1][0] in numbers:
                iters.append(int(args[1][:-1]))
                loss.append(float(args[3].strip(',')))
                map_line = lines[i-1].split(' ')
                if map_line[1] == '(next':
                    _map.append(0)
                else:
                    _map.append(float(map_line[-3]))

ax.plot(iters, loss)
ax.plot(iters, _map)
plt.xlabel('iters')
plt.ylabel('loss')
plt.grid()

ticks = range(0, 250, 10)

# ax.set_yticks(ticks)
plt.show()