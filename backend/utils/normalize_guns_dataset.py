import cv2
import os
import pandas as pd
import pdb
import shutil

d_dir = '../datasets/guns/'
merged = '../datasets/merged_dataset/Labels'
im_merged = '../datasets/merged_dataset/Images'



# elems = []
# val_elems = []
# counter = 0
#
# for file in os.listdir(d_dir):
#     if file.endswith('.txt') and not file.endswith('_normalized.txt'):
#         f_name = d_dir + file.strip('.txt')
#         print(f_name)
#         img = cv2.imread(f_name+'.jpg')
#         dh, dw, _ = img.shape
#
#         fl = open(d_dir+file, 'r')
#         data = fl.readlines()
#         fl.close()
#
#         lines = []
#
#         for dt in data:
#
#             # Split string to float
#             _, x, y, w, h = map(float, dt.split(' '))
#
#             # Taken from https://github.com/pjreddie/darknet/blob/810d7f797bdb2f021dbe65d2524c2ff6b8ab5c8b/src/image.c#L283-L291
#             # via https://stackoverflow.com/questions/44544471/how-to-get-the-coordinates-of-the-bounding-box-in-yolo-object-detection#comment102178409_44592380
#             l = int((x - w / 2) * dw)
#             r = int((x + w / 2) * dw)
#             t = int((y - h / 2) * dh)
#             b = int((y + h / 2) * dh)
#
#             if l < 0:
#                 l = 0
#             if r > dw - 1:
#                 r = dw - 1
#             if t < 0:
#                 t = 0
#             if b > dh - 1:
#                 b = dh - 1
#
#             line = {'xmin':l,'ymin':t,'xmax':r,'ymax':b}
#             lines.append(line)
#
#
#         df = pd.DataFrame(lines)
#         df.to_csv(f_name+'_normalized.txt', sep=' ', index=False)
#         elem_name = file.strip('.txt')
#         counter += 1
#         if counter % 5 == 0:
#             val_elems.append([elem_name + '.jpg', elem_name + '_normalized.txt'])
#         else:
#             elems.append([elem_name + '.jpg', elem_name + '_normalized.txt'])
#
# df = pd.DataFrame(elems)
# df.to_csv('../datasets/guns/train_split_file.csv', sep=' ', header=False,
#           index=False)
# df2 = pd.DataFrame(val_elems)
# df2.to_csv('../datasets/guns/val_split_file.csv', sep=' ', header=False,
#            index=False)

# for file in os.listdir(d_dir):
#     if file.endswith('_normalized.txt'):
#         shutil.copy(d_dir+file, merged)
#     elif file.endswith('jpg'):
#         shutil.copy(d_dir + file, im_merged)


df = pd.read_csv('../datasets/guns/train_split_file.csv', names=['a','b'], sep=' ')
df2 = pd.read_csv('../datasets/sasankguns/train_split_file.csv', names=['a','b'], sep=' ')
df3 = pd.concat([df,df2])
df3.to_csv('../datasets/merged_dataset/train_split_file.csv', sep=' ', header=False, index=False)

df = pd.read_csv('../datasets/guns/val_split_file.csv', names=['a','b'], sep=' ')
df2 = pd.read_csv('../datasets/sasankguns/val_split_file.csv', names=['a','b'], sep=' ')
df3 = pd.concat([df,df2])
df3.to_csv('../datasets/merged_dataset/val_split_file.csv', sep=' ', header=False, index=False)
pdb.set_trace()
