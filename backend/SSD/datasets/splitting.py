import os

train_paths = ['train/','test/','valid/']
for train_path in train_paths:
    files = os.listdir(train_path)

    annotation_file = None
    images = []
    for file in files:
        if file.startswith('_annotations'):
            annotation_file = file
        elif file.endswith('.jpg'):
            images.append(file)

    with open(train_path + annotation_file) as f:
        with open(train_path + 'train_split_file.txt', 'w') as split_file:
            for line in f.readlines():
                line = line.split(' ')
                file_name = line[0]
                objects = [l.strip('\n') for l in line[1:]]
                text_file_name = train_path + file_name.strip('.jpg')+'.txt'
                split_file.write(train_path + file_name +' '+ train_path+ text_file_name +'\n')
                # print(objects)
                with open(text_file_name, 'w') as temp_f:
                    for item in objects:
                        temp_f.write("%s\n" % item)