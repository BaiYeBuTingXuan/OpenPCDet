def read_label(path2label):
    # [name, cx, cy, cz, dx, dy, dz, heading]
    objects_list = []
    with open(path2label, 'r') as f:
        lines = f.readlines()
        for line in lines:
            obj = {}

            line = line.strip()
            line = line.split()
            
            obj['name'] = line[0]
            
            obj['cx'] = float(line[1])
            obj['cy'] = float(line[2])
            obj['cz'] = float(line[3])

            obj['dx'] = float(line[4])
            obj['dy'] = float(line[5])
            obj['dz'] = float(line[6])

            label_data = [line_list[0]] + [float(value) for value in line_list[1:]]
            
            # 进行进一步处理，例如保存到列表或者其他操作
            # 这里可以根据具体需求进行相应的处理
            # 比如将每一行的数据存储到一个列表中，或者执行其他操作
            print(label_data)  # 举例输出每一行处理