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

            obj['heading'] = float(line[7])

            objects_list.append(obj)
            
    return objects_list