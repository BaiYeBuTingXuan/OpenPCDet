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

            cx = float(line[1])
            cy = float(line[2])
            cz = float(line[3])

            dx = float(line[4])
            dy = float(line[5])
            dz = float(line[6])

            heading = float(line[7])

            obj['box'] = [cx,cy,cz,dx,dy,dz,heading]

            objects_list.append(obj)
            
    return objects_list