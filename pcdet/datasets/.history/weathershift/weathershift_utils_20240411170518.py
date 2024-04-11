def read_label(path2label):
    # [name, cx, cy, cz, dx, dy, dz, heading]
    with open(path2label, 'r') as f:
        lines = readlines(f)