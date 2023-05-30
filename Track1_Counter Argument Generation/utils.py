

def load_data(path, batch_size):
    with open(path, 'r', encoding = 'UTF-8') as f:
        lines = f.readlines()
    
    dataset = []
    for i in range(len(lines)//batch_size):
        this_batch = []
        for line in lines[i*batch_size : (i+1)*batch_size]:
            topic, source, target = line.strip('\n').split('\t')
            this_batch.append((topic, source, target))
        dataset.append(this_batch)
    return dataset
