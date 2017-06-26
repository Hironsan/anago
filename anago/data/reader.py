import os


def doc_read(file):
    sent = []
    with open(file) as f:
        for line in f.readlines():
            if line[0] in '#*+':
                continue
            if line.startswith('EOS'):
                continue
            tokens = line.split()
            word = tokens[0]
            sent.append(word)
    return sent
            
        


def reader(file_or_dir):
    docs = []
    for file_or_dir in os.listdir(file_or_dir):
        if not os.path.isdir(file_or_dir):
            continue
        for file in os.listdir(file_or_dir):
            if file.endswith('.KNP'):
                path = os.path.join(file_or_dir, file)
                doc = doc_read(path)
                docs.append(doc)
    return docs


if __name__ == '__main__':
    docs = reader('.')
    print(len(docs))
    print(''.join(docs[0]))
    print(sum([doc.count('。') for doc in docs]))
