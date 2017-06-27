import os
import re


class Reader(object):

    def __init__(self):
        pass

    def read_entity(self, path):
        for file_path in self._traverse(path):
            if not file_path.endswith('.KNP'):
                continue
            for line in self._read_lines(file_path):
                pass

    def _filter(self, line):
        if line[0] in '#*+':
            ne_type, ne_word = self._extract_named_entity(line)
        if line.startswith('EOS'):
            return True
        tokens = line.split()
        word = tokens[0]

    def _traverse(self, path):
        """
        Return all file path included specified path
        :param path:
        :return: all file path
        """
        file_paths = []
        for file_or_dir in os.listdir(path):
            _path = os.path.join(path, file_or_dir)
            if os.path.isdir(_path):
                file_paths.extend(self._traverse(_path))
            else:
                file_paths.append(_path)
        return file_paths

    def _read_lines(self, file_path):
        with open(file_path) as f:
            for line in f:
                yield line

    def _extract_named_entity(self, line):
        name_tag = self._extract_name_tag(line)
        ne_type = self._extract_type(name_tag)
        ne_word = self._extract_target(name_tag)
        return ne_type, ne_word

    def _extract_substring(self, text, substring):
        m = re.search(substring, text)
        if m:
            tag = m.group(1)
            return tag
        else:
            return ''

    def _extract_name_tag(self, text):
        return self._extract_substring(text, '<ne (.+?)/>')

    def _extract_type(self, text):
        return self._extract_substring(text, 'type="(.+?)"')

    def _extract_target(self, text):
        return self._extract_substring(text, 'target="(.+?)"')


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
            

def reader(path):
    docs = []
    for file_or_dir in os.listdir(path):
        dir_path = os.path.join(path, file_or_dir)
        if not os.path.isdir(dir_path):
            continue
        for file in os.listdir(dir_path):
            if file.endswith('.KNP'):
                file_path = os.path.join(dir_path, file)
                doc = doc_read(file_path)
                docs.append(doc)
    return docs


if __name__ == '__main__':
    docs = reader('.')
    print(len(docs))
    print(''.join(docs[0]))
    print(sum([doc.count('。') for doc in docs]))
