import json
import os
import tornado.ioloop
import tornado.web
from janome.tokenizer import Tokenizer

from anago.tagger import get_tagger, get_jp_tagger
model, vocab_tags, processing_word = get_tagger()
#model_jp, vocab_jp_tags, processing_jp_word = get_jp_tagger()
t = Tokenizer()


def get_chunk(sent, tags):
    res = []
    words = []
    chunk = []
    for w, t in zip(sent, tags):
        chunk_type = 'O' if t == 'O' else t.split('-')[1]
        if chunk == []:
            chunk.append(chunk_type)
            words.append(w)
        elif chunk_type in set(chunk):
            chunk.append(chunk_type)
            words.append(w)
        else:
            res.append((words, chunk[0]))
            chunk = [chunk_type]
            words = [w]
    else:
        res.append((words, chunk[0]))
    return res


class MainHandler(tornado.web.RequestHandler):

    def get(self):
        self.render('index.html', sent='')

    def post(self):
        sent = self.get_argument("sent")
        data = model.interactive_shell(sent, vocab_tags, processing_word)
        print(data)
        if data:
            chunk = get_chunk(data['x'], data['y'])
            print(chunk)
            self.write(json.dumps(chunk))
#        self.render('index.html', sent=sent)
"""

class JpNERHandler(tornado.web.RequestHandler):

    def get(self):
        self.render('index1.html', sent='')

    def post(self):
        sent = self.get_argument("sent")
        sent = ' '.join(t.tokenize(sent, wakati=True))
        data = model_jp.interactive_shell(sent, vocab_jp_tags, processing_jp_word)
        print(data)
        if data:
            chunk = get_chunk(data['x'], data['y'])
            print(chunk)
            self.write(json.dumps(chunk))
"""

BASE_DIR = os.path.dirname(__file__)

application = tornado.web.Application([
    (r'/', MainHandler),
    #(r'/jp', JpNERHandler),
    ],
    template_path=os.path.join(BASE_DIR, 'templates'),
    static_path=os.path.join(BASE_DIR, 'static'),
)

if __name__ == '__main__':
    application.listen(8888)
    tornado.ioloop.IOLoop.current().start()
