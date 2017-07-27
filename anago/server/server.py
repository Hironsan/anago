import json
import os
import tornado.ioloop
import tornado.web

from anago.tagger import get_tagger
model, vocab_tags, processing_word = get_tagger()


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

BASE_DIR = os.path.dirname(__file__)

application = tornado.web.Application([
        (r'/', MainHandler),
        ],
        template_path=os.path.join(BASE_DIR, 'templates'),
        static_path=os.path.join(BASE_DIR, 'static'),
)

if __name__ == '__main__':
    application.listen(8000)
    tornado.ioloop.IOLoop.current().start()
