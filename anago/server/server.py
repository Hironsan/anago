import json
import os
import tornado.ioloop
import tornado.web

from janome.tokenizer import Tokenizer

import anago

tokenizer = Tokenizer()
tagger = anago.Tagger(tokenizer=tokenizer)


class MainHandler(tornado.web.RequestHandler):

    def get(self):
        self.render('index.html', sent='')

    def post(self):
        sent = self.get_argument('sent')
        entities = tagger.get_entities(sent)
        if entities:
            self.write(json.dumps(entities))


BASE_DIR = os.path.dirname(__file__)

application = tornado.web.Application([
    (r'/', MainHandler),
    ],
    template_path=os.path.join(BASE_DIR, 'templates'),
    static_path=os.path.join(BASE_DIR, 'static'),
)

if __name__ == '__main__':
    application.listen(8888)
    tornado.ioloop.IOLoop.current().start()
