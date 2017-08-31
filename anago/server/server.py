import json
import os
import tornado.ioloop
import tornado.web

import anago
from anago.config import ModelConfig
from anago.data.preprocess import WordPreprocessor

SAVE_ROOT = os.path.join(os.path.dirname(__file__), '../../models')
model_config = ModelConfig()
p = WordPreprocessor.load(os.path.join(SAVE_ROOT, 'preprocessor.pkl'))
model_config.vocab_size = len(p.vocab_word)
model_config.char_vocab_size = len(p.vocab_char)
weights = 'model_weights.h5'
tagger = anago.Tagger(model_config, weights, save_path=SAVE_ROOT, preprocessor=p)


class MainHandler(tornado.web.RequestHandler):

    def get(self):
        self.render('index.html', sent='')

    def post(self):
        sent = self.get_argument('sent')
        entities = tagger.get_entities(sent)
        if entities:
            self.write(json.dumps(dict(entities)))


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
