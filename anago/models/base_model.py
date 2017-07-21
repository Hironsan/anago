from keras.models import Model, load_model, save_model


class BaseModel(object):

    def __init__(self):
        self.batch_size = None
        self.epoch_size = None

    def train(self, y, *X):
        self._build_model()
        self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epoch_size)

    def predict(self, X):
        y_pred = self.model.predict(X, batch_size=1)
        return y_pred

    def evaluate(self, y, *X):
        score = self.model.evaluate(X, y, batch_size=1)
        return score

    def save(self, filepath):
        save_model(self.model, filepath)

    def load(self, filepath):
        self.model = load_model(filepath=filepath)

    def _build_model(self):
        pass
