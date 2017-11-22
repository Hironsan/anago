from anago.reader import batch_iter

from anago.metrics import F1score


class Evaluator(object):

    def __init__(self,
                 model,
                 preprocessor=None):

        self.model = model
        self.preprocessor = preprocessor

    def eval(self, x_test, y_test):

        # Prepare test data(steps, generator)
        train_steps, train_batches = batch_iter(x_test,
                                                y_test,
                                                batch_size=20,  # Todo: if batch_size=1, eval does not work.
                                                shuffle=False,
                                                preprocessor=self.preprocessor)

        # Build the evaluator and evaluate the model
        f1score = F1score(train_steps, train_batches, self.preprocessor)
        f1score.model = self.model
        f1score.on_epoch_end(epoch=-1)  # epoch takes any integer.
