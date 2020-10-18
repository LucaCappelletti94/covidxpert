import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import Metric

class CustomMetric(Metric):
    def __init__(self, name, **kwargs):
        super(CustomMetric, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.tn = self.add_weight(name='tn', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')
        self.result = self.add_weight(name=name, initializer='zeros')

    def result(self):
        return self.result

    def reset_states(self):
        self.tp.assign(0)
        self.fp.assign(0)
        self.tn.assign(0)
        self.fn.assign(0)
        self.result.assign(0)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_pos = K.round(K.clip(y_pred, 0, 1))
        y_pred_neg = 1 - y_pred_pos

        y_pos = K.round(K.clip(y_true, 0, 1))
        y_neg = 1 - y_pos

        self.tp.assign_add(K.sum(y_pos * y_pred_pos))
        self.fp.assign_add(K.sum(y_neg * y_pred_pos))
        self.fn.assign_add(K.sum(y_pos * y_pred_neg))
        self.tn.assign_add(K.sum(y_neg * y_pred_neg))

        self.result.assign(self._custom_metric())

    def _custom_metric(self):
        raise NotImplementedError("This method should be implemented by subclasses")

class F1Score(Metric):
    def __init__(self, name, **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        
    def _custom_metric(self):
        return self.tp / (self.tp + 0.5 * (self.fp + self.fn) + tf.keras.backend.epsilon())

class MatthewsCorrelationCoefficinet(Metric):
    def __init__(self, name, **kwargs):
        super(F1MatthewsCorrelationCoefficinetcore, self).__init__(name=name, **kwargs)

    def _custom_metric(self):
        numerator = (self.tp * self.tn - self.fp * self.fn)
        denominator = tf.math.sqrt((self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn))
        return numerator / (denominator + tf.keras.backend.epsilon())
