import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import Metric
from tensorflow.keras.backend import epsilon

class CustomMetric(Metric):
    def __init__(self, name, **kwargs):
        super(CustomMetric, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.tn = self.add_weight(name='tn', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')
        self._result = self.add_weight(name=name, initializer='zeros')

    def result(self):
        return self._result

    def reset_states(self):
        self.tp.assign(0)
        self.fp.assign(0)
        self.tn.assign(0)
        self.fn.assign(0)
        self._result.assign(0)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, self.dtype)
        y_pred = tf.cast(y_pred, self.dtype)

        y_pred_pos = K.round(K.clip(y_pred, 0, 1))
        y_pred_neg = 1 - y_pred_pos

        y_pos = K.round(K.clip(y_true, 0, 1))
        y_neg = 1 - y_pos

        self.tp.assign_add(K.sum(y_pos * y_pred_pos))
        self.fp.assign_add(K.sum(y_neg * y_pred_pos))
        self.fn.assign_add(K.sum(y_pos * y_pred_neg))
        self.tn.assign_add(K.sum(y_neg * y_pred_neg))

        self._result.assign(self._custom_metric())

    def _custom_metric(self):
        raise NotImplementedError("This method should be implemented by subclasses")

class Recall(CustomMetric):
    def _custom_metric(self):
        return self.tp / (self.tp + self.fn + epsilon())

class Specificity(CustomMetric):
    def _custom_metric(self):
        return self.tn / (self.tp + self.fn + epsilon())

class Precision(CustomMetric):
    def _custom_metric(self):
        return self.tp / (self.tp + self.fp + epsilon())

class NegativePredictiveValue(CustomMetric):
    def _custom_metric(self):
        return self.tn / (self.fn + self.tn + epsilon())

class MissRate(CustomMetric):
    def _custom_metric(self):
        return self.fn / (self.fn + self.tp + epsilon())

class FallOut(CustomMetric):
    def _custom_metric(self):
        return self.fp / (self.fp + self.tn + epsilon())

class FalseDiscoveryRate(CustomMetric):
    def _custom_metric(self):
        return self.fp / (self.fp + self.tp + epsilon())

class FalseOmissionRate(CustomMetric):
    def _custom_metric(self):
        return self.fn / (self.fn + self.tn + epsilon())

class PrevalenceThreshold(CustomMetric):
    def _custom_metric(self):
        tpr = self.tp / (self.tp + self.fn + epsilon())
        tnr = self.tn / (self.tn + self.fp + epsilon())
        return (tf.math.sqrt(tpr*(1 - tnr)) + tnr - 1) / (tpr + tnr - 1 + epsilon())

class ThreatScore(CustomMetric):
    def _custom_metric(self):
        return self.tp / (self.tp + self.fn + self.fp + epsilon())
        
class Accuracy(CustomMetric):
    def _custom_metric(self):
        return (self.tp + self.fn) / (self.tp + self.tn + self.fp + self.fn + epsilon())
        
class BalancedAccuracy(CustomMetric):
    def _custom_metric(self):
        tpr = self.tp / (self.tp + self.fn + epsilon())
        tnr = self.tn / (self.tn + self.fp + epsilon())
        return (tpr + tnr) / 2

class F1Score(CustomMetric):
    def _custom_metric(self):
        return self.tp / (self.tp + 0.5 * (self.fp + self.fn) + epsilon())

class MatthewsCorrelationCoefficinet(CustomMetric):
    def _custom_metric(self):
        numerator = (self.tp * self.tn - self.fp * self.fn)
        denominator = tf.math.sqrt((self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn))
        return numerator / (denominator + epsilon())

class FowlkesMallowsIndex(CustomMetric):
    def _custom_metric(self):
        tpr = self.tp / (self.tp + self.fn + epsilon())
        tnr = self.tn / (self.tn + self.fp + epsilon())
        return (tpr + tnr) / 2

class Informedness(CustomMetric):
    def _custom_metric(self):
        tpr = self.tp / (self.tp + self.fn + epsilon())
        ppv = self.tp / (self.tp + self.fp + epsilon())
        return tf.math.sqrt(ppv * tpr)

class Markedness(CustomMetric):
    def _custom_metric(self):
        ppv = self.tp / (self.tp + self.fp + epsilon())
        npv = self.tn / (self.tn + self.fn + epsilon())
        return ppv + npv - 1
