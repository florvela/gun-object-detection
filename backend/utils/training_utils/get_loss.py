import tensorflow as tf


class SOFTMAX_LOSS:
    """ Calculates the softmax loss between the predicted classes and ground truth classes.

    Args:
        - y_true: The ground truth classes.
        - y_pred: The predicted classes.

    Code Reference:
        - https://github.com/pierluigiferrari/ssd_keras/blob/master/keras_loss_function/keras_ssd_loss.py
    """

    def compute(self, y_true, y_pred):
        y_pred = tf.maximum(y_pred, 1e-15)
        return -1 * tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)


class SMOOTH_L1_LOSS:
    """ Compute smooth l1 loss between the predicted bounding boxes and the ground truth bounding boxes.

    Args:
        - y_true: The ground truth bounding boxes.
        - y_pred: The predicted bounding boxes.

    Code References:
        - https://github.com/pierluigiferrari/ssd_keras/blob/master/keras_loss_function/keras_ssd_loss.py

    Paper References:
        - Girshick, R. (2015). Fast-RCNN. https://arxiv.org/pdf/1504.08083.pdf
        - Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C.Y., & Berg, A. C. (2016).
          SSD: Single Shot MultiBox Detector. https://arxiv.org/abs/1512.02325
    """

    def compute(self, y_true, y_pred):
        abs_loss = tf.abs(y_true - y_pred)
        square_loss = 0.5 * (y_true - y_pred) ** 2
        res = tf.where(tf.less(abs_loss, 1.0), square_loss, abs_loss - 0.5)
        return tf.reduce_sum(res, axis=-1)


class SSD_LOSS:
    """ Loss function as defined in the SSD paper.

    Args:
        - alpha: weight term from the SSD paper. Defaults to 1.
        - min_negative_boxes: the minimum number of negative boxes allowed in the loss calculation. Defaults to 0.
        - negative_boxes_ratio: the ratio of negative boxes to positive boxes. Defaults to 3 (3 times the possible boxes).

    Returns:
        - A tensor of shape (batch_size,) where each item in the tensor represents the loss for each batch item.

    Paper References:
        - Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C.Y., & Berg, A. C. (2016).
          SSD: Single Shot MultiBox Detector. https://arxiv.org/abs/1512.02325

    Code References:
        - https://github.com/pierluigiferrari/ssd_keras/blob/master/keras_loss_function/keras_ssd_loss.py
    """

    def __init__(
        self,
        alpha=1,
        min_negative_boxes=0,
        negative_boxes_ratio=3,
    ):
        self.alpha = alpha
        self.min_negative_boxes = min_negative_boxes
        self.negative_boxes_ratio = negative_boxes_ratio
        self.smooth_l1_loss = SMOOTH_L1_LOSS()
        self.softmax_loss = SOFTMAX_LOSS()

    def compute(self, y_true, y_pred):
        # calculate smooth l1 loss and softmax loss for all boxes
        batch_size = tf.shape(y_true)[0]
        num_boxes = tf.shape(y_true)[1]
        #
        bbox_true = y_true[:, :, -12:-8]
        bbox_pred = y_pred[:, :, -12:-8]
        class_true = y_true[:, :, :-12]
        class_pred = y_pred[:, :, :-12]
        #
        regression_loss = self.smooth_l1_loss.compute(bbox_true, bbox_pred)
        classification_loss = self.softmax_loss.compute(class_true, class_pred)
        #
        negatives = class_true[:, :, 0]  # (batch_size, num_boxes)
        positives = tf.reduce_max(class_true[:, :, 1:], axis=-1)  # (batch_size, num_boxes)
        num_positives = tf.cast(tf.reduce_sum(positives), tf.int32)
        #
        pos_regression_loss = tf.reduce_sum(regression_loss * positives, axis=-1)
        pos_classification_loss = tf.reduce_sum(classification_loss * positives, axis=-1)
        #
        neg_classification_loss = classification_loss * negatives
        num_neg_classification_loss = tf.math.count_nonzero(neg_classification_loss, dtype=tf.int32)
        num_neg_classification_loss_keep = tf.minimum(
            tf.maximum(self.negative_boxes_ratio * num_positives, self.min_negative_boxes),
            num_neg_classification_loss
        )

        def f1():
            return tf.zeros([batch_size])

        def f2():
            neg_classification_loss_1d = tf.reshape(neg_classification_loss, [-1])
            _, indices = tf.nn.top_k(
                neg_classification_loss_1d,
                k=num_neg_classification_loss_keep,
                sorted=False
            )
            negatives_keep = tf.scatter_nd(
                indices=tf.expand_dims(indices, axis=1),
                updates=tf.ones_like(indices, dtype=tf.int32),
                shape=tf.shape(neg_classification_loss_1d)
            )
            negatives_keep = tf.cast(tf.reshape(negatives_keep, [batch_size, num_boxes]), tf.float32)
            neg_class_loss = tf.reduce_sum(classification_loss * negatives_keep, axis=-1)
            return neg_class_loss

        neg_classification_loss = tf.cond(tf.equal(num_neg_classification_loss, tf.constant(0)), f1, f2)
        classification_loss = pos_classification_loss + neg_classification_loss

        total = (classification_loss + self.alpha * pos_regression_loss) / tf.maximum(1.0, tf.cast(num_positives, tf.float32))
        total = total * tf.cast(batch_size, tf.float32)
        return total


def get_loss(config):
    training_config = config["training"]
    return SSD_LOSS(
            alpha=training_config["alpha"],
            min_negative_boxes=training_config["min_negative_boxes"],
            negative_boxes_ratio=training_config["negative_boxes_ratio"]
        )
