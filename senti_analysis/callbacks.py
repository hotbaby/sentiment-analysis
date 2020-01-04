# encoding:utf8

import tensorflow as tf


class CustomTensorBoard(tf.keras.callbacks.TensorBoard):
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        if self.write_batch_performance == True:
            for name, value in logs.items():
                if name in ['batch', 'size']:
                    continue

                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.writer.add_summary(summary, self.seen)
            self.writer.flush()

        self.seen += self.batch_size
