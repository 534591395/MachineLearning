# 参考：https://blog.csdn.net/u011987514/article/details/71189491 
# github: https://github.com/cwellszhang/DetectMaliciousURL
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import data

def main(argv):
    (train_x, train_y), (test_x, test_y) = data.load_data()
    #feature_column = tf.feature_column.categorical_column_with_hash_bucket("url", 100)
    feature_column = tf.feature_column.categorical_column_with_vocabulary_list("url", ["url"])
    # feature_column = []
    # for key in train_x.keys():
    #     feature_column.append(tf.feature_column.numeric_column(key=key))  
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_column,
        hidden_units=[10, 10],
        n_classes=2)
    classifier.train(
        input_fn=lambda:data.train_input_fn(train_x, train_y,
                                                 100),
        steps=1000)
    eval_result = classifier.evaluate(
        input_fn=lambda:data.eval_input_fn(test_x, test_y,
                                                100))

    print('\n打印结果: {eval_result:0.3f}\n'.format(**eval_result))
if __name__ == '__main__':
   #tf.logging.set_verbosity(tf.logging.INFO)
   tf.app.run(main)  