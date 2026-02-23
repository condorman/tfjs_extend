import * as tf from '@tensorflow/tfjs';

export function clearMetricsTensorCache(): void;
export function auc(yTrue: tf.Tensor, yPred: tf.Tensor): tf.Scalar;
export function f1(yTrue: tf.Tensor, yPred: tf.Tensor): tf.Scalar;
