import * as tf from '@tensorflow/tfjs';

const EPSILON = 1e-7;
const F1_THRESHOLD = 0.5;
const AUC_NUM_THRESHOLDS = 200;

function toFloatVector(value) {
  if (value instanceof tf.Tensor) {
    return value.cast('float32').reshape([-1]);
  }
  return tf.tensor(value, undefined, 'float32').reshape([-1]);
}

function normalizeInputs(yTrueInput, yPredInput) {
  const yTrue = toFloatVector(yTrueInput);
  const yPred = toFloatVector(yPredInput);

  if (yTrue.size !== yPred.size) {
    yTrue.dispose();
    yPred.dispose();
    throw new Error(
      `yTrue and yPred must have the same length. Received ${yTrue.size} and ${yPred.size}.`
    );
  }

  return { yTrue, yPred };
}

/**
 * AUC as pure function for TensorFlow.js metrics
 * Matches Keras AUC by computing ROC using fixed thresholds.
 * Keras reference: https://github.com/keras-team/keras/blob/v3.13.2/keras/src/metrics/confusion_metrics.py#L1068
 * Uses trapezoidal rule for numerical integration of ROC curve
 */

export function auc(yTrue, yPred) {
  return tf.tidy(() => {
    const normalized = normalizeInputs(yTrue, yPred);
    const labels = normalized.yTrue.notEqual(tf.scalar(0, 'float32')).expandDims(0);
    const predictions = normalized.yPred.expandDims(0);

    const thresholds = new Float32Array(AUC_NUM_THRESHOLDS);
    thresholds[0] = -EPSILON;
    for (let i = 1; i < AUC_NUM_THRESHOLDS - 1; i += 1) {
      thresholds[i] = i / (AUC_NUM_THRESHOLDS - 1);
    }
    thresholds[AUC_NUM_THRESHOLDS - 1] = 1 + EPSILON;

    const thresholdTensor = tf.tensor1d(thresholds, 'float32').expandDims(1);

    const predIsPositive = predictions.greater(thresholdTensor);
    const predIsNegative = predIsPositive.logicalNot();
    const labelIsNegative = labels.logicalNot();

    const tp = predIsPositive.logicalAnd(labels).cast('float32').sum(1);
    const fp = predIsPositive.logicalAnd(labelIsNegative).cast('float32').sum(1);
    const fn = predIsNegative.logicalAnd(labels).cast('float32').sum(1);
    const tn = predIsNegative.logicalAnd(labelIsNegative).cast('float32').sum(1);

    const recall = tf.divNoNan(tp, tp.add(fn));
    const fpRate = tf.divNoNan(fp, fp.add(tn));

    const leftX = fpRate.slice([0], [AUC_NUM_THRESHOLDS - 1]);
    const rightX = fpRate.slice([1], [AUC_NUM_THRESHOLDS - 1]);
    const leftY = recall.slice([0], [AUC_NUM_THRESHOLDS - 1]);
    const rightY = recall.slice([1], [AUC_NUM_THRESHOLDS - 1]);

    const heights = leftY.add(rightY).div(tf.scalar(2, 'float32'));
    const riemannTerms = leftX.sub(rightX).mul(heights);
    return riemannTerms.sum();
  });
}

/**
 * F1Score as pure function for TensorFlow.js metrics
 * Matches Keras F1Score implementation exactly.
 * Keras reference: https://github.com/keras-team/keras/blob/v3.13.2/keras/src/metrics/f_score_metrics.py#L250
 */
export function f1(yTrue, yPred) {
  return tf.tidy(() => {
    const normalized = normalizeInputs(yTrue, yPred);
    const labels = normalized.yTrue;
    const predictions = normalized.yPred
      .greater(tf.scalar(F1_THRESHOLD, 'float32'))
      .cast('float32');

    const ones = tf.onesLike(labels);
    const epsilonScalar = tf.scalar(EPSILON, 'float32');
    const two = tf.scalar(2, 'float32');

    const tp = predictions.mul(labels).sum();
    const fp = predictions.mul(ones.sub(labels)).sum();
    const fn = ones.sub(predictions).mul(labels).sum();

    const precision = tp.div(tp.add(fp).add(epsilonScalar));
    const recall = tp.div(tp.add(fn).add(epsilonScalar));
    return precision
      .mul(recall)
      .div(precision.add(recall).add(epsilonScalar))
      .mul(two);
  });
}
