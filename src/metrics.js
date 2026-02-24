import * as tf from '@tensorflow/tfjs';

/**
 * AUC as pure function for TensorFlow.js metrics
 * Matches Keras AUC by computing ROC using fixed thresholds.
 * Keras reference: https://github.com/keras-team/keras/blob/v3.13.2/keras/src/metrics/confusion_metrics.py#L1068
 * Uses trapezoidal rule for numerical integration of ROC curve
 */
const numThresholds = 200;
const maxThresholdIndex = numThresholds - 1;
const numThresholdBuckets = numThresholds + 1;
const threshold = 0.5;
const epsilon = 1e-7; // Matches Keras backend.epsilon()

let cachedBackend = null;
let cachedThresholds = null;
let cachedZero = null;
let cachedHalf = null;
let cachedOne = null;
let cachedTwo = null;
let cachedThreshold = null;
let cachedEpsilon = null;

function isDisposed(tensor) {
  return !tensor || tensor.isDisposedInternal === true;
}

function disposeCachedTensors() {
  [cachedThresholds, cachedZero, cachedHalf, cachedOne, cachedTwo, cachedThreshold, cachedEpsilon].forEach((tensor) => {
    if (!isDisposed(tensor)) {
      tensor.dispose();
    }
  });

  cachedThresholds = null;
  cachedZero = null;
  cachedHalf = null;
  cachedOne = null;
  cachedTwo = null;
  cachedThreshold = null;
  cachedEpsilon = null;
}

export function clearMetricsTensorCache() {
  disposeCachedTensors();
  cachedBackend = null;
}

function getCachedTensors() {
  const backend = tf.getBackend();
  const cacheInvalid =
    backend !== cachedBackend ||
    isDisposed(cachedThresholds) ||
    isDisposed(cachedZero) ||
    isDisposed(cachedHalf) ||
    isDisposed(cachedOne) ||
    isDisposed(cachedTwo) ||
    isDisposed(cachedThreshold) ||
    isDisposed(cachedEpsilon);

  if (cacheInvalid) {
    disposeCachedTensors();

    cachedThresholds = tf.linspace(1, 0, numThresholds);
    cachedZero = tf.scalar(0);
    cachedHalf = tf.scalar(0.5);
    cachedOne = tf.scalar(1);
    cachedTwo = tf.scalar(2);
    cachedThreshold = tf.scalar(threshold);
    cachedEpsilon = tf.scalar(epsilon);
    cachedBackend = backend;
  }

  return {
    thresholds: cachedThresholds,
    zero: cachedZero,
    half: cachedHalf,
    one: cachedOne,
    two: cachedTwo,
    threshold: cachedThreshold,
    epsilon: cachedEpsilon,
  };
}

export function auc(yTrue, yPred) {
  const c = getCachedTensors();

  return tf.tidy(() => {
    const yTrueFlat = tf.cast(tf.reshape(yTrue, [-1]), 'float32');
    const yPredFlat = tf.cast(tf.reshape(yPred, [-1]), 'float32');

    const totalPos = tf.sum(yTrueFlat);
    const totalNeg = tf.sub(tf.scalar(yTrueFlat.size), totalPos);

    const hasPos = tf.greater(totalPos, c.zero);
    const hasNeg = tf.greater(totalNeg, c.zero);

    // Fast path: convert threshold comparisons into cumulative histogram counts.
    const floorIndex = tf.floor(tf.mul(yPredFlat, maxThresholdIndex));
    const startIndex = tf.sub(maxThresholdIndex, floorIndex);
    const clippedStart = tf.cast(tf.clipByValue(startIndex, 0, numThresholds), 'int32');

    const positiveStarts = tf.bincount(clippedStart, yTrueFlat, numThresholdBuckets);
    const negativeStarts = tf.bincount(clippedStart, tf.sub(c.one, yTrueFlat), numThresholdBuckets);
    const tp = tf.cumsum(positiveStarts.slice([0], [numThresholds]));
    const fp = tf.cumsum(negativeStarts.slice([0], [numThresholds]));

    const tpr = tf.divNoNan(tp, totalPos);
    const fpr = tf.divNoNan(fp, totalNeg);

    // Calculate AUC using trapezoidal rule: sum of (dx * (y1 + y2) / 2)
    const fprSliced = fpr.slice([1], [numThresholds - 1]);
    const fprPrev = fpr.slice([0], [numThresholds - 1]);
    const tprSliced = tpr.slice([1], [numThresholds - 1]);
    const tprPrev = tpr.slice([0], [numThresholds - 1]);

    const fprDiff = tf.sub(fprSliced, fprPrev);
    const tprSum = tf.add(tprSliced, tprPrev);
    const aucTensor = tf.sum(tf.mul(fprDiff, tf.mul(c.half, tprSum)));

    const safeAuc = tf.where(tf.logicalAnd(hasPos, hasNeg), aucTensor, c.half);
    return tf.clipByValue(safeAuc, 0, 1);
  });
}

/**
 * F1Score as pure function for TensorFlow.js metrics
 * Matches Keras F1Score implementation exactly.
 * Keras reference: https://github.com/keras-team/keras/blob/v3.13.2/keras/src/metrics/f_score_metrics.py#L250
 */
export function f1(yTrue, yPred) {
  const c = getCachedTensors();

  return tf.tidy(() => {
    const yTrueFlat = tf.cast(tf.reshape(yTrue, [-1]), 'float32');
    const yPredFlat = tf.cast(tf.reshape(yPred, [-1]), 'float32');

    // Apply threshold to predictions
    const yPredThresholded = tf.cast(tf.greaterEqual(yPredFlat, c.threshold), 'float32');

    // Aggregate counts first to reduce elementwise ops.
    const tp = tf.sum(tf.mul(yTrueFlat, yPredThresholded));
    const predPos = tf.sum(yPredThresholded);
    const truePos = tf.sum(yTrueFlat);
    const fp = tf.sub(predPos, tp);
    const fn = tf.sub(truePos, tp);

    // Calculate precision and recall with epsilon (Keras approach)
    const precision = tf.div(tp, tf.add(tf.add(tp, fp), c.epsilon));
    const recall = tf.div(tp, tf.add(tf.add(tp, fn), c.epsilon));

    // Calculate F1 score: 2 * (precision * recall) / (precision + recall + epsilon)
    const mulValue = tf.mul(precision, recall);
    const addValue = tf.add(precision, recall);
    return tf.div(
      tf.mul(c.two, mulValue),
      tf.add(addValue, c.epsilon)
    );
  });
}
