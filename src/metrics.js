import * as tf from '@tensorflow/tfjs';

const EPSILON = 1e-7;
const F1_THRESHOLD = 0.5;
const AUC_NUM_THRESHOLDS = 200;
const HALF = 0.5;
const AUC_FALLBACK = 0.5;

const AUC_THRESHOLDS = new Float32Array(AUC_NUM_THRESHOLDS);
AUC_THRESHOLDS[0] = -EPSILON;
for (let i = 1; i < AUC_NUM_THRESHOLDS - 1; i += 1) {
  AUC_THRESHOLDS[i] = i / (AUC_NUM_THRESHOLDS - 1);
}
AUC_THRESHOLDS[AUC_NUM_THRESHOLDS - 1] = 1 + EPSILON;

let cachedAucThresholdTensor = null;
let cachedAucThresholdBackend = null;

function getAucThresholdTensor() {
  const backend = tf.getBackend();

  if (cachedAucThresholdTensor && cachedAucThresholdBackend !== backend) {
    cachedAucThresholdTensor.dispose();
    cachedAucThresholdTensor = null;
  }

  if (!cachedAucThresholdTensor) {
    cachedAucThresholdTensor = tf.keep(tf.tensor1d(AUC_THRESHOLDS).expandDims(1));
    cachedAucThresholdBackend = backend;
  }

  return cachedAucThresholdTensor;
}

/**
 * AUC as pure function for TensorFlow.js metrics
 * Matches Keras AUC by computing ROC using fixed thresholds.
 * Keras reference: https://github.com/keras-team/keras/blob/v3.13.2/keras/src/metrics/confusion_metrics.py#L1068
 * Uses trapezoidal rule for numerical integration of ROC curve
 */
export function auc(yTrue, yPred) {
  return tf.tidy(() => {
    const yTrueFlat = yTrue.reshape([-1]);
    const yPredFlat = yPred.reshape([-1]);
    const thresholdTensor = getAucThresholdTensor();

    const labels = yTrueFlat.expandDims(0);
    const predictions = yPredFlat.expandDims(0);

    const totalPos = labels.sum();
    const totalNeg = tf.sub(yTrueFlat.size, totalPos);
    const hasPos = totalPos.greater(0);
    const hasNeg = totalNeg.greater(0);

    const predPos = predictions.greater(thresholdTensor).cast('float32');
    const tp = predPos.mul(labels).sum(1);
    const fp = predPos.sum(1).sub(tp);

    const tpr = tf.divNoNan(tp, totalPos);
    const fpr = tf.divNoNan(fp, totalNeg);

    const fprSliced = fpr.slice([1], [AUC_NUM_THRESHOLDS - 1]);
    const fprPrev = fpr.slice([0], [AUC_NUM_THRESHOLDS - 1]);
    const tprSliced = tpr.slice([1], [AUC_NUM_THRESHOLDS - 1]);
    const tprPrev = tpr.slice([0], [AUC_NUM_THRESHOLDS - 1]);

    const fprDiff = fprPrev.sub(fprSliced);
    const tprSum = tprSliced.add(tprPrev);
    const aucTensor = fprDiff.mul(tprSum).mul(HALF).sum();

    return tf.where(hasPos.logicalAnd(hasNeg), aucTensor, AUC_FALLBACK);
  });
}

/**
 * F1Score as pure function for TensorFlow.js metrics
 * Matches Keras F1Score implementation exactly.
 * Keras reference: https://github.com/keras-team/keras/blob/v3.13.2/keras/src/metrics/f_score_metrics.py#L250
 */
export function f1(yTrue, yPred) {
  return tf.tidy(() => {
    const yTrueFlat = yTrue.reshape([-1]);
    const yPredFlat = yPred.reshape([-1]);

    const yPredThresholded = yPredFlat.greater(F1_THRESHOLD).cast('float32');

    const tp = yPredThresholded.mul(yTrueFlat).sum();
    const predictedPos = yPredThresholded.sum();
    const possiblePos = yTrueFlat.sum();

    const precision = tp.div(predictedPos.add(EPSILON));
    const recall = tp.div(possiblePos.add(EPSILON));
    return precision.mul(recall).mul(2).div(precision.add(recall).add(EPSILON));
  });
}
