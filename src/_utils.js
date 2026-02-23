import * as tf from '@tensorflow/tfjs';

export function isTensorDisposed(tensor) {
  return tensor && tensor.isDisposedInternal === true;
}

export function tensorShapeEquals(left, right) {
  if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) {
    return false;
  }

  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) {
      return false;
    }
  }

  return true;
}

export function asFloatTensor(value, argName) {
  if (value instanceof tf.Tensor) {
    if (value.dtype === 'float32') {
      return { tensor: value, owned: false };
    }
    return { tensor: tf.cast(value, 'float32'), owned: true };
  }

  try {
    return { tensor: tf.tensor(value, undefined, 'float32'), owned: true };
  } catch (error) {
    throw new TypeError(`Argument \`${argName}\` must be a Tensor or tensor-like value.`);
  }
}

export function assertFiniteNumber(value, name) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(`Argument \`${name}\` must be a finite number. Received: ${value}`);
  }
}

export function normalizeGradientsInput(variableGradients) {
  if (Array.isArray(variableGradients)) {
    const map = {};
    for (const entry of variableGradients) {
      map[entry.name] = entry.tensor;
    }
    return { names: variableGradients.map((entry) => entry.name), map };
  }

  return { names: Object.keys(variableGradients), map: variableGradients };
}

export function getRegisteredVariable(name) {
  const variable = tf.engine().registeredVariables[name];
  if (variable == null) {
    throw new Error(`No registered variable found for gradient key \`${name}\`.`);
  }
  return variable;
}

export function createSlotVariable(referenceVariable) {
  return tf.tidy(() => tf.zerosLike(referenceVariable).variable(false));
}

export const ATTENTION_NEGATIVE_INF = -1e9;

const CLASS_KEY_ALIASES = Object.freeze(['className', 'class_name']);

const INITIALIZER_NAME_ALIASES = Object.freeze({
  glorot_uniform: 'glorotUniform',
  glorot_normal: 'glorotNormal',
  he_uniform: 'heUniform',
  he_normal: 'heNormal',
  lecun_uniform: 'leCunUniform',
  lecun_normal: 'leCunNormal',
  random_uniform: 'randomUniform',
  random_normal: 'randomNormal',
  truncated_normal: 'truncatedNormal',
  variance_scaling: 'varianceScaling',
});

const REGULARIZER_NAME_ALIASES = Object.freeze({
  l1_l2: 'l1l2',
});

const CONSTRAINT_NAME_ALIASES = Object.freeze({
  max_norm: 'maxNorm',
  min_max_norm: 'minMaxNorm',
  non_neg: 'nonNeg',
  unit_norm: 'unitNorm',
});

const CLASS_NAME_ALIASES = Object.freeze({
  variance_scaling: 'VarianceScaling',
  glorot_uniform: 'VarianceScaling',
  glorot_normal: 'VarianceScaling',
  zeros: 'Zeros',
  ones: 'Ones',
  random_normal: 'RandomNormal',
  random_uniform: 'RandomUniform',
  truncated_normal: 'TruncatedNormal',
  he_uniform: 'VarianceScaling',
  he_normal: 'VarianceScaling',
  lecun_uniform: 'VarianceScaling',
  lecun_normal: 'VarianceScaling',
  l1l2: 'L1L2',
  l1_l2: 'L1L2',
  l1: 'L1L2',
  l2: 'L1L2',
  max_norm: 'MaxNorm',
  min_max_norm: 'MinMaxNorm',
  non_neg: 'NonNeg',
  unit_norm: 'UnitNorm',
});

const CONFIG_KEY_ALIASES = Object.freeze({
  max_value: 'maxValue',
  min_value: 'minValue',
  rate: 'rate',
});

function normalizeIdentifierName(name, aliases) {
  if (typeof name !== 'string') {
    return name;
  }
  return aliases[name] ?? name;
}

function normalizeClassName(name) {
  if (typeof name !== 'string') {
    return name;
  }

  const lower = name.toLowerCase();
  const mapped = CLASS_NAME_ALIASES[lower];
  if (mapped != null) {
    return mapped;
  }

  if (name.includes('_')) {
    return name
      .split('_')
      .filter((part) => part.length > 0)
      .map((part) => part[0].toUpperCase() + part.slice(1))
      .join('');
  }

  return name;
}

function normalizeSerializedConfig(value) {
  if (value == null || typeof value !== 'object' || Array.isArray(value)) {
    return value;
  }

  const classKey = CLASS_KEY_ALIASES.find((key) => Object.prototype.hasOwnProperty.call(value, key));
  if (classKey == null) {
    return value;
  }

  const config = value.config != null && typeof value.config === 'object'
    ? value.config
    : {};

  const normalizedConfig = {};
  for (const [key, item] of Object.entries(config)) {
    const normalizedKey = CONFIG_KEY_ALIASES[key] ?? key;
    normalizedConfig[normalizedKey] = item;
  }

  return {
    className: normalizeClassName(value[classKey]),
    config: normalizedConfig,
  };
}

function normalizeKerasIdentifier(value, aliases) {
  if (typeof value === 'string') {
    return normalizeIdentifierName(value, aliases);
  }
  if (value == null) {
    return value;
  }
  if (typeof value === 'object') {
    return normalizeSerializedConfig(value);
  }
  return value;
}

export function normalizeKerasInitializerIdentifier(value) {
  return normalizeKerasIdentifier(value, INITIALIZER_NAME_ALIASES);
}

export function normalizeKerasConstraintIdentifier(value) {
  return normalizeKerasIdentifier(value, CONSTRAINT_NAME_ALIASES);
}

export function normalizeKerasRegularizerIdentifier(value) {
  if (typeof value === 'string') {
    const normalized = normalizeIdentifierName(value, REGULARIZER_NAME_ALIASES);
    if (normalized === 'l1') {
      return { className: 'L1L2', config: { l1: 0.01, l2: 0 } };
    }
    if (normalized === 'l2') {
      return { className: 'L1L2', config: { l1: 0, l2: 0.01 } };
    }
    if (normalized === 'l1l2') {
      return { className: 'L1L2', config: { l1: 0.01, l2: 0.01 } };
    }
    return normalized;
  }

  if (value != null && typeof value === 'object') {
    return normalizeSerializedConfig(value);
  }

  return value;
}

export function isShapeTuple(value) {
  return Array.isArray(value) && value.every((item) => Number.isInteger(item) || item == null);
}

export function toShapeTuple(value, fieldName) {
  if (value == null) {
    return null;
  }
  if (Number.isInteger(value)) {
    return [value];
  }
  if (!Array.isArray(value)) {
    throw new TypeError(`Argument \`${fieldName}\` must be an int, array or null.`);
  }
  const normalized = [...value];
  for (const dim of normalized) {
    if (!Number.isInteger(dim)) {
      throw new TypeError(`Argument \`${fieldName}\` contains non-integer dimension: ${dim}`);
    }
  }
  return normalized;
}

export function assertPositiveInteger(value, fieldName) {
  if (!Number.isInteger(value) || value <= 0) {
    throw new TypeError(`Argument \`${fieldName}\` must be a positive integer. Received: ${value}`);
  }
}

export function assertNonNegativeNumber(value, fieldName) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    throw new TypeError(`Argument \`${fieldName}\` must be a finite number >= 0. Received: ${value}`);
  }
}

export function assertBoolean(value, fieldName) {
  if (typeof value !== 'boolean') {
    throw new TypeError(`Argument \`${fieldName}\` must be a boolean. Received: ${value}`);
  }
}

export function serializeKerasObject(value) {
  if (value == null) {
    return null;
  }
  return {
    className: value.getClassName(),
    config: value.getConfig(),
  };
}

export function normalizeKwargBoolean(kwargs, snakeCaseKey, camelCaseKey, defaultValue) {
  if (kwargs == null) {
    return defaultValue;
  }

  if (kwargs[snakeCaseKey] != null) {
    return Boolean(kwargs[snakeCaseKey]);
  }
  if (kwargs[camelCaseKey] != null) {
    return Boolean(kwargs[camelCaseKey]);
  }
  return defaultValue;
}

export function getKwargValue(kwargs, snakeCaseKey, camelCaseKey, defaultValue = null) {
  if (kwargs == null) {
    return defaultValue;
  }
  if (kwargs[snakeCaseKey] !== undefined) {
    return kwargs[snakeCaseKey];
  }
  if (kwargs[camelCaseKey] !== undefined) {
    return kwargs[camelCaseKey];
  }
  return defaultValue;
}

function indexToEinsumVariable(index) {
  return 'abcdefghijklmnopqrstuvwxyz'[index];
}

export function buildAttentionEquation(rank, attentionAxes) {
  let targetNotation = '';
  for (let index = 0; index < rank; index += 1) {
    targetNotation += indexToEinsumVariable(index);
  }

  const excluded = new Set([...attentionAxes, rank - 1]);
  const batchDims = [];
  for (let index = 0; index < rank; index += 1) {
    if (!excluded.has(index)) {
      batchDims.push(index);
    }
  }

  let letterOffset = rank;
  let sourceNotation = '';
  for (let index = 0; index < rank; index += 1) {
    if (batchDims.includes(index) || index === rank - 1) {
      sourceNotation += targetNotation[index];
    } else {
      sourceNotation += indexToEinsumVariable(letterOffset);
      letterOffset += 1;
    }
  }

  const productNotation = [
    ...batchDims.map((index) => targetNotation[index]),
    ...attentionAxes.map((index) => targetNotation[index]),
    ...attentionAxes.map((index) => sourceNotation[index]),
  ].join('');

  const dotProductEquation = `${sourceNotation},${targetNotation}->${productNotation}`;
  const combineEquation = `${productNotation},${sourceNotation}->${targetNotation}`;
  return {
    dotProductEquation,
    combineEquation,
    attentionScoresRank: productNotation.length,
    batchDims,
  };
}

export function buildProjectionEquation(freeDims, boundDims, outputDims) {
  let inputStr = '';
  let kernelStr = '';
  let outputStr = '';
  let biasAxes = '';
  let letterOffset = 0;

  for (let index = 0; index < freeDims; index += 1) {
    const char = indexToEinsumVariable(index + letterOffset);
    inputStr += char;
    outputStr += char;
  }

  letterOffset += freeDims;
  for (let index = 0; index < boundDims; index += 1) {
    const char = indexToEinsumVariable(index + letterOffset);
    inputStr += char;
    kernelStr += char;
  }

  letterOffset += boundDims;
  for (let index = 0; index < outputDims; index += 1) {
    const char = indexToEinsumVariable(index + letterOffset);
    kernelStr += char;
    outputStr += char;
    biasAxes += char;
  }

  return {
    equation: `${inputStr},${kernelStr}->${outputStr}`,
    biasAxes,
    outputRank: outputStr.length,
  };
}

export function resolveLayerConfigArgs(options) {
  const layerArgs = {};
  for (const key of [
    'name',
    'trainable',
    'dtype',
    'batchInputShape',
    'batchSize',
    'weights',
    'inputShape',
  ]) {
    if (Object.prototype.hasOwnProperty.call(options, key)) {
      layerArgs[key] = options[key];
    }
  }
  return layerArgs;
}

export function resolveMaskFromTensor(tensor) {
  if (tensor != null && Object.prototype.hasOwnProperty.call(tensor, 'kerasMask')) {
    return tensor.kerasMask;
  }
  return null;
}

export function softmaxAlongLastDims(values, dims) {
  if (dims <= 0) {
    return values;
  }
  const rank = values.shape.length;
  const axes = [];
  for (let index = rank - dims; index < rank; index += 1) {
    axes.push(index);
  }

  const maxValues = tf.max(values, axes, true);
  const stabilized = tf.sub(values, maxValues);
  const expValues = tf.exp(stabilized);
  const summed = tf.sum(expValues, axes, true);
  return tf.div(expValues, tf.add(summed, 1e-9));
}

export function productOfShape(shape) {
  return shape.reduce((acc, item) => acc * item, 1);
}
