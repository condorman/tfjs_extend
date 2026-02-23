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
