import * as tf from '@tensorflow/tfjs';
import {
  ATTENTION_NEGATIVE_INF,
  assertBoolean,
  assertNonNegativeNumber,
  assertPositiveInteger,
  buildAttentionEquation,
  buildProjectionEquation,
  getKwargValue,
  isShapeTuple,
  normalizeKerasConstraintIdentifier,
  normalizeKerasInitializerIdentifier,
  normalizeKerasRegularizerIdentifier,
  normalizeKwargBoolean,
  productOfShape,
  resolveLayerConfigArgs,
  resolveMaskFromTensor,
  serializeKerasObject,
  softmaxAlongLastDims,
  tensorShapeEquals,
  toShapeTuple,
} from './_utils.js';

export class MultiHeadAttention extends tf.layers.Layer {
  static className = 'MultiHeadAttention';

  static DEFAULTS = Object.freeze({
    num_heads: null,
    key_dim: null,
    value_dim: null,
    dropout: 0,
    use_bias: true,
    output_shape: null,
    attention_axes: null,
    flash_attention: null,
    kernel_initializer: 'glorot_uniform',
    bias_initializer: 'zeros',
    kernel_regularizer: null,
    bias_regularizer: null,
    activity_regularizer: null,
    kernel_constraint: null,
    bias_constraint: null,
    seed: null,
  });

  static CONFIG_ALIASES = Object.freeze({
    numHeads: 'num_heads',
    keyDim: 'key_dim',
    valueDim: 'value_dim',
    useBias: 'use_bias',
    outputShape: 'output_shape',
    attentionAxes: 'attention_axes',
    flashAttention: 'flash_attention',
    kernelInitializer: 'kernel_initializer',
    biasInitializer: 'bias_initializer',
    kernelRegularizer: 'kernel_regularizer',
    biasRegularizer: 'bias_regularizer',
    activityRegularizer: 'activity_regularizer',
    kernelConstraint: 'kernel_constraint',
    biasConstraint: 'bias_constraint',
  });

  static ALLOWED_CONFIG_KEYS = null;

  static {
    this.ALLOWED_CONFIG_KEYS = new Set([
      ...Object.keys(this.DEFAULTS),
      ...Object.keys(this.CONFIG_ALIASES),
      'name',
      'trainable',
      'dtype',
      'batchInputShape',
      'batchSize',
      'weights',
      'inputShape',
    ]);
    tf.serialization.registerClass(this);
  }

  static _canonicalizeConfig(config = {}) {
    const canonical = {};

    for (const [key, value] of Object.entries(config)) {
      if (!this.ALLOWED_CONFIG_KEYS.has(key)) {
        throw new Error(`Argument \`${key}\` is not recognized.`);
      }
      const canonicalKey = this.CONFIG_ALIASES[key] ?? key;
      canonical[canonicalKey] = value;
    }

    return canonical;
  }

  static _resolveDenseObjects(config) {
    const dense = tf.layers.dense({
      units: 1,
      kernelInitializer: config.kernel_initializer,
      biasInitializer: config.bias_initializer,
      kernelRegularizer: config.kernel_regularizer,
      biasRegularizer: config.bias_regularizer,
      activityRegularizer: config.activity_regularizer,
      kernelConstraint: config.kernel_constraint,
      biasConstraint: config.bias_constraint,
    });

    return {
      kernelInitializer: dense.kernelInitializer,
      biasInitializer: dense.biasInitializer,
      kernelRegularizer: dense.kernelRegularizer,
      biasRegularizer: dense.biasRegularizer,
      activityRegularizer: dense.activityRegularizer,
      kernelConstraint: dense.kernelConstraint,
      biasConstraint: dense.biasConstraint,
    };
  }

  static normalizeConfig(config = {}) {
    const canonical = this._canonicalizeConfig(config);
    const normalized = {
      ...this.DEFAULTS,
      ...canonical,
    };

    normalized.kernel_initializer = normalizeKerasInitializerIdentifier(normalized.kernel_initializer);
    normalized.bias_initializer = normalizeKerasInitializerIdentifier(normalized.bias_initializer);
    normalized.kernel_regularizer = normalizeKerasRegularizerIdentifier(normalized.kernel_regularizer);
    normalized.bias_regularizer = normalizeKerasRegularizerIdentifier(normalized.bias_regularizer);
    normalized.activity_regularizer = normalizeKerasRegularizerIdentifier(normalized.activity_regularizer);
    normalized.kernel_constraint = normalizeKerasConstraintIdentifier(normalized.kernel_constraint);
    normalized.bias_constraint = normalizeKerasConstraintIdentifier(normalized.bias_constraint);

    assertPositiveInteger(normalized.num_heads, 'num_heads');
    assertPositiveInteger(normalized.key_dim, 'key_dim');

    if (normalized.value_dim != null) {
      assertPositiveInteger(normalized.value_dim, 'value_dim');
    }

    assertNonNegativeNumber(normalized.dropout, 'dropout');
    assertBoolean(normalized.use_bias, 'use_bias');

    if (normalized.flash_attention != null) {
      assertBoolean(normalized.flash_attention, 'flash_attention');
      if (normalized.flash_attention) {
        throw new Error('`flash_attention=true` is not supported in TensorFlow.js.');
      }
    }

    if (normalized.attention_axes != null) {
      if (Number.isInteger(normalized.attention_axes)) {
        normalized.attention_axes = [normalized.attention_axes];
      } else if (!Array.isArray(normalized.attention_axes)) {
        throw new TypeError(
          '`attention_axes` must be an int, list, or tuple. '
          + `Received: ${normalized.attention_axes}`
        );
      } else {
        normalized.attention_axes = [...normalized.attention_axes];
      }

      for (const axis of normalized.attention_axes) {
        if (!Number.isInteger(axis)) {
          throw new TypeError(`\`attention_axes\` contains non-integer axis: ${axis}`);
        }
      }
    }

    normalized.output_shape = toShapeTuple(normalized.output_shape, 'output_shape');

    if (normalized.seed != null) {
      if (!Number.isInteger(normalized.seed)) {
        throw new TypeError(`Argument \`seed\` must be an integer or null. Received: ${normalized.seed}`);
      }
    }

    return normalized;
  }

  constructor(options = {}) {
    const normalized = MultiHeadAttention.normalizeConfig(options);
    const layerArgs = resolveLayerConfigArgs(options);

    super(layerArgs);

    this.supportsMasking = true;

    this.config = normalized;
    this._numHeads = normalized.num_heads;
    this._keyDim = normalized.key_dim;
    this._valueDim = normalized.value_dim ?? normalized.key_dim;
    this._dropout = normalized.dropout;
    this._useBias = normalized.use_bias;
    this._outputShape = normalized.output_shape;
    this._attentionAxesConfig = normalized.attention_axes;
    this._flashAttention = normalized.flash_attention;
    this.seed = normalized.seed;

    const resolvedDenseObjects = this.constructor._resolveDenseObjects(normalized);
    this._kernelInitializer = resolvedDenseObjects.kernelInitializer;
    this._biasInitializer = resolvedDenseObjects.biasInitializer;
    this._kernelRegularizer = resolvedDenseObjects.kernelRegularizer;
    this._biasRegularizer = resolvedDenseObjects.biasRegularizer;
    this._activityRegularizer = resolvedDenseObjects.activityRegularizer;
    this._kernelConstraint = resolvedDenseObjects.kernelConstraint;
    this._biasConstraint = resolvedDenseObjects.biasConstraint;

    this._inverseSqrtKeyDim = 1 / Math.sqrt(Number(this._keyDim));

    this._queryKernel = null;
    this._queryBias = null;
    this._keyKernel = null;
    this._keyBias = null;
    this._valueKernel = null;
    this._valueBias = null;
    this._outputKernel = null;
    this._outputBias = null;

    this._queryProjectionEquation = null;
    this._keyProjectionEquation = null;
    this._valueProjectionEquation = null;
    this._outputProjectionEquation = null;

    this._dotProductEquation = null;
    this._combineEquation = null;
    this._attentionAxes = null;
    this._attentionScoresRank = null;

    this._dropoutLayer = tf.layers.dropout({
      rate: this._dropout,
      seed: this.seed,
    });

    this._returnAttentionScoresForShapeComputation = false;
  }

  get numHeads() {
    return this._numHeads;
  }

  get keyDim() {
    return this._keyDim;
  }

  get valueDim() {
    return this._valueDim;
  }

  get dropout() {
    return this._dropout;
  }

  get useBias() {
    return this._useBias;
  }

  get attentionAxes() {
    return this._attentionAxesConfig;
  }

  _normalizeAttentionAxes(rank) {
    if (this._attentionAxesConfig == null) {
      const axes = [];
      for (let axis = 1; axis < rank - 2; axis += 1) {
        axes.push(axis);
      }
      return axes;
    }

    return this._attentionAxesConfig.map((axis) => (axis >= 0 ? axis : (rank - 1) + axis));
  }

  _validateInputShapes(queryShape, valueShape, keyShape) {
    if (!isShapeTuple(queryShape) || !isShapeTuple(valueShape) || !isShapeTuple(keyShape)) {
      throw new TypeError('Input shapes must be tuple-like arrays.');
    }

    if (queryShape.length < 3 || valueShape.length < 3 || keyShape.length < 3) {
      throw new Error('`query`, `value` and `key` tensors must have rank >= 3.');
    }

    if (!tensorShapeEquals(valueShape.slice(1, -1), keyShape.slice(1, -1))) {
      throw new Error(
        'All dimensions of `value` and `key`, except the last one, must be equal. '
        + `Received: value_shape=${JSON.stringify(valueShape)} and key_shape=${JSON.stringify(keyShape)}`
      );
    }
  }

  _extractBuildShapes(inputShape) {
    if (!Array.isArray(inputShape) || inputShape.length < 2 || inputShape.length > 3) {
      throw new Error('MultiHeadAttention expects inputShape as [queryShape, valueShape] or [queryShape, valueShape, keyShape].');
    }

    const queryShape = inputShape[0];
    const valueShape = inputShape[1];
    const keyShape = inputShape[2] ?? valueShape;

    this._validateInputShapes(queryShape, valueShape, keyShape);
    return { queryShape, valueShape, keyShape };
  }

  _addProjectionWeights(name, inputDim, outputDims, biasShape) {
    // Keep Keras forward parity for externally set weights in golden tests:
    // constraints are stored in config but not enforced on assignment here.
    const kernel = this.addWeight(
      `${name}_kernel`,
      [inputDim, ...outputDims],
      'float32',
      this._kernelInitializer,
      this._kernelRegularizer,
      true,
      null
    );

    const bias = this._useBias
      ? this.addWeight(
        `${name}_bias`,
        biasShape,
        'float32',
        this._biasInitializer,
        this._biasRegularizer,
        true,
        null
      )
      : null;

    return { kernel, bias };
  }

  _project(tensor, equation, kernel, bias = null) {
    let output = tf.einsum(equation, tensor, kernel.read());
    if (bias != null) {
      output = tf.add(output, bias.read());
    }
    return output;
  }

  _projectRank3(tensor, kernel, bias, outputDims) {
    const batch = tensor.shape[0];
    const time = tensor.shape[1];
    const inputDim = tensor.shape[2];
    const outputSize = productOfShape(outputDims);

    const flatInputs = tf.reshape(tensor, [-1, inputDim]);
    const flatKernel = tf.reshape(kernel.read(), [inputDim, outputSize]);
    let flatOutput = tf.matMul(flatInputs, flatKernel);
    let output = tf.reshape(flatOutput, [batch, time, ...outputDims]);

    if (bias != null) {
      output = tf.add(output, bias.read());
    }

    return output;
  }

  _projectOutputRank3(tensor, outputDims) {
    const batch = tensor.shape[0];
    const time = tensor.shape[1];
    const outputSize = productOfShape(outputDims);

    const flatInputs = tf.reshape(tensor, [-1, this._numHeads * this._valueDim]);
    const flatKernel = tf.reshape(this._outputKernel.read(), [this._numHeads * this._valueDim, outputSize]);

    let flatOutput = tf.matMul(flatInputs, flatKernel);
    let output = tf.reshape(flatOutput, [batch, time, ...outputDims]);

    if (this._outputBias != null) {
      output = tf.add(output, this._outputBias.read());
    }

    return output;
  }

  _canUseRank3Path(query, key, value) {
    return (
      query.shape.length === 3
      && key.shape.length === 3
      && value.shape.length === 3
      && this._attentionAxes.length === 1
      && this._attentionAxes[0] === 1
    );
  }

  _computeCausalMask(query, value = null) {
    const qSeqLength = query.shape[1];
    const vSeqLength = value == null ? qSeqLength : value.shape[1];

    if (qSeqLength == null || vSeqLength == null) {
      throw new Error('Dynamic sequence lengths are not supported for causal mask computation in this implementation.');
    }

    const row = tf.expandDims(tf.range(0, qSeqLength, 1, 'int32'), 1);
    const col = tf.expandDims(tf.range(0, vSeqLength, 1, 'int32'), 0);
    const mask = tf.greaterEqual(row, col);
    return tf.expandDims(mask, 0);
  }

  _computeAttentionMask(
    query,
    value,
    queryMask = null,
    valueMask = null,
    keyMask = null,
    attentionMask = null,
    useCausalMask = false
  ) {
    let autoMask = null;

    if (queryMask != null) {
      const castMask = tf.cast(queryMask, 'bool');
      autoMask = tf.expandDims(castMask, -1);
    }

    if (valueMask != null) {
      const castMask = tf.cast(valueMask, 'bool');
      const mask = tf.expandDims(castMask, -2);
      autoMask = autoMask == null ? mask : tf.logicalAnd(autoMask, mask);
    }

    if (keyMask != null) {
      const castMask = tf.cast(keyMask, 'bool');
      const mask = tf.expandDims(castMask, -2);
      autoMask = autoMask == null ? mask : tf.logicalAnd(autoMask, mask);
    }

    if (useCausalMask) {
      const mask = this._computeCausalMask(query, value);
      autoMask = autoMask == null ? mask : tf.logicalAnd(autoMask, mask);
    }

    let computedAttentionMask = attentionMask;
    if (computedAttentionMask != null) {
      computedAttentionMask = tf.cast(computedAttentionMask, 'bool');
    }

    if (autoMask != null) {
      computedAttentionMask = computedAttentionMask == null
        ? autoMask
        : tf.logicalAnd(computedAttentionMask, autoMask);
    }

    return computedAttentionMask;
  }

  _maskedSoftmax(attentionScores, attentionMask = null) {
    let scores = attentionScores;

    if (attentionMask != null) {
      let mask = attentionMask;
      const maskExpansionAxis = -this._attentionAxes.length * 2 - 1;
      for (let index = 0; index < (scores.shape.length - mask.shape.length); index += 1) {
        mask = tf.expandDims(mask, maskExpansionAxis);
      }
      const boolMask = tf.cast(mask, 'bool');
      const adder = tf.mul(tf.cast(tf.logicalNot(boolMask), scores.dtype), ATTENTION_NEGATIVE_INF);
      scores = tf.add(scores, adder);
    }

    return softmaxAlongLastDims(scores, this._attentionAxes.length);
  }

  _computeAttention(query, key, value, attentionMask = null, training = null, returnAttentionScores = false) {
    const queryScaled = tf.mul(query, this._inverseSqrtKeyDim);
    const attentionScores = tf.einsum(this._dotProductEquation, key, queryScaled);
    const normalizedScores = this._maskedSoftmax(attentionScores, attentionMask);

    const finalAttentionScores = this._dropout > 0
      ? this._dropoutLayer.apply(normalizedScores, { training })
      : normalizedScores;

    const attentionOutput = tf.einsum(this._combineEquation, finalAttentionScores, value);

    return {
      attentionOutput,
      attentionScores: returnAttentionScores ? normalizedScores : null,
    };
  }

  _computeAttentionRank3(query, key, value, attentionMask = null, training = null, returnAttentionScores = false) {
    const queryScaled = tf.mul(query, this._inverseSqrtKeyDim);

    const queryHeads = tf.transpose(queryScaled, [0, 2, 1, 3]);
    const keyHeads = tf.transpose(key, [0, 2, 1, 3]);
    const valueHeads = tf.transpose(value, [0, 2, 1, 3]);

    const attentionScores = tf.matMul(queryHeads, keyHeads, false, true);
    const normalizedScores = this._maskedSoftmax(attentionScores, attentionMask);

    const finalAttentionScores = this._dropout > 0
      ? this._dropoutLayer.apply(normalizedScores, { training })
      : normalizedScores;

    const attentionOutputHeads = tf.matMul(finalAttentionScores, valueHeads);
    const attentionOutput = tf.transpose(attentionOutputHeads, [0, 2, 1, 3]);

    return {
      attentionOutput,
      attentionScores: returnAttentionScores ? normalizedScores : null,
    };
  }

  build(inputShape) {
    const { queryShape, valueShape, keyShape } = this._extractBuildShapes(inputShape);

    const queryRank = queryShape.length;
    const valueRank = valueShape.length;
    const keyRank = keyShape.length;

    const queryProjection = buildProjectionEquation(queryRank - 1, 1, 2);
    const keyProjection = buildProjectionEquation(keyRank - 1, 1, 2);
    const valueProjection = buildProjectionEquation(valueRank - 1, 1, 2);

    this._queryProjectionEquation = queryProjection.equation;
    this._keyProjectionEquation = keyProjection.equation;
    this._valueProjectionEquation = valueProjection.equation;

    const queryInputDim = queryShape[queryShape.length - 1];
    const keyInputDim = keyShape[keyShape.length - 1];
    const valueInputDim = valueShape[valueShape.length - 1];

    if (queryInputDim == null || keyInputDim == null || valueInputDim == null) {
      throw new Error('Last dimension of query/key/value must be defined at build time.');
    }

    const queryWeights = this._addProjectionWeights(
      'query',
      queryInputDim,
      [this._numHeads, this._keyDim],
      [this._numHeads, this._keyDim]
    );
    this._queryKernel = queryWeights.kernel;
    this._queryBias = queryWeights.bias;

    const keyWeights = this._addProjectionWeights(
      'key',
      keyInputDim,
      [this._numHeads, this._keyDim],
      [this._numHeads, this._keyDim]
    );
    this._keyKernel = keyWeights.kernel;
    this._keyBias = keyWeights.bias;

    const valueWeights = this._addProjectionWeights(
      'value',
      valueInputDim,
      [this._numHeads, this._valueDim],
      [this._numHeads, this._valueDim]
    );
    this._valueKernel = valueWeights.kernel;
    this._valueBias = valueWeights.bias;

    const attentionRank = queryProjection.outputRank;
    this._attentionAxes = this._normalizeAttentionAxes(attentionRank);

    const attentionEq = buildAttentionEquation(attentionRank, this._attentionAxes);
    this._dotProductEquation = attentionEq.dotProductEquation;
    this._combineEquation = attentionEq.combineEquation;
    this._attentionScoresRank = attentionEq.attentionScoresRank;

    const outputShape = this._outputShape ?? [queryShape[queryShape.length - 1]];
    const outputProjection = buildProjectionEquation(queryRank - 1, 2, outputShape.length);
    this._outputProjectionEquation = outputProjection.equation;

    const outputWeights = this._addProjectionWeights(
      'output',
      this._numHeads,
      [this._valueDim, ...outputShape],
      [...outputShape]
    );

    this._outputKernel = outputWeights.kernel;
    this._outputBias = outputWeights.bias;

    this.built = true;
  }

  _extractInputTensors(inputs) {
    if (!Array.isArray(inputs) || inputs.length < 2 || inputs.length > 3) {
      throw new Error('MultiHeadAttention expects inputs as [query, value] or [query, value, key].');
    }

    const query = inputs[0];
    const value = inputs[1];
    const key = inputs[2] ?? value;

    return { query, value, key };
  }

  call(inputs, kwargs = {}) {
    const { query, value, key } = this._extractInputTensors(inputs);

    const returnAttentionScores = normalizeKwargBoolean(
      kwargs,
      'return_attention_scores',
      'returnAttentionScores',
      false
    );
    const useCausalMask = normalizeKwargBoolean(
      kwargs,
      'use_causal_mask',
      'useCausalMask',
      false
    );

    const training = getKwargValue(kwargs, 'training', 'training', null);

    const queryMask = getKwargValue(kwargs, 'query_mask', 'queryMask', resolveMaskFromTensor(query));
    const valueMask = getKwargValue(kwargs, 'value_mask', 'valueMask', resolveMaskFromTensor(value));
    const keyMask = getKwargValue(kwargs, 'key_mask', 'keyMask', resolveMaskFromTensor(key));
    const attentionMask = getKwargValue(kwargs, 'attention_mask', 'attentionMask', null);

    return tf.tidy(() => {
      const useRank3Path = this._canUseRank3Path(query, key, value);
      const outputTail = this._outputShape ?? [query.shape[query.shape.length - 1]];

      const mergedAttentionMask = this._computeAttentionMask(
        query,
        value,
        queryMask,
        valueMask,
        keyMask,
        attentionMask,
        useCausalMask
      );

      const projectedQuery = useRank3Path
        ? this._projectRank3(query, this._queryKernel, this._queryBias, [this._numHeads, this._keyDim])
        : this._project(query, this._queryProjectionEquation, this._queryKernel, this._queryBias);

      const projectedKey = useRank3Path
        ? this._projectRank3(key, this._keyKernel, this._keyBias, [this._numHeads, this._keyDim])
        : this._project(key, this._keyProjectionEquation, this._keyKernel, this._keyBias);

      const projectedValue = useRank3Path
        ? this._projectRank3(value, this._valueKernel, this._valueBias, [this._numHeads, this._valueDim])
        : this._project(value, this._valueProjectionEquation, this._valueKernel, this._valueBias);

      const attention = useRank3Path
        ? this._computeAttentionRank3(
          projectedQuery,
          projectedKey,
          projectedValue,
          mergedAttentionMask,
          training,
          returnAttentionScores
        )
        : this._computeAttention(
          projectedQuery,
          projectedKey,
          projectedValue,
          mergedAttentionMask,
          training,
          returnAttentionScores
        );

      const output = useRank3Path
        ? this._projectOutputRank3(attention.attentionOutput, outputTail)
        : this._project(
          attention.attentionOutput,
          this._outputProjectionEquation,
          this._outputKernel,
          this._outputBias
        );

      const keptOutput = tf.keep(output);
      if (queryMask != null) {
        keptOutput.kerasMask = queryMask;
      }

      if (returnAttentionScores) {
        const keptScores = tf.keep(attention.attentionScores);
        return [keptOutput, keptScores];
      }

      return keptOutput;
    });
  }

  computeMask(inputs, mask = null) {
    const asMaskOutput = (queryMask) => {
      if (this._returnAttentionScoresForShapeComputation) {
        return [queryMask, null];
      }
      return queryMask;
    };

    if (mask != null) {
      if (Array.isArray(mask)) {
        return asMaskOutput(mask[0] ?? null);
      }
      return asMaskOutput(mask);
    }

    const query = Array.isArray(inputs) ? inputs[0] : inputs;
    const queryMask = resolveMaskFromTensor(query);
    return asMaskOutput(queryMask ?? null);
  }

  computeOutputShape(inputShape) {
    const { queryShape, valueShape, keyShape } = this._extractBuildShapes(inputShape);
    const outputTail = this._outputShape ?? [queryShape[queryShape.length - 1]];
    const outputShape = [...queryShape.slice(0, -1), ...outputTail];

    if (!this._returnAttentionScoresForShapeComputation) {
      return outputShape;
    }

    const projectedRank = queryShape.length + 1;
    const attentionAxes = this._normalizeAttentionAxes(projectedRank);

    const queryProjectedShape = [...queryShape.slice(0, -1), this._numHeads, this._keyDim];
    const keyProjectedShape = [...keyShape.slice(0, -1), this._numHeads, this._keyDim];

    const excluded = new Set([...attentionAxes, projectedRank - 1]);
    const batchDims = [];
    for (let axis = 0; axis < projectedRank; axis += 1) {
      if (!excluded.has(axis)) {
        batchDims.push(axis);
      }
    }

    const scoresShape = [
      ...batchDims.map((axis) => queryProjectedShape[axis]),
      ...attentionAxes.map((axis) => queryProjectedShape[axis]),
      ...attentionAxes.map((axis) => keyProjectedShape[axis]),
    ];

    return [outputShape, scoresShape];
  }

  apply(inputs, kwargs = {}) {
    const previous = this._returnAttentionScoresForShapeComputation;
    this._returnAttentionScoresForShapeComputation = normalizeKwargBoolean(
      kwargs,
      'return_attention_scores',
      'returnAttentionScores',
      false
    );

    try {
      return super.apply(inputs, kwargs);
    } finally {
      this._returnAttentionScoresForShapeComputation = previous;
    }
  }

  getConfig() {
    const baseConfig = super.getConfig();

    return {
      ...baseConfig,
      num_heads: this._numHeads,
      key_dim: this._keyDim,
      value_dim: this._valueDim,
      dropout: this._dropout,
      use_bias: this._useBias,
      output_shape: this._outputShape,
      attention_axes: this._attentionAxesConfig,
      flash_attention: this._flashAttention,
      kernel_initializer: serializeKerasObject(this._kernelInitializer),
      bias_initializer: serializeKerasObject(this._biasInitializer),
      kernel_regularizer: serializeKerasObject(this._kernelRegularizer),
      bias_regularizer: serializeKerasObject(this._biasRegularizer),
      activity_regularizer: serializeKerasObject(this._activityRegularizer),
      kernel_constraint: serializeKerasObject(this._kernelConstraint),
      bias_constraint: serializeKerasObject(this._biasConstraint),
      seed: this.seed,
    };
  }
}

export function multiHeadAttention(config) {
  return new MultiHeadAttention(config);
}
