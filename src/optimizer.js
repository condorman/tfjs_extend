import * as tf from '@tensorflow/tfjs';
import {
  asFloatTensor,
  assertFiniteNumber,
  createSlotVariable,
  getRegisteredVariable,
  isTensorDisposed,
  normalizeGradientsInput,
  tensorShapeEquals,
} from './_utils.js';

export class AdamW extends tf.Optimizer {
  static DEFAULTS = Object.freeze({
    learning_rate: 0.001,
    weight_decay: 0.004,
    beta_1: 0.9,
    beta_2: 0.999,
    epsilon: 1e-7,
    amsgrad: false,
    clipnorm: null,
    clipvalue: null,
    global_clipnorm: null,
    use_ema: false,
    ema_momentum: 0.99,
    ema_overwrite_frequency: null,
    loss_scale_factor: null,
    gradient_accumulation_steps: null,
    name: 'adamw',
  });

  static CONFIG_ALIASES = Object.freeze({
    learningRate: 'learning_rate',
    weightDecay: 'weight_decay',
    beta1: 'beta_1',
    beta2: 'beta_2',
    clipNorm: 'clipnorm',
    clipValue: 'clipvalue',
    globalClipNorm: 'global_clipnorm',
    useEma: 'use_ema',
    emaMomentum: 'ema_momentum',
    emaOverwriteFrequency: 'ema_overwrite_frequency',
    lossScaleFactor: 'loss_scale_factor',
    gradientAccumulationSteps: 'gradient_accumulation_steps',
    globalClipnorm: 'global_clipnorm',
  });

  static ALLOWED_CONFIG_KEYS = null;

  static {
    this.ALLOWED_CONFIG_KEYS = new Set([
      ...Object.keys(this.DEFAULTS),
      ...Object.keys(this.CONFIG_ALIASES),
    ]);
    tf.serialization.registerClass(this);
  }

  static get className() {
    // Keep Python-compatible naming for serialization.
    return 'AdamW';
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

  static normalizeConfig(config = {}) {
    const canonical = this._canonicalizeConfig(config);
    const normalized = {
      ...this.DEFAULTS,
      ...canonical,
    };

    assertFiniteNumber(normalized.learning_rate, 'learning_rate');
    assertFiniteNumber(normalized.weight_decay, 'weight_decay');
    assertFiniteNumber(normalized.beta_1, 'beta_1');
    assertFiniteNumber(normalized.beta_2, 'beta_2');
    assertFiniteNumber(normalized.epsilon, 'epsilon');
    assertFiniteNumber(normalized.ema_momentum, 'ema_momentum');

    if (typeof normalized.amsgrad !== 'boolean') {
      throw new TypeError(`Argument \`amsgrad\` must be a boolean. Received: ${normalized.amsgrad}`);
    }
    if (typeof normalized.use_ema !== 'boolean') {
      throw new TypeError(`Argument \`use_ema\` must be a boolean. Received: ${normalized.use_ema}`);
    }

    if (normalized.gradient_accumulation_steps != null) {
      if (!Number.isInteger(normalized.gradient_accumulation_steps) || normalized.gradient_accumulation_steps < 2) {
        throw new Error(
          '`gradient_accumulation_steps` must be an integer >= 2. '
          + `Received: ${normalized.gradient_accumulation_steps}`
        );
      }
    }

    if (normalized.ema_momentum > 1 || normalized.ema_momentum < 0) {
      throw new Error(`\`ema_momentum\` must be in [0, 1]. Received: ${normalized.ema_momentum}`);
    }

    if (normalized.ema_overwrite_frequency != null) {
      if (!Number.isInteger(normalized.ema_overwrite_frequency) || normalized.ema_overwrite_frequency < 1) {
        throw new Error(
          '`ema_overwrite_frequency` must be an integer >= 1 or null. '
          + `Received: ${normalized.ema_overwrite_frequency}`
        );
      }
    }

    if (normalized.loss_scale_factor != null) {
      assertFiniteNumber(normalized.loss_scale_factor, 'loss_scale_factor');
    }

    const clipCount = [normalized.clipnorm, normalized.clipvalue, normalized.global_clipnorm]
      .filter((value) => value != null)
      .length;

    if (clipCount > 1) {
      throw new Error(
        'Only one of `clipnorm`, `clipvalue` and `global_clipnorm` can be set. '
        + `Received: clipnorm=${normalized.clipnorm}, clipvalue=${normalized.clipvalue}, `
        + `global_clipnorm=${normalized.global_clipnorm}`
      );
    }

    for (const key of ['clipnorm', 'clipvalue', 'global_clipnorm']) {
      if (normalized[key] != null) {
        assertFiniteNumber(normalized[key], key);
      }
    }

    if (normalized.name != null && typeof normalized.name !== 'string') {
      throw new TypeError(`Argument \`name\` must be a string. Received: ${normalized.name}`);
    }

    return normalized;
  }

  static getOptimizerIterations(internalIterations, config = {}) {
    const normalized = this.normalizeConfig(config);
    if (normalized.gradient_accumulation_steps) {
      return Math.floor(internalIterations / normalized.gradient_accumulation_steps);
    }
    return internalIterations;
  }

  static createState(variable, config = {}) {
    const normalized = this.normalizeConfig(config);
    const { tensor: variableTensor, owned } = asFloatTensor(variable, 'variable');

    const state = tf.tidy(() => {
      const zeros = tf.zerosLike(variableTensor);
      return {
        iterations: 0,
        momentum: tf.keep(tf.clone(zeros)),
        velocity: tf.keep(tf.clone(zeros)),
        velocityHat: normalized.amsgrad ? tf.keep(tf.clone(zeros)) : null,
        gradientAccumulator: normalized.gradient_accumulation_steps ? tf.keep(tf.clone(zeros)) : null,
        ema: normalized.use_ema ? tf.keep(tf.clone(zeros)) : null,
      };
    });

    if (owned) {
      variableTensor.dispose();
    }

    return state;
  }

  static disposeState(state) {
    if (!state) {
      return;
    }

    for (const key of ['momentum', 'velocity', 'velocityHat', 'gradientAccumulator', 'ema']) {
      const tensor = state[key];
      if (tensor instanceof tf.Tensor && !isTensorDisposed(tensor)) {
        tensor.dispose();
      }
    }
  }

  static _validateStateTensor(tensor, referenceShape, name) {
    if (!(tensor instanceof tf.Tensor)) {
      throw new TypeError(`State field \`${name}\` must be a Tensor.`);
    }
    if (tensor.dtype !== 'float32') {
      throw new TypeError(`State field \`${name}\` must have dtype float32.`);
    }
    if (!tensorShapeEquals(tensor.shape, referenceShape)) {
      throw new Error(
        `State field \`${name}\` shape mismatch. Expected [${referenceShape.join(', ')}], `
        + `received [${tensor.shape.join(', ')}].`
      );
    }
    if (isTensorDisposed(tensor)) {
      throw new Error(`State field \`${name}\` is already disposed.`);
    }
  }

  static _resolveState(state, variableTensor, config) {
    if (!state) {
      throw new Error('State is required. Use `AdamW.createState(variable, config)` for the first step.');
    }

    const iterations = Number.isInteger(state.iterations) && state.iterations >= 0
      ? state.iterations
      : 0;

    this._validateStateTensor(state.momentum, variableTensor.shape, 'momentum');
    this._validateStateTensor(state.velocity, variableTensor.shape, 'velocity');

    let velocityHat = null;
    if (config.amsgrad) {
      if (state.velocityHat == null) {
        throw new Error('State field `velocityHat` is required when `amsgrad=true`.');
      }
      this._validateStateTensor(state.velocityHat, variableTensor.shape, 'velocityHat');
      velocityHat = state.velocityHat;
    }

    let gradientAccumulator = null;
    if (config.gradient_accumulation_steps) {
      if (state.gradientAccumulator == null) {
        throw new Error(
          'State field `gradientAccumulator` is required when `gradient_accumulation_steps` is configured.'
        );
      }
      this._validateStateTensor(state.gradientAccumulator, variableTensor.shape, 'gradientAccumulator');
      gradientAccumulator = state.gradientAccumulator;
    }

    let ema = null;
    if (config.use_ema) {
      if (state.ema == null) {
        throw new Error('State field `ema` is required when `use_ema=true`.');
      }
      this._validateStateTensor(state.ema, variableTensor.shape, 'ema');
      ema = state.ema;
    }

    return {
      iterations,
      momentum: state.momentum,
      velocity: state.velocity,
      velocityHat,
      gradientAccumulator,
      ema,
    };
  }

  static _clipByNorm(values, clipnorm) {
    const l2sum = tf.sum(tf.square(values));
    const pred = tf.greater(l2sum, 0);
    const l2sumSafe = tf.where(pred, l2sum, tf.onesLike(l2sum));
    const l2norm = tf.where(pred, tf.sqrt(l2sumSafe), l2sum);
    const scale = tf.div(clipnorm, tf.maximum(l2norm, clipnorm));
    return tf.mul(values, scale);
  }

  static _globalNormScaleFromNorm(useNorm, clipnorm) {
    const scaleForFinite = tf.mul(
      clipnorm,
      tf.minimum(tf.div(1, useNorm), 1 / clipnorm)
    );
    return tf.add(scaleForFinite, tf.sub(useNorm, useNorm));
  }

  static _clipByGlobalNorm(values, clipnorm) {
    const useNorm = tf.sqrt(tf.sum(tf.square(values)));
    const scale = this._globalNormScaleFromNorm(useNorm, clipnorm);
    return tf.mul(values, scale);
  }

  static _clipGradient(values, config) {
    if (config.clipnorm != null && config.clipnorm > 0) {
      return this._clipByNorm(values, config.clipnorm);
    }
    if (config.global_clipnorm != null && config.global_clipnorm > 0) {
      return this._clipByGlobalNorm(values, config.global_clipnorm);
    }
    if (config.clipvalue != null && config.clipvalue > 0) {
      return tf.clipByValue(values, -config.clipvalue, config.clipvalue);
    }
    return values;
  }

  static _clipGradients(gradients, config) {
    if (gradients.length === 0) {
      return gradients;
    }

    if (config.clipnorm != null && config.clipnorm > 0) {
      return gradients.map((gradient) => this._clipByNorm(gradient, config.clipnorm));
    }

    if (config.global_clipnorm != null && config.global_clipnorm > 0) {
      const squaredNorms = gradients.map((gradient) => tf.sum(tf.square(gradient)));
      const squaredNormSum = squaredNorms.length === 1 ? squaredNorms[0] : tf.addN(squaredNorms);
      const useNorm = tf.sqrt(squaredNormSum);
      const scale = this._globalNormScaleFromNorm(useNorm, config.global_clipnorm);
      return gradients.map((gradient) => tf.mul(gradient, scale));
    }

    if (config.clipvalue != null && config.clipvalue > 0) {
      return gradients.map((gradient) => tf.clipByValue(gradient, -config.clipvalue, config.clipvalue));
    }

    return gradients;
  }

  static step(variable, gradient, state = null, config = {}) {
    const normalized = this.normalizeConfig(config);
    const { tensor: variableTensor, owned: ownsVariable } = asFloatTensor(variable, 'variable');
    const { tensor: gradientTensor, owned: ownsGradient } = asFloatTensor(gradient, 'gradient');

    if (!tensorShapeEquals(variableTensor.shape, gradientTensor.shape)) {
      if (ownsVariable) {
        variableTensor.dispose();
      }
      if (ownsGradient) {
        gradientTensor.dispose();
      }
      throw new Error(
        `Gradient shape mismatch. variable shape [${variableTensor.shape.join(', ')}], `
        + `gradient shape [${gradientTensor.shape.join(', ')}].`
      );
    }

    const activeState = state ?? this.createState(variableTensor, normalized);
    const resolvedState = this._resolveState(activeState, variableTensor, normalized);
    const nextState = tf.tidy(() => {
      const steps = normalized.gradient_accumulation_steps;
      const nextInternalIterations = resolvedState.iterations + 1;
      const shouldUpdate = !steps || (nextInternalIterations % steps === 0);

      const optimizerIterations = steps
        ? Math.floor(resolvedState.iterations / steps)
        : resolvedState.iterations;

      let grad = gradientTensor;
      if (normalized.loss_scale_factor != null) {
        grad = tf.div(grad, normalized.loss_scale_factor);
      }

      let currentVariable = variableTensor;
      let currentMomentum = resolvedState.momentum;
      let currentVelocity = resolvedState.velocity;
      let currentVelocityHat = resolvedState.velocityHat;
      let currentAccumulator = resolvedState.gradientAccumulator;
      let currentEma = resolvedState.ema;

      if (steps) {
        if (shouldUpdate) {
          grad = tf.div(tf.add(grad, resolvedState.gradientAccumulator), steps);
          currentAccumulator = tf.zerosLike(resolvedState.gradientAccumulator);
        } else {
          currentAccumulator = tf.add(resolvedState.gradientAccumulator, grad);
        }
      }

      if (shouldUpdate) {
        grad = this._clipGradient(grad, normalized);

        currentVariable = tf.sub(
          currentVariable,
          tf.mul(currentVariable, normalized.weight_decay * normalized.learning_rate)
        );

        const localStep = optimizerIterations + 1;
        const beta1Power = Math.pow(normalized.beta_1, localStep);
        const beta2Power = Math.pow(normalized.beta_2, localStep);
        const alpha = (
          normalized.learning_rate
          * Math.sqrt(1 - beta2Power)
          / (1 - beta1Power)
        );

        currentMomentum = tf.add(
          currentMomentum,
          tf.mul(tf.sub(grad, currentMomentum), 1 - normalized.beta_1)
        );
        currentVelocity = tf.add(
          currentVelocity,
          tf.mul(tf.sub(tf.square(grad), currentVelocity), 1 - normalized.beta_2)
        );

        let sourceVelocity = currentVelocity;
        if (normalized.amsgrad) {
          currentVelocityHat = tf.maximum(resolvedState.velocityHat, currentVelocity);
          sourceVelocity = currentVelocityHat;
        }

        currentVariable = tf.sub(
          currentVariable,
          tf.div(
            tf.mul(currentMomentum, alpha),
            tf.add(tf.sqrt(sourceVelocity), normalized.epsilon)
          )
        );
      }

      if (normalized.use_ema) {
        const notFirstStep = optimizerIterations !== 0;
        const momentum = notFirstStep ? normalized.ema_momentum : 0;

        currentEma = tf.add(
          tf.mul(currentEma, momentum),
          tf.mul(currentVariable, 1 - momentum)
        );

        if (normalized.ema_overwrite_frequency) {
          const shouldOverwrite = ((optimizerIterations + 1) % normalized.ema_overwrite_frequency) === 0;
          if (shouldOverwrite) {
            currentVariable = currentEma;
          }
        }
      }

      return {
        iterations: nextInternalIterations,
        variable: tf.keep(tf.clone(currentVariable)),
        momentum: tf.keep(tf.clone(currentMomentum)),
        velocity: tf.keep(tf.clone(currentVelocity)),
        velocityHat: normalized.amsgrad ? tf.keep(tf.clone(currentVelocityHat)) : null,
        gradientAccumulator: normalized.gradient_accumulation_steps
          ? tf.keep(tf.clone(currentAccumulator))
          : null,
        ema: normalized.use_ema ? tf.keep(tf.clone(currentEma)) : null,
      };
    });

    if (state == null) {
      this.disposeState(activeState);
    }

    if (ownsVariable) {
      variableTensor.dispose();
    }
    if (ownsGradient) {
      gradientTensor.dispose();
    }

    return {
      variable: nextState.variable,
      state: {
        iterations: nextState.iterations,
        momentum: nextState.momentum,
        velocity: nextState.velocity,
        velocityHat: nextState.velocityHat,
        gradientAccumulator: nextState.gradientAccumulator,
        ema: nextState.ema,
      },
    };
  }

  constructor(options = {}) {
    super();
    this.config = this.constructor.normalizeConfig(options);
    this._useEma = this.config.use_ema;
    this._emaMomentum = this.config.ema_momentum;
    this._emaOverwriteFrequency = this.config.ema_overwrite_frequency;
    this._learningRate = this.config.learning_rate;
    this._beta1 = this.config.beta_1;
    this._beta2 = this.config.beta_2;
    this._epsilon = this.config.epsilon;
    this._amsgrad = this.config.amsgrad;
    this._clipnorm = this.config.clipnorm;
    this._globalClipNorm = this.config.global_clipnorm;
    this._clipvalue = this.config.clipvalue;
    this.slotsByName = new Map();
    this._oneMinusBeta1 = 1 - this._beta1;
    this._oneMinusBeta2 = 1 - this._beta2;
    this._weightDecayStep = this.config.weight_decay * this._learningRate;
    this._inverseLossScale = this.config.loss_scale_factor == null
      ? null
      : (1 / this.config.loss_scale_factor);
    this._beta1Power = 1;
    this._beta2Power = 1;
    this._biasPowerIteration = 0;
    this._scratchVariables = [];
    this._scratchSlots = [];
    this._scratchGradients = [];
    this._scratchSquaredNorms = [];

    if (this._clipnorm != null && this._clipnorm > 0) {
      this._clipMode = 'norm';
    } else if (this._globalClipNorm != null && this._globalClipNorm > 0) {
      this._clipMode = 'global';
    } else if (this._clipvalue != null && this._clipvalue > 0) {
      this._clipMode = 'value';
    } else {
      this._clipMode = 'none';
    }
  }

  _createSlotsForVariable(name, variable) {
    return {
      name,
      variableRef: variable,
      momentum: {
        originalName: `${name}/m`,
        variable: createSlotVariable(variable),
      },
      velocity: {
        originalName: `${name}/v`,
        variable: createSlotVariable(variable),
      },
      velocityHat: this._amsgrad ? {
        originalName: `${name}/vhat`,
        variable: createSlotVariable(variable),
      } : null,
      gradientAccumulator: this.config.gradient_accumulation_steps ? {
        originalName: `${name}/gacc`,
        variable: createSlotVariable(variable),
      } : null,
      ema: this._useEma ? {
        originalName: `${name}/ema`,
        variable: createSlotVariable(variable),
      } : null,
    };
  }

  _disposeSlotEntry(slotEntry) {
    const values = [
      slotEntry?.momentum?.variable,
      slotEntry?.velocity?.variable,
      slotEntry?.velocityHat?.variable,
      slotEntry?.gradientAccumulator?.variable,
      slotEntry?.ema?.variable,
    ];

    for (const variable of values) {
      if (variable instanceof tf.Tensor && !isTensorDisposed(variable)) {
        variable.dispose();
      }
    }
  }

  _ensureSlots(name, variable) {
    let slotEntry = this.slotsByName.get(name);
    if (!slotEntry) {
      slotEntry = this._createSlotsForVariable(name, variable);
      this.slotsByName.set(name, slotEntry);
      return slotEntry;
    }

    if (
      !tensorShapeEquals(slotEntry.momentum.variable.shape, variable.shape)
      || slotEntry.momentum.variable.dtype !== variable.dtype
    ) {
      this._disposeSlotEntry(slotEntry);
      slotEntry = this._createSlotsForVariable(name, variable);
      this.slotsByName.set(name, slotEntry);
    }

    slotEntry.name = name;
    slotEntry.variableRef = variable;
    return slotEntry;
  }

  _applyEma(optimizerIterations) {
    if (!this._useEma) {
      return;
    }

    const momentum = optimizerIterations !== 0 ? this._emaMomentum : 0;

    for (const slotEntry of this.slotsByName.values()) {
      let variable = slotEntry.variableRef;
      if (!(variable instanceof tf.Variable) || isTensorDisposed(variable)) {
        variable = tf.engine().registeredVariables[slotEntry.name];
        slotEntry.variableRef = variable ?? null;
      }
      if (variable == null) {
        continue;
      }
      const nextEma = tf.add(
        tf.mul(slotEntry.ema.variable, momentum),
        tf.mul(variable, 1 - momentum)
      );
      slotEntry.ema.variable.assign(nextEma);
    }

    if (this._emaOverwriteFrequency) {
      const shouldOverwrite = ((optimizerIterations + 1) % this._emaOverwriteFrequency) === 0;
      if (shouldOverwrite) {
        for (const slotEntry of this.slotsByName.values()) {
          let variable = slotEntry.variableRef;
          if (!(variable instanceof tf.Variable) || isTensorDisposed(variable)) {
            variable = tf.engine().registeredVariables[slotEntry.name];
            slotEntry.variableRef = variable ?? null;
          }
          if (variable == null) {
            continue;
          }
          variable.assign(slotEntry.ema.variable);
        }
      }
    }
  }

  _syncBiasCorrectionPowers(optimizerIterations) {
    if (this._biasPowerIteration === optimizerIterations) {
      return;
    }
    this._beta1Power = Math.pow(this._beta1, optimizerIterations);
    this._beta2Power = Math.pow(this._beta2, optimizerIterations);
    this._biasPowerIteration = optimizerIterations;
  }

  applyGradients(variableGradients) {
    const { names: gradientNames, map: gradientMap } = normalizeGradientsInput(variableGradients);
    const internalIterations = this.iterations;
    const steps = this.config.gradient_accumulation_steps;
    const shouldUpdate = !steps || ((internalIterations + 1) % steps === 0);
    const optimizerIterations = steps
      ? Math.floor(internalIterations / steps)
      : internalIterations;
    const inverseSteps = steps ? (1 / steps) : null;
    const inverseLossScale = this._inverseLossScale;
    const oneMinusBeta1 = this._oneMinusBeta1;
    const oneMinusBeta2 = this._oneMinusBeta2;
    const epsilon = this._epsilon;
    const weightDecayStep = this._weightDecayStep;
    const amsgrad = this._amsgrad;

    tf.tidy(() => {
      const variables = this._scratchVariables;
      const slotsList = this._scratchSlots;
      const gradients = this._scratchGradients;
      const squaredNorms = this._scratchSquaredNorms;
      variables.length = 0;
      slotsList.length = 0;
      gradients.length = 0;
      squaredNorms.length = 0;

      if (steps && !shouldUpdate) {
        for (const name of gradientNames) {
          let gradient = gradientMap[name];
          if (gradient == null) {
            continue;
          }

          const variable = getRegisteredVariable(name);
          const slots = this._ensureSlots(name, variable);

          if (gradient.dtype !== variable.dtype) {
            gradient = tf.cast(gradient, variable.dtype);
          }

          if (inverseLossScale != null) {
            gradient = tf.mul(gradient, inverseLossScale);
          }

          const nextAccumulator = tf.add(slots.gradientAccumulator.variable, gradient);
          slots.gradientAccumulator.variable.assign(nextAccumulator);
        }

        this._applyEma(optimizerIterations);
        return;
      }

      for (const name of gradientNames) {
        let gradient = gradientMap[name];
        if (gradient == null) {
          continue;
        }

        const variable = getRegisteredVariable(name);
        const slots = this._ensureSlots(name, variable);

        if (gradient.dtype !== variable.dtype) {
          gradient = tf.cast(gradient, variable.dtype);
        }

        if (inverseLossScale != null) {
          gradient = tf.mul(gradient, inverseLossScale);
        }

        if (steps) {
          gradient = tf.mul(tf.add(gradient, slots.gradientAccumulator.variable), inverseSteps);
        }

        variables.push(variable);
        slotsList.push(slots);
        gradients.push(gradient);
      }

      if (gradients.length > 0) {
        if (this._clipMode === 'norm') {
          for (let index = 0; index < gradients.length; index += 1) {
            gradients[index] = this.constructor._clipByNorm(gradients[index], this._clipnorm);
          }
        } else if (this._clipMode === 'global') {
          for (const gradient of gradients) {
            squaredNorms.push(tf.sum(tf.square(gradient)));
          }
          const squaredNormSum = squaredNorms.length === 1 ? squaredNorms[0] : tf.addN(squaredNorms);
          const useNorm = tf.sqrt(squaredNormSum);
          const scale = this.constructor._globalNormScaleFromNorm(useNorm, this._globalClipNorm);
          for (let index = 0; index < gradients.length; index += 1) {
            gradients[index] = tf.mul(gradients[index], scale);
          }
        } else if (this._clipMode === 'value') {
          for (let index = 0; index < gradients.length; index += 1) {
            gradients[index] = tf.clipByValue(gradients[index], -this._clipvalue, this._clipvalue);
          }
        }

        this._syncBiasCorrectionPowers(optimizerIterations);
        this._beta1Power *= this._beta1;
        this._beta2Power *= this._beta2;
        this._biasPowerIteration = optimizerIterations + 1;
        const alpha = (
          this._learningRate
          * Math.sqrt(1 - this._beta2Power)
          / (1 - this._beta1Power)
        );

        for (let index = 0; index < gradients.length; index += 1) {
          const variable = variables[index];
          const slots = slotsList[index];
          const gradient = gradients[index];
          const m = slots.momentum.variable;
          const v = slots.velocity.variable;

          const nextMomentum = tf.add(
            m,
            tf.mul(tf.sub(gradient, m), oneMinusBeta1)
          );
          const nextVelocity = tf.add(
            v,
            tf.mul(tf.sub(tf.square(gradient), v), oneMinusBeta2)
          );

          m.assign(nextMomentum);
          v.assign(nextVelocity);

          let sourceVelocity = nextVelocity;
          if (amsgrad) {
            const velocityHat = slots.velocityHat.variable;
            const nextVelocityHat = tf.maximum(velocityHat, nextVelocity);
            velocityHat.assign(nextVelocityHat);
            sourceVelocity = nextVelocityHat;
          }

          let nextVariable = tf.sub(
            variable,
            tf.div(
              tf.mul(nextMomentum, alpha),
              tf.add(tf.sqrt(sourceVelocity), epsilon)
            )
          );

          if (weightDecayStep !== 0) {
            nextVariable = tf.sub(nextVariable, tf.mul(variable, weightDecayStep));
          }

          variable.assign(nextVariable);

          if (steps) {
            slots.gradientAccumulator.variable.assign(
              tf.zerosLike(slots.gradientAccumulator.variable)
            );
          }
        }
      }

      variables.length = 0;
      slotsList.length = 0;
      gradients.length = 0;
      squaredNorms.length = 0;
      this._applyEma(optimizerIterations);
    });

    this.incrementIterations();
  }

  getConfig() {
    return { ...this.config };
  }

  static fromConfig(cls, config) {
    return new cls(config);
  }

  dispose() {
    for (const slotEntry of this.slotsByName.values()) {
      this._disposeSlotEntry(slotEntry);
    }
    this.slotsByName.clear();
  }
}
