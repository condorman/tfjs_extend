import * as tf from '@tensorflow/tfjs';

export interface AdamWConfig {
  learning_rate?: number;
  weight_decay?: number;
  beta_1?: number;
  beta_2?: number;
  epsilon?: number;
  amsgrad?: boolean;
  clipnorm?: number | null;
  clipvalue?: number | null;
  global_clipnorm?: number | null;
  use_ema?: boolean;
  ema_momentum?: number;
  ema_overwrite_frequency?: number | null;
  loss_scale_factor?: number | null;
  gradient_accumulation_steps?: number | null;
  name?: string;
  learningRate?: number;
  weightDecay?: number;
  beta1?: number;
  beta2?: number;
  clipNorm?: number | null;
  clipValue?: number | null;
  globalClipNorm?: number | null;
  useEma?: boolean;
  emaMomentum?: number;
  emaOverwriteFrequency?: number | null;
  lossScaleFactor?: number | null;
  gradientAccumulationSteps?: number | null;
  globalClipnorm?: number | null;
}

export interface AdamWNormalizedConfig {
  learning_rate: number;
  weight_decay: number;
  beta_1: number;
  beta_2: number;
  epsilon: number;
  amsgrad: boolean;
  clipnorm: number | null;
  clipvalue: number | null;
  global_clipnorm: number | null;
  use_ema: boolean;
  ema_momentum: number;
  ema_overwrite_frequency: number | null;
  loss_scale_factor: number | null;
  gradient_accumulation_steps: number | null;
  name: string;
}

export interface AdamWState {
  iterations: number;
  momentum: tf.Tensor;
  velocity: tf.Tensor;
  velocityHat: tf.Tensor | null;
  gradientAccumulator: tf.Tensor | null;
  ema: tf.Tensor | null;
}

export interface AdamWStepResult {
  variable: tf.Tensor;
  state: AdamWState;
}

export interface AdamWNamedGradient {
  name: string;
  tensor: tf.Tensor;
}

export type AdamWGradientInput = Record<string, tf.Tensor> | AdamWNamedGradient[];

export class AdamW extends tf.Optimizer {
  static readonly DEFAULTS: Readonly<AdamWNormalizedConfig>;
  static readonly CONFIG_ALIASES: Readonly<Record<string, keyof AdamWNormalizedConfig>>;
  static ALLOWED_CONFIG_KEYS: Set<string> | null;
  static readonly className: string;

  static normalizeConfig(config?: AdamWConfig): AdamWNormalizedConfig;
  static getOptimizerIterations(internalIterations: number, config?: AdamWConfig): number;
  static createState(variable: tf.Tensor | tf.TensorLike, config?: AdamWConfig): AdamWState;
  static disposeState(state?: AdamWState | null): void;
  static step(
    variable: tf.Tensor | tf.TensorLike,
    gradient: tf.Tensor | tf.TensorLike,
    state?: AdamWState | null,
    config?: AdamWConfig
  ): AdamWStepResult;
  static fromConfig<T extends tf.serialization.Serializable>(
    cls: tf.serialization.SerializableConstructor<T>,
    config: tf.serialization.ConfigDict
  ): T;

  constructor(options?: AdamWConfig);

  applyGradients(variableGradients: AdamWGradientInput): void;
  getConfig(): tf.serialization.ConfigDict;
  dispose(): void;
}
