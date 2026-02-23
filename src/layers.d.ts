import * as tf from '@tensorflow/tfjs';

export interface MultiHeadAttentionConfig extends tf.layers.LayerArgs {
  num_heads?: number;
  key_dim?: number;
  value_dim?: number | null;
  dropout?: number;
  use_bias?: boolean;
  output_shape?: number | number[] | null;
  attention_axes?: number | number[] | null;
  flash_attention?: boolean | null;
  kernel_initializer?: tf.initializers.InitializerIdentifier | tf.serialization.ConfigDict | tf.initializers.Initializer;
  bias_initializer?: tf.initializers.InitializerIdentifier | tf.serialization.ConfigDict | tf.initializers.Initializer;
  kernel_regularizer?: tf.regularizers.RegularizerIdentifier | tf.serialization.ConfigDict | tf.regularizers.Regularizer;
  bias_regularizer?: tf.regularizers.RegularizerIdentifier | tf.serialization.ConfigDict | tf.regularizers.Regularizer;
  activity_regularizer?: tf.regularizers.RegularizerIdentifier | tf.serialization.ConfigDict | tf.regularizers.Regularizer;
  kernel_constraint?: tf.constraints.ConstraintIdentifier | tf.serialization.ConfigDict | tf.constraints.Constraint;
  bias_constraint?: tf.constraints.ConstraintIdentifier | tf.serialization.ConfigDict | tf.constraints.Constraint;
  seed?: number | null;

  numHeads?: number;
  keyDim?: number;
  valueDim?: number | null;
  useBias?: boolean;
  outputShape?: number | number[] | null;
  attentionAxes?: number | number[] | null;
  flashAttention?: boolean | null;
  kernelInitializer?: tf.initializers.InitializerIdentifier | tf.serialization.ConfigDict | tf.initializers.Initializer;
  biasInitializer?: tf.initializers.InitializerIdentifier | tf.serialization.ConfigDict | tf.initializers.Initializer;
  kernelRegularizer?: tf.regularizers.RegularizerIdentifier | tf.serialization.ConfigDict | tf.regularizers.Regularizer;
  biasRegularizer?: tf.regularizers.RegularizerIdentifier | tf.serialization.ConfigDict | tf.regularizers.Regularizer;
  activityRegularizer?: tf.regularizers.RegularizerIdentifier | tf.serialization.ConfigDict | tf.regularizers.Regularizer;
  kernelConstraint?: tf.constraints.ConstraintIdentifier | tf.serialization.ConfigDict | tf.constraints.Constraint;
  biasConstraint?: tf.constraints.ConstraintIdentifier | tf.serialization.ConfigDict | tf.constraints.Constraint;
}

export interface MultiHeadAttentionCallKwargs {
  attention_mask?: tf.Tensor | tf.TensorLike;
  attentionMask?: tf.Tensor | tf.TensorLike;
  return_attention_scores?: boolean;
  returnAttentionScores?: boolean;
  training?: boolean;
  use_causal_mask?: boolean;
  useCausalMask?: boolean;
  query_mask?: tf.Tensor | tf.TensorLike;
  queryMask?: tf.Tensor | tf.TensorLike;
  value_mask?: tf.Tensor | tf.TensorLike;
  valueMask?: tf.Tensor | tf.TensorLike;
  key_mask?: tf.Tensor | tf.TensorLike;
  keyMask?: tf.Tensor | tf.TensorLike;
}

export class MultiHeadAttention extends tf.layers.Layer {
  static className: string;
  static readonly DEFAULTS: Readonly<Record<string, unknown>>;
  static readonly CONFIG_ALIASES: Readonly<Record<string, string>>;
  static ALLOWED_CONFIG_KEYS: Set<string> | null;

  static normalizeConfig(config?: MultiHeadAttentionConfig): Record<string, unknown>;

  readonly numHeads: number;
  readonly keyDim: number;
  readonly valueDim: number;
  readonly dropout: number;
  readonly useBias: boolean;
  readonly attentionAxes: number | number[] | null;

  constructor(options?: MultiHeadAttentionConfig);

  call(inputs: tf.Tensor | tf.Tensor[], kwargs?: MultiHeadAttentionCallKwargs): tf.Tensor | tf.Tensor[];
  apply(inputs: tf.Tensor | tf.SymbolicTensor | Array<tf.Tensor | tf.SymbolicTensor>, kwargs?: MultiHeadAttentionCallKwargs):
    tf.Tensor | tf.SymbolicTensor | Array<tf.Tensor | tf.SymbolicTensor>;
  getConfig(): tf.serialization.ConfigDict;
}

export function multiHeadAttention(config?: MultiHeadAttentionConfig): MultiHeadAttention;
