#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import keras
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "golden.json"

CASES = [
    {
        "id": "case_1",
        "yTrue": [
            1,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            1,
            0,
        ],
        "yPred": [
            0.4117071330547333,
            0.902843713760376,
            0.6409997940063477,
            0.5055350065231323,
            0.431951105594635,
            0.8664866089820862,
            0.8498815894126892,
            0.7033447623252869,
            0.4847583770751953,
            0.5959601998329163,
            0.7057868242263794,
            0.7102806568145752,
            0.5983006358146667,
            0.517433226108551,
            0.860936164855957,
            0.4068519175052643,
            0.5899436473846436,
            0.5051687359809875,
            0.3856818377971649,
            0.5184531211853027,
            0.49085915088653564,
            0.5222195386886597,
            0.4186125695705414,
            0.5656412839889526,
            0.572314441204071,
            0.5605166554450989,
            0.5832576751708984,
            0.8286200165748596,
            0.596936047077179,
            0.5264272093772888,
            0.5534994006156921,
            0.47203877568244934,
            0.7819519639015198,
            0.8969417214393616,
            0.7793705463409424,
            0.4761424660682678,
            0.5101733207702637,
            0.6439071297645569,
            0.46762746572494507,
            0.8310959339141846,
            0.9729565382003784,
            0.9431852102279663,
            0.5766065716743469,
            0.5784488320350647,
            0.4366229176521301,
            0.5479909181594849,
            0.9358892440795898,
            0.24162320792675018,
            0.3140191435813904,
            0.13414569199085236,
            0.8507281541824341,
            0.581500232219696,
        ],
    },
    {
        "id": "case_2",
        "yTrue": [
            1,
            0,
            1,
            1,
            0,
            0,
            0,
            1,
            1,
            0,
            1,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            1,
            1,
            0,
            1,
            0,
            1,
            1,
            1,
            1,
            0,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            0,
            0,
            1,
            1,
            0,
            1,
        ],
        "yPred": [
            0.47700831294059753,
            0.48151567578315735,
            0.4810827076435089,
            0.4817756712436676,
            0.4835691452026367,
            0.49366289377212524,
            0.4819158911705017,
            0.4812319576740265,
            0.4831272065639496,
            0.4782041311264038,
            0.47673967480659485,
            0.48444560170173645,
            0.4803743064403534,
            0.48957690596580505,
            0.4784674644470215,
            0.48430752754211426,
            0.48035526275634766,
            0.4747447967529297,
            0.48019540309906006,
            0.48177847266197205,
            0.48502862453460693,
            0.4848526120185852,
            0.4851522147655487,
            0.48799189925193787,
            0.4896046817302704,
            0.48687872290611267,
            0.4879835844039917,
            0.47689881920814514,
            0.4754854440689087,
            0.48371413350105286,
            0.47623777389526367,
            0.4720148742198944,
            0.4768664538860321,
            0.4896802604198456,
            0.48693400621414185,
            0.49438050389289856,
            0.47986382246017456,
            0.4898321032524109,
            0.48782578110694885,
            0.4893357753753662,
            0.49546897411346436,
            0.48710694909095764,
            0.4941944181919098,
            0.49582749605178833,
            0.5027676224708557,
            0.4950699508190155,
            0.4832078814506531,
            0.4855714738368988,
            0.48447471857070923,
            0.4758506715297699,
            0.4844207167625427,
            0.4774407148361206,
        ],
    },
]

ADAMW_REFERENCE = "https://github.com/keras-team/keras/blob/v3.13.2/keras/src/optimizers/adamw.py#L6"

ADAMW_DEFAULTS = {
    "learning_rate": 0.001,
    "weight_decay": 0.004,
    "beta_1": 0.9,
    "beta_2": 0.999,
    "epsilon": 1e-7,
    "amsgrad": False,
    "clipnorm": None,
    "clipvalue": None,
    "global_clipnorm": None,
    "use_ema": False,
    "ema_momentum": 0.99,
    "ema_overwrite_frequency": None,
    "loss_scale_factor": None,
    "gradient_accumulation_steps": None,
    "name": "adamw",
}

ADAMW_PARAMETERS = [
    {"name": "learning_rate", "default": ADAMW_DEFAULTS["learning_rate"]},
    {"name": "weight_decay", "default": ADAMW_DEFAULTS["weight_decay"]},
    {"name": "beta_1", "default": ADAMW_DEFAULTS["beta_1"]},
    {"name": "beta_2", "default": ADAMW_DEFAULTS["beta_2"]},
    {"name": "epsilon", "default": ADAMW_DEFAULTS["epsilon"]},
    {"name": "amsgrad", "default": ADAMW_DEFAULTS["amsgrad"]},
    {"name": "clipnorm", "default": ADAMW_DEFAULTS["clipnorm"]},
    {"name": "clipvalue", "default": ADAMW_DEFAULTS["clipvalue"]},
    {"name": "global_clipnorm", "default": ADAMW_DEFAULTS["global_clipnorm"]},
    {"name": "use_ema", "default": ADAMW_DEFAULTS["use_ema"]},
    {"name": "ema_momentum", "default": ADAMW_DEFAULTS["ema_momentum"]},
    {
        "name": "ema_overwrite_frequency",
        "default": ADAMW_DEFAULTS["ema_overwrite_frequency"],
    },
    {"name": "loss_scale_factor", "default": ADAMW_DEFAULTS["loss_scale_factor"]},
    {
        "name": "gradient_accumulation_steps",
        "default": ADAMW_DEFAULTS["gradient_accumulation_steps"],
    },
    {"name": "name", "default": ADAMW_DEFAULTS["name"]},
]

ADAMW_CASES = [
    {
        "id": "adamw_default_two_steps",
        "description": "Default AdamW with two different gradient inputs.",
        "initial_variable": [1.0, -2.0, 3.0],
        "gradients": [[0.1, -0.2, 0.3], [-0.4, 0.05, 0.2]],
        "config": {},
        "covers": ["learning_rate", "weight_decay"],
    },
    {
        "id": "adamw_tuned_core_hyperparams",
        "description": "Custom learning_rate, betas, epsilon, weight_decay and name.",
        "initial_variable": [-0.5, 1.5, -2.5],
        "gradients": [[0.3, -0.1, 0.2], [0.25, -0.15, 0.05]],
        "config": {
            "learning_rate": 0.01,
            "weight_decay": 0.02,
            "beta_1": 0.8,
            "beta_2": 0.95,
            "epsilon": 1e-5,
            "name": "adamw_tuned",
        },
        "covers": ["learning_rate", "weight_decay", "beta_1", "beta_2", "epsilon", "name"],
    },
    {
        "id": "adamw_amsgrad",
        "description": "AMSGrad variant with multiple update steps.",
        "initial_variable": [0.8, -1.2, 0.4],
        "gradients": [[0.5, -0.3, 0.1], [0.05, -0.02, 0.01], [0.4, -0.25, 0.08]],
        "config": {"amsgrad": True},
        "covers": ["amsgrad"],
    },
    {
        "id": "adamw_clipvalue",
        "description": "Gradient clipping via clipvalue.",
        "initial_variable": [2.0, -1.0, 0.5],
        "gradients": [[1.2, -0.8, 0.3], [-0.6, 0.4, -0.2]],
        "config": {"clipvalue": 0.1},
        "covers": ["clipvalue"],
    },
    {
        "id": "adamw_clipnorm",
        "description": "Gradient clipping via clipnorm.",
        "initial_variable": [1.5, -0.7, 0.2],
        "gradients": [[0.8, -0.6, 0.4], [0.2, -0.1, 0.05]],
        "config": {"clipnorm": 0.25},
        "covers": ["clipnorm"],
    },
    {
        "id": "adamw_global_clipnorm",
        "description": "Gradient clipping via global_clipnorm.",
        "initial_variable": [-1.0, 0.25, 1.75],
        "gradients": [[0.9, 0.9, -0.9], [0.3, -0.4, 0.2]],
        "config": {"global_clipnorm": 0.35},
        "covers": ["global_clipnorm"],
    },
    {
        "id": "adamw_loss_scale_factor",
        "description": "Gradient unscale before update with loss_scale_factor.",
        "initial_variable": [0.3, -0.6, 1.2],
        "gradients": [[12.8, -6.4, 3.2], [6.4, -3.2, 1.6]],
        "config": {"loss_scale_factor": 128.0},
        "covers": ["loss_scale_factor"],
    },
    {
        "id": "adamw_gradient_accumulation",
        "description": "Accumulate gradients over 2 steps before each update.",
        "initial_variable": [1.2, -0.4, 0.8],
        "gradients": [[0.2, -0.1, 0.05], [0.4, -0.2, 0.1], [0.6, -0.3, 0.15], [0.8, -0.4, 0.2]],
        "config": {"gradient_accumulation_steps": 2},
        "covers": ["gradient_accumulation_steps"],
    },
    {
        "id": "adamw_use_ema",
        "description": "EMA updates with custom ema_momentum.",
        "initial_variable": [0.9, -0.3, 0.1],
        "gradients": [[0.05, -0.02, 0.01], [0.08, -0.01, 0.03]],
        "config": {"use_ema": True, "ema_momentum": 0.5},
        "covers": ["use_ema", "ema_momentum"],
    },
    {
        "id": "adamw_ema_overwrite_frequency",
        "description": "EMA overwrite every 2 optimizer steps.",
        "initial_variable": [1.1, -1.1, 0.55],
        "gradients": [[0.2, -0.15, 0.1], [0.05, -0.02, 0.07]],
        "config": {"use_ema": True, "ema_momentum": 0.9, "ema_overwrite_frequency": 2},
        "covers": ["ema_overwrite_frequency"],
    },
]

MHA_REFERENCE = (
    "https://github.com/keras-team/keras/blob/v3.13.2/keras/src/layers/attention/"
    "multi_head_attention.py#L19"
)

MHA_DEFAULTS = {
    "num_heads": None,
    "key_dim": None,
    "value_dim": None,
    "dropout": 0.0,
    "use_bias": True,
    "output_shape": None,
    "attention_axes": None,
    "flash_attention": None,
    "kernel_initializer": "glorot_uniform",
    "bias_initializer": "zeros",
    "kernel_regularizer": None,
    "bias_regularizer": None,
    "activity_regularizer": None,
    "kernel_constraint": None,
    "bias_constraint": None,
    "seed": None,
}

MHA_PARAMETERS = [
    {"name": "num_heads", "default": MHA_DEFAULTS["num_heads"]},
    {"name": "key_dim", "default": MHA_DEFAULTS["key_dim"]},
    {"name": "value_dim", "default": MHA_DEFAULTS["value_dim"]},
    {"name": "dropout", "default": MHA_DEFAULTS["dropout"]},
    {"name": "use_bias", "default": MHA_DEFAULTS["use_bias"]},
    {"name": "output_shape", "default": MHA_DEFAULTS["output_shape"]},
    {"name": "attention_axes", "default": MHA_DEFAULTS["attention_axes"]},
    {"name": "flash_attention", "default": MHA_DEFAULTS["flash_attention"]},
    {"name": "kernel_initializer", "default": MHA_DEFAULTS["kernel_initializer"]},
    {"name": "bias_initializer", "default": MHA_DEFAULTS["bias_initializer"]},
    {"name": "kernel_regularizer", "default": MHA_DEFAULTS["kernel_regularizer"]},
    {"name": "bias_regularizer", "default": MHA_DEFAULTS["bias_regularizer"]},
    {"name": "activity_regularizer", "default": MHA_DEFAULTS["activity_regularizer"]},
    {"name": "kernel_constraint", "default": MHA_DEFAULTS["kernel_constraint"]},
    {"name": "bias_constraint", "default": MHA_DEFAULTS["bias_constraint"]},
    {"name": "seed", "default": MHA_DEFAULTS["seed"]},
]

MHA_CASES = [
    {
        "id": "mha_self_default",
        "description": "Self-attention with Keras defaults.",
        "query_shape": (2, 5, 8),
        "value_shape": (2, 5, 8),
        "config": {"num_heads": 2, "key_dim": 4},
        "kwargs": {},
        "attention_mask_kind": None,
        "covers": ["num_heads", "key_dim"],
    },
    {
        "id": "mha_cross_without_explicit_key",
        "description": "Cross-attention shape with implicit key=value.",
        "query_shape": (2, 4, 8),
        "value_shape": (2, 6, 8),
        "config": {"num_heads": 2, "key_dim": 4},
        "kwargs": {},
        "attention_mask_kind": None,
        "covers": [],
    },
    {
        "id": "mha_cross_with_explicit_key",
        "description": "Cross-attention with explicit key tensor.",
        "query_shape": (2, 3, 8),
        "value_shape": (2, 5, 8),
        "key_shape": (2, 5, 8),
        "config": {"num_heads": 2, "key_dim": 4},
        "kwargs": {},
        "attention_mask_kind": None,
        "covers": [],
    },
    {
        "id": "mha_value_dim",
        "description": "Custom value_dim different from key_dim.",
        "query_shape": (2, 4, 8),
        "value_shape": (2, 4, 8),
        "config": {"num_heads": 2, "key_dim": 3, "value_dim": 5},
        "kwargs": {},
        "attention_mask_kind": None,
        "covers": ["value_dim"],
    },
    {
        "id": "mha_use_bias_false",
        "description": "Bias disabled on all projections.",
        "query_shape": (2, 4, 8),
        "value_shape": (2, 4, 8),
        "config": {"num_heads": 2, "key_dim": 4, "use_bias": False},
        "kwargs": {},
        "attention_mask_kind": None,
        "covers": ["use_bias"],
    },
    {
        "id": "mha_output_shape_int",
        "description": "Output projection with integer output_shape.",
        "query_shape": (2, 4, 8),
        "value_shape": (2, 4, 8),
        "config": {"num_heads": 2, "key_dim": 4, "output_shape": 6},
        "kwargs": {},
        "attention_mask_kind": None,
        "covers": ["output_shape"],
    },
    {
        "id": "mha_output_shape_tuple",
        "description": "Output projection with tuple output_shape.",
        "query_shape": (2, 4, 8),
        "value_shape": (2, 4, 8),
        "config": {"num_heads": 2, "key_dim": 4, "output_shape": (4, 2)},
        "kwargs": {},
        "attention_mask_kind": None,
        "covers": ["output_shape"],
    },
    {
        "id": "mha_attention_axes_rank4",
        "description": "Rank-4 inputs with positive attention_axes.",
        "query_shape": (2, 3, 4, 8),
        "value_shape": (2, 3, 2, 8),
        "config": {"num_heads": 2, "key_dim": 4, "attention_axes": (2,)},
        "kwargs": {},
        "attention_mask_kind": None,
        "covers": ["attention_axes"],
    },
    {
        "id": "mha_attention_axes_rank5",
        "description": "Rank-5 inputs with 2D attention.",
        "query_shape": (2, 2, 3, 4, 8),
        "value_shape": (2, 2, 3, 2, 8),
        "config": {"num_heads": 2, "key_dim": 4, "attention_axes": (2, 3)},
        "kwargs": {},
        "attention_mask_kind": None,
        "covers": ["attention_axes"],
    },
    {
        "id": "mha_attention_mask_2d",
        "description": "2D attention mask broadcasting.",
        "query_shape": (2, 4, 8),
        "value_shape": (2, 4, 8),
        "config": {"num_heads": 2, "key_dim": 4},
        "kwargs": {},
        "attention_mask_kind": "2d",
        "covers": [],
    },
    {
        "id": "mha_attention_mask_3d",
        "description": "3D attention mask broadcasting.",
        "query_shape": (2, 3, 8),
        "value_shape": (2, 5, 8),
        "config": {"num_heads": 2, "key_dim": 4},
        "kwargs": {},
        "attention_mask_kind": "3d",
        "covers": [],
    },
    {
        "id": "mha_attention_mask_4d",
        "description": "4D attention mask with explicit head dim.",
        "query_shape": (2, 4, 8),
        "value_shape": (2, 4, 8),
        "config": {"num_heads": 2, "key_dim": 4},
        "kwargs": {},
        "attention_mask_kind": "4d",
        "covers": [],
    },
    {
        "id": "mha_use_causal_mask",
        "description": "Causal masking path.",
        "query_shape": (2, 5, 8),
        "value_shape": (2, 5, 8),
        "config": {"num_heads": 2, "key_dim": 4},
        "kwargs": {"use_causal_mask": True},
        "attention_mask_kind": None,
        "covers": [],
    },
    {
        "id": "mha_return_attention_scores",
        "description": "Return attention scores alongside output.",
        "query_shape": (2, 4, 8),
        "value_shape": (2, 4, 8),
        "config": {"num_heads": 2, "key_dim": 4},
        "kwargs": {"return_attention_scores": True},
        "attention_mask_kind": None,
        "covers": [],
    },
    {
        "id": "mha_dropout_inference",
        "description": "Dropout configured but deterministic inference mode.",
        "query_shape": (2, 4, 8),
        "value_shape": (2, 4, 8),
        "config": {"num_heads": 2, "key_dim": 4, "dropout": 0.25, "seed": 17},
        "kwargs": {"training": False},
        "attention_mask_kind": None,
        "covers": ["dropout", "seed"],
    },
    {
        "id": "mha_advanced_config",
        "description": "Initializer/regularizer/constraint configuration coverage.",
        "query_shape": (2, 4, 8),
        "value_shape": (2, 4, 8),
        "config": {
            "num_heads": 2,
            "key_dim": 4,
            "kernel_initializer": "he_uniform",
            "bias_initializer": "ones",
            "kernel_regularizer": "l2",
            "bias_regularizer": "l1",
            "activity_regularizer": "l2",
            "kernel_constraint": "max_norm",
            "bias_constraint": "non_neg",
        },
        "kwargs": {},
        "attention_mask_kind": None,
        "covers": [
            "kernel_initializer",
            "bias_initializer",
            "kernel_regularizer",
            "bias_regularizer",
            "activity_regularizer",
            "kernel_constraint",
            "bias_constraint",
        ],
    },
    {
        "id": "mha_flash_attention_false",
        "description": "Explicit flash_attention=False compatibility.",
        "query_shape": (2, 4, 8),
        "value_shape": (2, 4, 8),
        "config": {"num_heads": 2, "key_dim": 4, "flash_attention": False},
        "kwargs": {},
        "attention_mask_kind": None,
        "covers": ["flash_attention"],
    },
    {
        "id": "mha_combined_cross_scores_mask",
        "description": "Combined cross-attention, scores and 3D mask.",
        "query_shape": (2, 3, 8),
        "value_shape": (2, 5, 8),
        "key_shape": (2, 5, 8),
        "config": {"num_heads": 2, "key_dim": 3, "value_dim": 5, "output_shape": 7},
        "kwargs": {"return_attention_scores": True},
        "attention_mask_kind": "3d",
        "covers": ["value_dim", "output_shape"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate golden fixture from Keras metrics, AdamW and MultiHeadAttention."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output path for golden JSON (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def compute_expected(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true_2d = y_true.reshape(-1, 1)
    y_pred_2d = y_pred.reshape(-1, 1)

    f1_metric = keras.metrics.F1Score(name="f1", threshold=0.5, average="micro")
    auc_metric = keras.metrics.AUC(name="auc")

    f1_metric.update_state(y_true_2d, y_pred_2d)
    auc_metric.update_state(y_true_2d, y_pred_2d)

    return {
        "f1": float(f1_metric.result().numpy()),
        "auc": float(auc_metric.result().numpy()),
    }


def _f32(value: float) -> np.float32:
    return np.float32(value)


def _adamw_optimizer_iterations(internal_iterations: int, gradient_accumulation_steps: int | None) -> int:
    if gradient_accumulation_steps:
        return internal_iterations // gradient_accumulation_steps
    return internal_iterations


def _clip_by_norm(values: np.ndarray, clipnorm: float) -> np.ndarray:
    l2sum = np.sum(np.square(values, dtype=np.float32), dtype=np.float32)
    pred = l2sum > _f32(0.0)
    l2sum_safe = l2sum if pred else _f32(1.0)
    l2norm = _f32(np.sqrt(l2sum_safe)) if pred else l2sum
    numerator = values * _f32(clipnorm)
    denominator = max(l2norm, _f32(clipnorm))
    return numerator / denominator


def _clip_by_global_norm(values: np.ndarray, clipnorm: float) -> np.ndarray:
    use_norm = _f32(np.sqrt(np.sum(np.square(values, dtype=np.float32), dtype=np.float32)))
    if use_norm == _f32(0.0):
        scale_for_finite = _f32(1.0)
    else:
        scale_for_finite = _f32(clipnorm) * min(_f32(1.0) / use_norm, _f32(1.0) / _f32(clipnorm))
    scale = scale_for_finite + (use_norm - use_norm)
    return values * _f32(scale)


def _clip_gradient(values: np.ndarray, config: dict[str, object]) -> np.ndarray:
    clipnorm = config["clipnorm"]
    global_clipnorm = config["global_clipnorm"]
    clipvalue = config["clipvalue"]

    if clipnorm is not None and clipnorm > 0:
        return _clip_by_norm(values, float(clipnorm))
    if global_clipnorm is not None and global_clipnorm > 0:
        return _clip_by_global_norm(values, float(global_clipnorm))
    if clipvalue is not None and clipvalue > 0:
        clipvalue32 = _f32(float(clipvalue))
        return np.clip(values, -clipvalue32, clipvalue32)
    return values


def simulate_adamw_case(
    initial_variable: list[float], gradients: list[list[float]], config_overrides: dict[str, object]
) -> dict[str, object]:
    config = {**ADAMW_DEFAULTS, **config_overrides}

    variable = np.asarray(initial_variable, dtype=np.float32)
    gradients_f32 = [np.asarray(gradient, dtype=np.float32) for gradient in gradients]

    momentum = np.zeros_like(variable)
    velocity = np.zeros_like(variable)
    velocity_hat = np.zeros_like(variable) if config["amsgrad"] else None
    gradient_accumulator = (
        np.zeros_like(variable) if config["gradient_accumulation_steps"] is not None else None
    )
    ema = np.zeros_like(variable) if config["use_ema"] else None
    internal_iterations = 0

    for gradient_raw in gradients_f32:
        gradient = np.array(gradient_raw, dtype=np.float32, copy=True)

        loss_scale_factor = config["loss_scale_factor"]
        if loss_scale_factor is not None:
            gradient = gradient / _f32(float(loss_scale_factor))

        gradient_accumulation_steps = config["gradient_accumulation_steps"]
        should_update = True
        if gradient_accumulation_steps is not None:
            steps = int(gradient_accumulation_steps)
            should_update = ((internal_iterations + 1) % steps) == 0
            if should_update:
                gradient = (gradient + gradient_accumulator) / _f32(steps)
                gradient_accumulator = np.zeros_like(gradient_accumulator)
            else:
                gradient_accumulator = gradient_accumulator + gradient

        if should_update:
            gradient = _clip_gradient(gradient, config)

            learning_rate = _f32(float(config["learning_rate"]))
            weight_decay = _f32(float(config["weight_decay"]))
            variable = variable - variable * weight_decay * learning_rate

            optimizer_iterations = _adamw_optimizer_iterations(
                internal_iterations, config["gradient_accumulation_steps"]
            )
            local_step = _f32(optimizer_iterations + 1)
            beta_1 = _f32(float(config["beta_1"]))
            beta_2 = _f32(float(config["beta_2"]))
            beta_1_power = _f32(np.power(beta_1, local_step))
            beta_2_power = _f32(np.power(beta_2, local_step))
            alpha = learning_rate * _f32(np.sqrt(_f32(1.0) - beta_2_power)) / (_f32(1.0) - beta_1_power)

            momentum = momentum + (gradient - momentum) * (_f32(1.0) - beta_1)
            velocity = velocity + ((gradient * gradient) - velocity) * (_f32(1.0) - beta_2)

            source_velocity = velocity
            if config["amsgrad"]:
                velocity_hat = np.maximum(velocity_hat, velocity)
                source_velocity = velocity_hat

            epsilon = _f32(float(config["epsilon"]))
            variable = variable - (momentum * alpha) / (_f32(np.sqrt(source_velocity)) + epsilon)

        if config["use_ema"]:
            optimizer_iterations = _adamw_optimizer_iterations(
                internal_iterations, config["gradient_accumulation_steps"]
            )
            not_first_step = optimizer_iterations != 0
            ema_momentum = _f32(float(config["ema_momentum"])) if not_first_step else _f32(0.0)
            ema = ema_momentum * ema + (_f32(1.0) - ema_momentum) * variable

            ema_overwrite_frequency = config["ema_overwrite_frequency"]
            if ema_overwrite_frequency:
                overwrite = ((optimizer_iterations + 1) % int(ema_overwrite_frequency)) == 0
                if overwrite:
                    variable = np.array(ema, copy=True)

        internal_iterations += 1

    return {
        "final_variable": variable.astype(np.float32).tolist(),
        "final_internal_iterations": internal_iterations,
        "final_optimizer_iterations": _adamw_optimizer_iterations(
            internal_iterations, config["gradient_accumulation_steps"]
        ),
        "momentum": momentum.astype(np.float32).tolist(),
        "velocity": velocity.astype(np.float32).tolist(),
        "velocity_hat": velocity_hat.astype(np.float32).tolist() if velocity_hat is not None else None,
        "gradient_accumulator": (
            gradient_accumulator.astype(np.float32).tolist() if gradient_accumulator is not None else None
        ),
        "ema": ema.astype(np.float32).tolist() if ema is not None else None,
    }


def build_adamw_fixture() -> dict[str, object]:
    fixture_cases = []
    covered_parameters = set()

    for case in ADAMW_CASES:
        covered_parameters.update(case["covers"])
        result = simulate_adamw_case(
            initial_variable=case["initial_variable"],
            gradients=case["gradients"],
            config_overrides=case["config"],
        )
        config = {**ADAMW_DEFAULTS, **case["config"]}
        fixture_cases.append(
            {
                "id": case["id"],
                "description": case["description"],
                "initialVariable": np.asarray(case["initial_variable"], dtype=np.float32).tolist(),
                "gradients": [np.asarray(gradient, dtype=np.float32).tolist() for gradient in case["gradients"]],
                "config": config,
                "expected": {
                    "finalVariable": result["final_variable"],
                    "finalInternalIterations": result["final_internal_iterations"],
                    "finalOptimizerIterations": result["final_optimizer_iterations"],
                    "momentum": result["momentum"],
                    "velocity": result["velocity"],
                    "velocityHat": result["velocity_hat"],
                    "gradientAccumulator": result["gradient_accumulator"],
                    "ema": result["ema"],
                },
            }
        )

    return {
        "meta": {
            "reference": ADAMW_REFERENCE,
            "defaults": ADAMW_DEFAULTS,
            "parameters": ADAMW_PARAMETERS,
            "covered_parameters": sorted(covered_parameters),
        },
        "cases": fixture_cases,
    }


def _random_f32(rng: np.random.Generator, shape: tuple[int, ...], scale: float = 0.5) -> np.ndarray:
    return rng.normal(loc=0.0, scale=scale, size=shape).astype(np.float32)


def _ensure_mask_rows(mask: np.ndarray) -> np.ndarray:
    flat = mask.reshape((-1, mask.shape[-1]))
    for row in flat:
        if not np.any(row):
            row[0] = True
    return flat.reshape(mask.shape)


def _make_attention_mask(
    rng: np.random.Generator,
    kind: str | None,
    query_shape: tuple[int, ...],
    value_shape: tuple[int, ...],
    num_heads: int,
) -> np.ndarray | None:
    if kind is None:
        return None

    q_length = int(query_shape[1])
    v_length = int(value_shape[1])

    if kind == "2d":
        shape = (q_length, v_length)
    elif kind == "3d":
        shape = (int(query_shape[0]), q_length, v_length)
    elif kind == "4d":
        shape = (int(query_shape[0]), num_heads, q_length, v_length)
    else:
        raise ValueError(f"Unsupported attention mask kind: {kind}")

    mask = rng.uniform(size=shape) > 0.25
    return _ensure_mask_rows(mask)


def _serialize_optional_array(value: np.ndarray | None) -> list | None:
    if value is None:
        return None
    return value.tolist()


def build_mha_fixture() -> dict[str, object]:
    fixture_cases = []
    covered_parameters = set()

    for case_index, case in enumerate(MHA_CASES):
        keras.backend.clear_session()

        covered_parameters.update(case["covers"])
        rng = np.random.default_rng(202600 + case_index)

        query_shape = tuple(case["query_shape"])
        value_shape = tuple(case["value_shape"])
        key_shape = tuple(case["key_shape"]) if case.get("key_shape") is not None else None

        config = {**MHA_DEFAULTS, **case["config"]}
        kwargs = {"training": False, **case.get("kwargs", {})}
        call_kwargs = dict(kwargs)

        query = _random_f32(rng, query_shape)
        value = _random_f32(rng, value_shape)
        key = _random_f32(rng, key_shape) if key_shape is not None else None

        layer = keras.layers.MultiHeadAttention(**case["config"])
        if key is None:
            _ = layer(query=query, value=value, training=False)
        else:
            _ = layer(query=query, value=value, key=key, training=False)

        generated_weights = []
        for weight in layer.get_weights():
            generated_weights.append(_random_f32(rng, weight.shape, scale=0.35))
        layer.set_weights(generated_weights)

        attention_mask = _make_attention_mask(
            rng,
            case.get("attention_mask_kind"),
            query_shape,
            key_shape if key_shape is not None else value_shape,
            num_heads=int(config["num_heads"]),
        )

        if attention_mask is not None:
            call_kwargs["attention_mask"] = attention_mask

        if key is None:
            output = layer(query=query, value=value, **call_kwargs)
        else:
            output = layer(query=query, value=value, key=key, **call_kwargs)

        return_scores = bool(kwargs.get("return_attention_scores", False))
        if return_scores:
            output_tensor, score_tensor = output
            expected_output = np.asarray(output_tensor, dtype=np.float32)
            expected_scores = np.asarray(score_tensor, dtype=np.float32)
        else:
            expected_output = np.asarray(output, dtype=np.float32)
            expected_scores = None

        fixture_cases.append(
            {
                "id": case["id"],
                "description": case["description"],
                "query": query.astype(np.float32).tolist(),
                "value": value.astype(np.float32).tolist(),
                "key": _serialize_optional_array(key.astype(np.float32) if key is not None else None),
                "attentionMask": _serialize_optional_array(attention_mask),
                "config": config,
                "kwargs": kwargs,
                "weights": [weight.astype(np.float32).tolist() for weight in generated_weights],
                "expected": {
                    "output": expected_output.tolist(),
                    "attentionScores": (
                        expected_scores.tolist()
                        if expected_scores is not None
                        else None
                    ),
                },
            }
        )

    return {
        "meta": {
            "reference": MHA_REFERENCE,
            "runtime_keras_version": keras.__version__,
            "defaults": MHA_DEFAULTS,
            "parameters": MHA_PARAMETERS,
            "covered_parameters": sorted(covered_parameters),
            "notes": [
                "Golden values generated with local Keras runtime (Python 3.9).",
                "Semantic implementation target remains Keras v3.13.2 source reference.",
            ],
        },
        "cases": fixture_cases,
    }


def build_fixture() -> dict:
    fixture = {
        "meta": {
            "f1": {"name": "f1", "threshold": 0.5, "average": "micro"},
            "auc": {
                "name": "auc",
                "curve": "ROC",
                "summation_method": "interpolation",
                "num_thresholds": 200,
            },
            "epsilon": float(keras.backend.epsilon()),
        },
        "cases": [],
        "optimizerAdamW": build_adamw_fixture(),
        "layerMultiHeadAttention": build_mha_fixture(),
    }

    for case in CASES:
        y_true = np.asarray(case["yTrue"], dtype=np.float32)
        y_pred = np.asarray(case["yPred"], dtype=np.float32)
        expected = compute_expected(y_true, y_pred)
        fixture["cases"].append(
            {
                "id": case["id"],
                "yTrue": y_true.tolist(),
                "yPred": y_pred.tolist(),
                "expected": expected,
            }
        )

    return fixture


def main() -> None:
    args = parse_args()
    fixture = build_fixture()
    output_path = args.output.expanduser().resolve()
    output_path.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote golden fixture to {output_path}")


if __name__ == "__main__":
    main()
