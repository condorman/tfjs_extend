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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate golden fixture from Keras metrics.")
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
