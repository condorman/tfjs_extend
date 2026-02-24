# tfjs_extend

TensorFlow.js extensions with Keras-compatible behavior for:

- metrics: `auc`, `f1`
- optimizer: `AdamW`
- layers: `MultiHeadAttention`

## Install

```bash
npm install tfjs_extend @tensorflow/tfjs
```

## Metrics

Available exports:

- `auc(yTrue, yPred): tf.Scalar`
- `f1(yTrue, yPred): tf.Scalar`
- `clearMetricsTensorCache(): void`

Example with `model.compile`:

```js
import * as tf from '@tensorflow/tfjs'
import { auc, f1 } from 'tfjs_extend'

const model = tf.sequential()
model.add(tf.layers.dense({ units: 1, inputShape: [4], activation: 'sigmoid' }))

model.compile({
  optimizer: tf.train.adam(1e-3),
  loss: 'binaryCrossentropy',
  metrics: [auc, f1],
})
```

`auc`/`f1` use internal tensor caches for performance. If you switch backend or aggressively recreate runtimes, clear cache with:

```js
import { clearMetricsTensorCache } from 'tfjs_extend'

clearMetricsTensorCache()
```

## Optimizer (AdamW)

`AdamW` extends `tf.Optimizer` and can be passed directly to `model.compile`.

```js
import * as tf from '@tensorflow/tfjs'
import { AdamW } from 'tfjs_extend'

const model = tf.sequential()
model.add(tf.layers.dense({ units: 1, inputShape: [4] }))

const optimizer = new AdamW({
  learning_rate: 1e-3,
  weight_decay: 4e-3,
  beta_1: 0.9,
  beta_2: 0.999,
  epsilon: 1e-7,
})

model.compile({
  optimizer,
  loss: 'meanSquaredError',
})
```

Supported options include:

- `learning_rate`, `weight_decay`, `beta_1`, `beta_2`, `epsilon`
- `amsgrad`
- gradient clipping: `clipnorm` or `clipvalue` or `global_clipnorm` (mutually exclusive)
- `use_ema`, `ema_momentum`, `ema_overwrite_frequency`
- `loss_scale_factor`
- `gradient_accumulation_steps`

Both snake_case and camelCase aliases are accepted (for example `learning_rate` and `learningRate`).

### Custom training loop (functional API)

If you need manual tensor/state updates outside `model.compile`, you can use static helpers:

```js
import * as tf from '@tensorflow/tfjs'
import { AdamW } from 'tfjs_extend'

const config = AdamW.normalizeConfig({ learning_rate: 1e-3 })
let variable = tf.tensor1d([0.5, -0.5])
let state = AdamW.createState(variable, config)

const gradient = tf.tensor1d([0.1, -0.2])
const next = AdamW.step(variable, gradient, state, config)

variable.dispose()
AdamW.disposeState(state)
gradient.dispose()

variable = next.variable
state = next.state
```

## MultiHeadAttention

```js
import { multiHeadAttention } from 'tfjs_extend'

const layer = multiHeadAttention({
  num_heads: 2,
  key_dim: 4,
})
```

## Tests

```bash
npm test
```

## Benchmarks (WASM only)

The Playwright benchmark is hard-pinned to TensorFlow.js `wasm` backend and fails if backend initialization does not resolve to `wasm`.

Install Chromium for Playwright (first run only):

```bash
npx playwright install chromium
```

Run benchmark:

```bash
npm run bench:wasm
```

Benchmark files:

- `scripts/benchmarks/wasm.bench.mjs`
- `scripts/benchmarks/wasm.bench.html`

## Python setup (golden generation)

```bash
python3.13 -m venv .venv_golden
source .venv_golden/bin/activate
pip install -U pip
pip install -r requirements.in
```

Generate fixture:

```bash
./.venv_golden/bin/python generate_golden.py
```
