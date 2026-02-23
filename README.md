# Keras extend function

- Extended metrics (AUC, F1)
- Extended optimizer (AdamW)
- Extended layers (MultiHeadAttention)

## Python setup (golden generation)

```bash
python3 -m venv .venv_golden
source .venv_golden/bin/activate
pip install -U pip
pip install -r requirements.in
```

Generate fixture:

```bash
./.venv_golden/bin/python generate_golden.py
```

## MultiHeadAttention

```js
import { multiHeadAttention } from 'tfjs_extend'

const layer = multiHeadAttention({
  num_heads: 2,
  key_dim: 4,
})
```

Notes:
- Golden values for `MultiHeadAttention` are generated with local `keras==3.10.0` runtime.
- Semantic source reference remains Keras `v3.13.2`.
- `flash_attention=true` is kept for config compatibility but is not supported in tfjs.
