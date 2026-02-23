import { describe, expect, it } from 'vitest'
import * as tf from '@tensorflow/tfjs'
import { multiHeadAttention } from './src/layers.js'

describe('MultiHeadAttention integration with model.compile', () => {
  it('can train in a functional model with two inputs', async () => {
    const queryInput = tf.input({ shape: [4, 8], name: 'query' })
    const valueInput = tf.input({ shape: [4, 8], name: 'value' })

    const mhaLayer = multiHeadAttention({
      num_heads: 2,
      key_dim: 4,
      value_dim: 4,
      use_bias: true,
    })

    const mhaOutput = mhaLayer.apply([queryInput, valueInput])
    const pooled = tf.layers.globalAveragePooling1d().apply(mhaOutput)
    const prediction = tf.layers.dense({ units: 1 }).apply(pooled)

    const model = tf.model({ inputs: [queryInput, valueInput], outputs: prediction })

    model.compile({
      optimizer: tf.train.adam(0.01),
      loss: 'meanSquaredError',
    })

    const query = tf.randomNormal([8, 4, 8], 0, 1, 'float32', 11)
    const value = tf.randomNormal([8, 4, 8], 0, 1, 'float32', 13)
    const target = tf.randomNormal([8, 1], 0, 1, 'float32', 17)

    const history = await model.fit([query, value], target, {
      epochs: 2,
      batchSize: 4,
      verbose: 0,
    })

    expect(history.history.loss.length).toBe(2)
    expect(typeof history.history.loss[0]).toBe('number')
    expect(mhaLayer.trainableWeights.length).toBeGreaterThan(0)

    query.dispose()
    value.dispose()
    target.dispose()
    model.dispose()
  })
})
