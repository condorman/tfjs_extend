import { describe, expect, it } from 'vitest'
import * as tf from '@tensorflow/tfjs'
import { AdamWOptimizer } from './src/optimizer.js'

describe('AdamW optimizer for model.compile', () => {
  it('returns a tf.Optimizer instance with python defaults', () => {
    const optimizer = new AdamWOptimizer()

    expect(optimizer instanceof tf.Optimizer).toBe(true)
    expect(optimizer instanceof AdamWOptimizer).toBe(true)

    const config = optimizer.getConfig()
    expect(config.learning_rate).toBe(AdamWOptimizer.DEFAULTS.learning_rate)
    expect(config.weight_decay).toBe(AdamWOptimizer.DEFAULTS.weight_decay)
    expect(config.beta_1).toBe(AdamWOptimizer.DEFAULTS.beta_1)
    expect(config.beta_2).toBe(AdamWOptimizer.DEFAULTS.beta_2)
    expect(config.epsilon).toBe(AdamWOptimizer.DEFAULTS.epsilon)

    optimizer.dispose()
  })

  it('can be passed directly to model.compile', async () => {
    const model = tf.sequential()
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }))

    const optimizer = new AdamWOptimizer({
      learning_rate: 0.01,
      weight_decay: 0.02,
      beta_1: 0.8,
      beta_2: 0.95,
    })

    model.compile({
      optimizer,
      loss: 'meanSquaredError',
    })

    const x = tf.tensor2d([[0], [1], [2], [3]], [4, 1])
    const y = tf.tensor2d([[0], [1], [2], [3]], [4, 1])

    const history = await model.fit(x, y, {
      epochs: 2,
      batchSize: 2,
      verbose: 0,
    })

    expect(history.history.loss.length).toBe(2)
    expect(typeof history.history.loss[0]).toBe('number')
    expect(optimizer.iterations).toBe(4)

    x.dispose()
    y.dispose()
    model.dispose()
    optimizer.dispose()
  })
})
