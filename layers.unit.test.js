import { describe, expect, it } from 'vitest'
import * as tf from '@tensorflow/tfjs'
import { MultiHeadAttention, multiHeadAttention } from './src/layers.js'

function expectArrayCloseTo(actual, expected, digits = 5) {
  expect(actual.length).toBe(expected.length)
  for (let index = 0; index < expected.length; index += 1) {
    expect(actual[index]).toBeCloseTo(expected[index], digits)
  }
}

describe('MultiHeadAttention unit behavior', () => {
  it('throws explicit error when flash_attention=true', () => {
    expect(() => new MultiHeadAttention({
      num_heads: 2,
      key_dim: 4,
      flash_attention: true,
    })).toThrow(/flash_attention=true/)
  })

  it('supports python-style initializer/constraint identifiers', () => {
    const layer = new MultiHeadAttention({
      num_heads: 2,
      key_dim: 4,
      kernel_initializer: 'he_uniform',
      bias_initializer: 'ones',
      kernel_constraint: 'max_norm',
      bias_constraint: 'non_neg',
    })

    const query = tf.randomNormal([2, 4, 8])
    const value = tf.randomNormal([2, 4, 8])

    const output = layer.apply([query, value])
    expect(output instanceof tf.Tensor).toBe(true)

    output.dispose()
    query.dispose()
    value.dispose()
    layer.dispose()
  })

  it('matches outputs for equivalent positive and negative attention_axes', () => {
    const x = tf.randomUniform([2, 3, 8, 4], -1, 1, 'float32', 7)

    const mhaPos = new MultiHeadAttention({ num_heads: 2, key_dim: 4, attention_axes: 2 })
    const mhaNeg = new MultiHeadAttention({ num_heads: 2, key_dim: 4, attention_axes: -2 })

    const buildPos = mhaPos.apply([x, x])
    const buildNeg = mhaNeg.apply([x, x])
    buildPos.dispose()
    buildNeg.dispose()

    const weights = mhaPos.getWeights()
    mhaNeg.setWeights(weights)

    const [outputPos, scoresPos] = mhaPos.apply([x, x], { return_attention_scores: true })
    const [outputNeg, scoresNeg] = mhaNeg.apply([x, x], { return_attention_scores: true })

    expect(outputPos.shape).toEqual(outputNeg.shape)
    expect(scoresPos.shape).toEqual(scoresNeg.shape)

    expectArrayCloseTo(Array.from(outputPos.dataSync()), Array.from(outputNeg.dataSync()))
    expectArrayCloseTo(Array.from(scoresPos.dataSync()), Array.from(scoresNeg.dataSync()))

    outputPos.dispose()
    scoresPos.dispose()
    outputNeg.dispose()
    scoresNeg.dispose()
    x.dispose()
    mhaPos.dispose()
    mhaNeg.dispose()
  })

  it('broadcasts masks consistently across 2D/3D/4D forms', () => {
    const layer = new MultiHeadAttention({ num_heads: 2, key_dim: 4 })

    const x = tf.randomNormal([2, 5, 8], 0, 1, 'float32', 3)

    const baseMask = tf.linalg.bandPart(tf.ones([5, 5], 'bool'), -1, 0)
    const mask3d = tf.tile(tf.expandDims(baseMask, 0), [2, 1, 1])
    const mask4d = tf.tile(tf.expandDims(mask3d, 1), [1, 2, 1, 1])

    const out2d = layer.apply([x, x], { attention_mask: baseMask })
    const out3d = layer.apply([x, x], { attention_mask: mask3d })
    const out4d = layer.apply([x, x], { attention_mask: mask4d })

    expectArrayCloseTo(Array.from(out2d.dataSync()), Array.from(out3d.dataSync()))
    expectArrayCloseTo(Array.from(out2d.dataSync()), Array.from(out4d.dataSync()))

    out2d.dispose()
    out3d.dispose()
    out4d.dispose()
    x.dispose()
    baseMask.dispose()
    mask3d.dispose()
    mask4d.dispose()
    layer.dispose()
  })

  it('throws on invalid output_shape and shape mismatches between value and key', () => {
    expect(() => new MultiHeadAttention({
      num_heads: 2,
      key_dim: 4,
      output_shape: 8.5,
    })).toThrow(/output_shape/)

    const layer = new MultiHeadAttention({ num_heads: 2, key_dim: 2, value_dim: 2 })

    const query = tf.zeros([2, 4, 8])
    const value = tf.zeros([2, 2, 8])
    const key = tf.zeros([2, 1, 8])

    expect(() => layer.apply([query, value, key])).toThrow(/must be equal/)

    query.dispose()
    value.dispose()
    key.dispose()
  })

  it('returns attention scores in eager and symbolic modes', () => {
    const layer = new MultiHeadAttention({ num_heads: 2, key_dim: 4 })

    const query = tf.randomNormal([2, 4, 8], 0, 1, 'float32', 2)
    const value = tf.randomNormal([2, 4, 8], 0, 1, 'float32', 4)

    const eagerOut = layer.apply([query, value], { return_attention_scores: true })
    expect(Array.isArray(eagerOut)).toBe(true)
    expect(eagerOut).toHaveLength(2)

    const queryInput = tf.input({ shape: [4, 8] })
    const valueInput = tf.input({ shape: [4, 8] })
    const symbolicOut = layer.apply([queryInput, valueInput], { return_attention_scores: true })
    expect(Array.isArray(symbolicOut)).toBe(true)
    expect(symbolicOut).toHaveLength(2)

    const [eagerTensor, eagerScores] = eagerOut
    eagerTensor.dispose()
    eagerScores.dispose()
    query.dispose()
    value.dispose()
    layer.dispose()
  })

  it('factory returns a MultiHeadAttention instance', () => {
    const layer = multiHeadAttention({ num_heads: 2, key_dim: 4 })
    expect(layer instanceof MultiHeadAttention).toBe(true)
  })
})
