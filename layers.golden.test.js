import { describe, expect, it } from 'vitest'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import * as tf from '@tensorflow/tfjs'
import { MultiHeadAttention } from './src/layers.js'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const fixturePath = path.join(__dirname, 'golden.json')
const fixture = JSON.parse(fs.readFileSync(fixturePath, 'utf8'))
const mhaFixture = fixture.layerMultiHeadAttention

function flatten(values) {
  return values.flat(Infinity)
}

function expectArrayCloseTo(actual, expected, digits = 5) {
  expect(actual.length).toBe(expected.length)
  for (let index = 0; index < expected.length; index += 1) {
    expect(actual[index]).toBeCloseTo(expected[index], digits)
  }
}

function disposeOutput(output) {
  if (Array.isArray(output)) {
    for (const item of output) {
      if (item instanceof tf.Tensor) {
        item.dispose()
      }
    }
    return
  }

  if (output instanceof tf.Tensor) {
    output.dispose()
  }
}

describe('multi head attention fixture schema', () => {
  it('contains metadata and cases', () => {
    expect(mhaFixture).toBeDefined()
    expect(Array.isArray(mhaFixture.cases)).toBe(true)
    expect(Array.isArray(mhaFixture.meta.parameters)).toBe(true)
    expect(Array.isArray(mhaFixture.meta.covered_parameters)).toBe(true)
    expect(typeof mhaFixture.meta.runtime_keras_version).toBe('string')
  })

  it('covers every configurable parameter declared in fixture metadata', () => {
    const parameterNames = mhaFixture.meta.parameters.map((item) => item.name)
    const covered = new Set(mhaFixture.meta.covered_parameters)

    for (const parameterName of parameterNames) {
      expect(covered.has(parameterName)).toBe(true)
    }
  })
})

describe('multi head attention parity with generated golden', () => {
  for (const testCase of mhaFixture.cases) {
    it(`matches expected for ${testCase.id}`, () => {
      const layer = new MultiHeadAttention(testCase.config)

      const query = tf.tensor(testCase.query, undefined, 'float32')
      const value = tf.tensor(testCase.value, undefined, 'float32')
      const key = testCase.key == null ? null : tf.tensor(testCase.key, undefined, 'float32')
      const attentionMask = testCase.attentionMask == null
        ? null
        : tf.tensor(testCase.attentionMask, undefined, 'bool')

      const inputs = key == null ? [query, value] : [query, value, key]

      const buildOutput = layer.apply(inputs, { training: false })
      disposeOutput(buildOutput)

      const weightTensors = testCase.weights.map((weight) => tf.tensor(weight, undefined, 'float32'))
      layer.setWeights(weightTensors)
      tf.dispose(weightTensors)

      const callKwargs = {
        ...testCase.kwargs,
      }
      if (attentionMask != null) {
        callKwargs.attention_mask = attentionMask
      }

      const actual = layer.apply(inputs, callKwargs)

      const expectedOutputFlat = flatten(testCase.expected.output)
      if (Array.isArray(actual)) {
        const [actualOutput, actualScores] = actual
        expectArrayCloseTo(Array.from(actualOutput.dataSync()), expectedOutputFlat)

        if (testCase.expected.attentionScores == null) {
          throw new Error(`Fixture mismatch: ${testCase.id} returned scores but expected.attentionScores is null`)
        }

        const expectedScoresFlat = flatten(testCase.expected.attentionScores)
        expectArrayCloseTo(Array.from(actualScores.dataSync()), expectedScoresFlat)
      } else {
        expectArrayCloseTo(Array.from(actual.dataSync()), expectedOutputFlat)
        expect(testCase.expected.attentionScores).toBeNull()
      }

      disposeOutput(actual)
      query.dispose()
      value.dispose()
      key?.dispose()
      attentionMask?.dispose()
      layer.dispose()
    })
  }
})
