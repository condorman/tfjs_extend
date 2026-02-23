import { describe, expect, it } from 'vitest'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import * as tf from '@tensorflow/tfjs'
import {
  adamWStep,
  createAdamWState,
  disposeAdamWState,
  getAdamWOptimizerIterations,
  normalizeAdamWConfig,
} from './src/optimizer.js'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const fixturePath = path.join(__dirname, 'golden.json')
const fixture = JSON.parse(fs.readFileSync(fixturePath, 'utf8'))
const optimizerFixture = fixture.optimizerAdamW

function expectArrayCloseTo(actual, expected, digits = 6) {
  expect(actual.length).toBe(expected.length)
  for (let index = 0; index < expected.length; index += 1) {
    expect(actual[index]).toBeCloseTo(expected[index], digits)
  }
}

function expectTensorCloseTo(tensor, expected, digits = 6) {
  const actual = Array.from(tensor.dataSync())
  expectArrayCloseTo(actual, expected, digits)
}

describe('adamw fixture schema', () => {
  it('contains optimizer metadata and parameters', () => {
    expect(optimizerFixture).toBeDefined()
    expect(Array.isArray(optimizerFixture.cases)).toBe(true)
    expect(Array.isArray(optimizerFixture.meta.parameters)).toBe(true)
    expect(Array.isArray(optimizerFixture.meta.covered_parameters)).toBe(true)
  })

  it('covers every configurable AdamW parameter from the fixture', () => {
    const parameterNames = optimizerFixture.meta.parameters.map((item) => item.name)
    const covered = new Set(optimizerFixture.meta.covered_parameters)

    for (const parameterName of parameterNames) {
      expect(covered.has(parameterName)).toBe(true)
    }
  })
})

describe('adamw parity with generated golden', () => {
  for (const testCase of optimizerFixture.cases) {
    it(`matches expected for ${testCase.id}`, () => {
      const config = normalizeAdamWConfig(testCase.config)
      let variable = tf.tensor1d(testCase.initialVariable, 'float32')
      let state = createAdamWState(variable, config)

      for (const gradientValues of testCase.gradients) {
        const gradient = tf.tensor1d(gradientValues, 'float32')
        const next = adamWStep(variable, gradient, state, config)

        gradient.dispose()
        variable.dispose()
        disposeAdamWState(state)

        variable = next.variable
        state = next.state
      }

      expectTensorCloseTo(variable, testCase.expected.finalVariable)
      expect(state.iterations).toBe(testCase.expected.finalInternalIterations)

      const optimizerIterations = getAdamWOptimizerIterations(state.iterations, config)
      expect(optimizerIterations).toBe(testCase.expected.finalOptimizerIterations)

      expectTensorCloseTo(state.momentum, testCase.expected.momentum)
      expectTensorCloseTo(state.velocity, testCase.expected.velocity)

      if (testCase.expected.velocityHat == null) {
        expect(state.velocityHat).toBeNull()
      } else {
        expect(state.velocityHat).not.toBeNull()
        expectTensorCloseTo(state.velocityHat, testCase.expected.velocityHat)
      }

      if (testCase.expected.gradientAccumulator == null) {
        expect(state.gradientAccumulator).toBeNull()
      } else {
        expect(state.gradientAccumulator).not.toBeNull()
        expectTensorCloseTo(state.gradientAccumulator, testCase.expected.gradientAccumulator)
      }

      if (testCase.expected.ema == null) {
        expect(state.ema).toBeNull()
      } else {
        expect(state.ema).not.toBeNull()
        expectTensorCloseTo(state.ema, testCase.expected.ema)
      }

      variable.dispose()
      disposeAdamWState(state)
    })
  }
})
