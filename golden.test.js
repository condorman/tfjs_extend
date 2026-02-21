import { describe, expect, it } from 'vitest'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import * as tf from '@tensorflow/tfjs'
import { auc, f1 } from './src/metrics.js'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const fixturePath = path.join(__dirname, 'golden.json')
const fixture = JSON.parse(fs.readFileSync(fixturePath, 'utf8'))

describe('golden fixture', () => {
  it('contains required schema', () => {
    expect(Array.isArray(fixture.cases)).toBe(true)
    expect(fixture.meta).toBeDefined()
    expect(fixture.meta.f1).toBeDefined()
    expect(fixture.meta.auc).toBeDefined()
  })
})

describe('metrics parity with keras golden', () => {
  for (const testCase of fixture.cases) {
    it(`matches expected values for ${testCase.id}`, () => {
      const f1Tensor = f1(testCase.yTrue, testCase.yPred)
      const aucTensor = auc(testCase.yTrue, testCase.yPred)

      const actualF1 = f1Tensor.arraySync()
      const actualAuc = aucTensor.arraySync()

      tf.dispose([f1Tensor, aucTensor])

      expect(actualF1).toBe(testCase.expected.f1)
      expect(actualAuc).toBe(testCase.expected.auc)
    })
  }
})

describe('input validation', () => {
  it('throws when yTrue and yPred lengths differ', () => {
    expect(() => f1([1, 0], [0.8])).toThrow(/same length/)
    expect(() => auc([1, 0], [0.8])).toThrow(/same length/)
  })
})
