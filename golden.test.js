import { afterAll, describe, expect, it } from 'vitest'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import * as tf from '@tensorflow/tfjs'
import { auc, f1 } from './src/metrics.js'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const fixturePath = path.join(__dirname, 'golden.json')
const fixture = JSON.parse(fs.readFileSync(fixturePath, 'utf8'))


const tensorCases = fixture.cases.map((testCase) => ({
  ...testCase,
  yTrueTensor: tf.tensor2d([testCase.yTrue], [1, testCase.yTrue.length], 'float32'),
  yPredTensor: tf.tensor2d([testCase.yPred], [1, testCase.yPred.length], 'float32'),
}))

afterAll(() => {
  tf.dispose(
    tensorCases.flatMap((testCase) => [testCase.yTrueTensor, testCase.yPredTensor])
  )
})

describe('golden fixture', () => {
  it('contains required schema', () => {
    expect(Array.isArray(fixture.cases)).toBe(true)
    expect(fixture.meta).toBeDefined()
    expect(fixture.meta.f1).toBeDefined()
    expect(fixture.meta.auc).toBeDefined()
  })
})

describe('metrics parity with keras golden', () => {
  for (const testCase of tensorCases) {
    it(`matches expected values for ${testCase.id}`, () => {
      const f1Tensor = f1(testCase.yTrueTensor, testCase.yPredTensor)
      const aucTensor = auc(testCase.yTrueTensor, testCase.yPredTensor)

      const actualF1 = f1Tensor.arraySync()
      const actualAuc = aucTensor.arraySync()

      tf.dispose([f1Tensor, aucTensor])

      expect(actualF1).toBe(testCase.expected.f1)
      expect(actualAuc).toBe(testCase.expected.auc)
    })
  }
})
