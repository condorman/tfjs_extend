import http from 'node:http'
import path from 'node:path'
import fs from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { chromium } from 'playwright'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const repoRoot = path.resolve(__dirname, '..', '..')

function median(values) {
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  if (sorted.length % 2 === 0) {
    return (sorted[mid - 1] + sorted[mid]) / 2
  }
  return sorted[mid]
}

async function createStaticServer(rootDir) {
  const server = http.createServer(async (req, res) => {
    try {
      const requestUrl = new URL(req.url, 'http://127.0.0.1')
      let pathname = decodeURIComponent(requestUrl.pathname)
      if (pathname === '/') {
        pathname = '/scripts/benchmarks/wasm.bench.html'
      }

      const filePath = path.join(rootDir, pathname)
      const normalized = path.normalize(filePath)
      if (!normalized.startsWith(path.normalize(rootDir))) {
        res.writeHead(403)
        res.end('Forbidden')
        return
      }

      const stat = await fs.stat(normalized)
      if (stat.isDirectory()) {
        res.writeHead(403)
        res.end('Directory listing disabled')
        return
      }

      const extension = path.extname(normalized)
      const contentType = {
        '.html': 'text/html; charset=utf-8',
        '.js': 'text/javascript; charset=utf-8',
        '.mjs': 'text/javascript; charset=utf-8',
        '.json': 'application/json; charset=utf-8',
      }[extension] ?? 'text/plain; charset=utf-8'

      const content = await fs.readFile(normalized)
      res.writeHead(200, { 'Content-Type': contentType })
      res.end(content)
    } catch {
      res.writeHead(404)
      res.end('Not found')
    }
  })

  await new Promise((resolve) => {
    server.listen(0, '127.0.0.1', resolve)
  })

  const address = server.address()
  const port = typeof address === 'object' && address ? address.port : 0
  return { server, port }
}

async function runBench(page, label, runFn, options = {}) {
  const warmup = options.warmup ?? 1
  const runs = options.runs ?? 5

  for (let i = 0; i < warmup; i += 1) {
    await runFn()
  }

  const samples = []
  for (let i = 0; i < runs; i += 1) {
    const t0 = Date.now()
    await runFn()
    const t1 = Date.now()
    samples.push(t1 - t0)
  }

  const result = {
    label,
    backend: 'wasm',
    medianMs: median(samples),
    samples,
  }
  console.log(JSON.stringify(result))
  return result
}

async function main() {
  const { server, port } = await createStaticServer(repoRoot)
  const baseUrl = `http://127.0.0.1:${port}`

  const browser = await chromium.launch({ headless: true })
  const page = await browser.newPage()

  try {
    await page.goto(`${baseUrl}/scripts/benchmarks/wasm.bench.html`, { waitUntil: 'load' })

    await page.evaluate(async () => {
      const tf = await import('@tensorflow/tfjs')
      const wasmBackend = await import('@tensorflow/tfjs-backend-wasm')
      const optimizerModule = await import('/src/optimizer.js')
      const metricsModule = await import('/src/metrics.js')

      if (typeof wasmBackend.setWasmPaths === 'function') {
        wasmBackend.setWasmPaths('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/')
      }

      await tf.ready()
      const backendOk = await tf.setBackend('wasm')
      if (!backendOk || tf.getBackend() !== 'wasm') {
        throw new Error(`Failed to initialize wasm backend. Current backend: ${tf.getBackend()}`)
      }

      window.__tf = tf
      window.__AdamW = optimizerModule.AdamW
      window.__auc = metricsModule.auc
      window.__f1 = metricsModule.f1
      window.__clearMetricsTensorCache = metricsModule.clearMetricsTensorCache
    })

    const backend = await page.evaluate(() => window.__tf.getBackend())
    if (backend !== 'wasm') {
      throw new Error(`Benchmark must run on wasm backend, got: ${backend}`)
    }

    const runOptimizer = async (config) => page.evaluate(async (payload) => {
      const tf = window.__tf
      const AdamW = window.__AdamW
      const size = 120000
      const steps = 50

      const weights = tf.variable(tf.randomNormal([size]))
      const grads = tf.randomNormal([size])
      const optimizer = new AdamW(payload)

      for (let i = 0; i < steps; i += 1) {
        optimizer.applyGradients({ [weights.name]: grads })
      }

      await tf.nextFrame()
      grads.dispose()
      weights.dispose()
      optimizer.dispose()
    }, config)

    await page.evaluate(() => {
      const tf = window.__tf
      if (window.__metricYTrue) {
        window.__metricYTrue.dispose()
        window.__metricYPred.dispose()
      }
      window.__metricYTrue = tf.cast(tf.greater(tf.randomUniform([50000]), 0.5), 'float32')
      window.__metricYPred = tf.randomUniform([50000])
    })

    const runAuc = () => page.evaluate(async () => {
      const tf = window.__tf
      const auc = window.__auc
      for (let i = 0; i < 10; i += 1) {
        const value = auc(window.__metricYTrue, window.__metricYPred)
        value.dispose()
      }
      await tf.nextFrame()
    })

    const runF1 = () => page.evaluate(async () => {
      const tf = window.__tf
      const f1 = window.__f1
      for (let i = 0; i < 120; i += 1) {
        const value = f1(window.__metricYTrue, window.__metricYPred)
        value.dispose()
      }
      await tf.nextFrame()
    })

    const optimizerDefault = await runBench(page, 'optimizer_default', () => runOptimizer({}))
    const optimizerGlobalClip = await runBench(page, 'optimizer_global_clipnorm', () => runOptimizer({ global_clipnorm: 1.0 }))
    const metricAuc = await runBench(page, 'metric_auc', () => runAuc())
    const metricF1 = await runBench(page, 'metric_f1', () => runF1())

    const end = await page.evaluate(() => {
      const tf = window.__tf
      window.__metricYTrue.dispose()
      window.__metricYPred.dispose()
      window.__clearMetricsTensorCache()
      return { backend: tf.getBackend(), numTensors: tf.memory().numTensors }
    })

    console.log(JSON.stringify({ end }))
    console.log(JSON.stringify({
      summary: {
        optimizerDefault,
        optimizerGlobalClip,
        metricAuc,
        metricF1,
      },
    }, null, 2))
  } finally {
    await browser.close()
    await new Promise((resolve, reject) => {
      server.close((error) => {
        if (error) {
          reject(error)
          return
        }
        resolve()
      })
    })
  }
}

main().catch((error) => {
  console.error(error?.stack ?? error)
  process.exit(1)
})
