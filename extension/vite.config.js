import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv } from 'vite'
import { readFileSync } from 'node:fs'
import { DEFAULT_API_URL } from './api-config.js'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', 'VITE_')
  const raw = env.VITE_HIBRIA_API_URL || DEFAULT_API_URL
  const url = new URL(raw)
  if (!['https:', 'http:'].includes(url.protocol) || url.username || url.password ||
      url.search || url.hash || (url.pathname !== '/' && url.pathname !== '')) {
    throw new Error('VITE_HIBRIA_API_URL deve conter apenas a origem HTTP(S) da API.')
  }
  const apiUrl = url.origin
  return {
    define: { 'import.meta.env.VITE_HIBRIA_API_URL': JSON.stringify(apiUrl) },
    plugins: [react(), {
      name: 'hibria-api-config',
      generateBundle() {
        const manifest = JSON.parse(readFileSync(new URL('./manifest.json', import.meta.url), 'utf8'))
        manifest.host_permissions = [apiUrl + '/*']
        this.emitFile({ type: 'asset', fileName: 'manifest.json', source: JSON.stringify(manifest, null, 2) })
        this.emitFile({ type: 'asset', fileName: 'api-config.js',
          source: 'globalThis.HIBRIA_API_URL = ' + JSON.stringify(apiUrl) + ';\n' })
      },
    }],
  }
})
