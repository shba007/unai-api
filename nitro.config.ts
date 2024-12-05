import { defineNitroConfig } from 'nitropack'

export default defineNitroConfig({
  routeRules: {
    '/**': { cors: true, headers: { 'access-control-allow-methods': 'GET,PUT,POST,DELETE' } },
  },
  imports: {
    imports: [{ name: 'ofetch', from: 'ofetch' }],
  },
  storage: {
    db: {
      driver: 'fs',
      base: './data/db',
    },
  },
  runtimeConfig: {
    apiUrl: '',
    corsUrl: '',
  },
})
