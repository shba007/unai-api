import { defineNitroConfig } from 'nitropack'

export default defineNitroConfig({
  imports: {
    imports: [
      { name: 'ofetch', from: 'ofetch' },
    ],
  },
  storage: {
    'db': {
      driver: 'fs',
      base: './data/db'
    }
  },
  runtimeConfig: {
    apiURL: process.env.TF_SERVING_URL,
    corsURL: process.env.CORS_URL,
  },
})
