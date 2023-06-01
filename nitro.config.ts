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
    apiUrl: '',
    corsUrl: '',
  },
})
