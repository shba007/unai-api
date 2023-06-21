const config = useRuntimeConfig()

export default defineEventHandler((event) => {
  const origins: string[] = config.corsUrl

  const origin: string = event.node.req.headers.origin
  const selectedOrigin = origins.findIndex((o) => o == origin)
  // console.log({ origins, origin, selectedOrigin });

  setResponseHeaders(event, {
    "Access-Control-Allow-Methods": "GET,HEAD,PUT,PATCH,POST,DELETE",
    "Access-Control-Allow-Origin": selectedOrigin !== -1 ? origins[selectedOrigin] : origins[0],
    "Access-Control-Allow-Credentials": "true",
    "Access-Control-Allow-Headers": "*",
    "Access-Control-Expose-Headers": "*"
  })
  if (getMethod(event) === "OPTIONS") {
    event.node.res.statusCode = 204
    event.node.res.statusMessage = "No Content."
    return "OK"
  }
})
