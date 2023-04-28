import os

from dotenv import load_dotenv
import uvicorn

load_dotenv()

PRESET = os.getenv("PRESET")
PRESET = "deploy" if PRESET == None else PRESET
HOST = os.getenv("HOST")
HOST = "127.0.0.1" if HOST == None else HOST
PORT = os.getenv("PORT")
PORT = 8000 if PORT == None else int(PORT)
RELOAD = PRESET == "dev"
LOG_LEVEL = "warning" if PRESET == "deploy" else "debug"


if __name__ == "__main__":
    uvicorn.run("main:app", reload=RELOAD, host=HOST, port=PORT, log_level=LOG_LEVEL)
