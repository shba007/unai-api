from pydantic_settings import BaseSettings


class Config(BaseSettings):
    tensorflow_api_url: str = ""


config = Config()
