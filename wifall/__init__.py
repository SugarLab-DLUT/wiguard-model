from typing import TypedDict
import toml


class Config(TypedDict):
    server: str


config: Config = toml.load('config.toml')
