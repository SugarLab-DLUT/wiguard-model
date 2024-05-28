from typing import TypedDict, cast
import toml


class Config(TypedDict):
    server: str
    init_cmd: str
    log_cmd: str
    ping_cmd: str
    model: str


config: Config = cast(Config, toml.load('config.toml'))
