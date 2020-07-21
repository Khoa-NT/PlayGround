import hydra
from omegaconf import DictConfig
import logging
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

class DBConnection:
    def connect(self) -> None:
        pass


class MySQLConnection(DBConnection):
    def __init__(self, host: str, user: str, password: str) -> None:
        self.host = host
        self.user = user
        self.password = password

    def connect(self) -> None:
        print(
            f"MySQL connecting to {self.host} with user={self.user} and password={self.password}"
        )


class PostgreSQLConnection(DBConnection):
    def __init__(self, host: str, user: str, password: str, database: str) -> None:
        self.host = host
        self.user = user
        self.password = password
        self.database = database

    def connect(self) -> None:
        print(
            f"PostgreSQL connecting to {self.host} with user={self.user} "
            f"and password={self.password} and database={self.database}"
        )

class test_class():
    def __init__(self, cfg):
        self.cfg = cfg

    def connect(self) -> None:
        print(self.cfg.pretty())

@hydra.main(config_path="conf", config_name="config")
def my_app(cfg):
    connection = hydra.utils.instantiate(cfg.db)
    connection.connect()

    agent = hydra.utils.instantiate(cfg.agent, cfg)
    agent.connect()


if __name__ == "__main__":
    my_app()