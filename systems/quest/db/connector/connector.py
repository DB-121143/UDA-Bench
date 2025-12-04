"""Database connector module for OpenGauss database.

提供OpenGauss数据库连接功能，包括方言注册和连接字符串构建。
使用opengauss数据库，兼容postgreSQL语法
"""
import os
from urllib.parse import quote_plus
from sqlalchemy import create_engine
from configparser import ConfigParser
from sqlalchemy.dialects.postgresql import psycopg2
from sqlalchemy.dialects import registry
import logging

logger = logging.getLogger(__name__)


class OpenGaussDialect(psycopg2.PGDialect_psycopg2):
    """OpenGauss数据库方言

    继承自PostgreSQL psycopg2方言，添加OpenGauss特有的行为。
    """
    pass


# 注册OpenGauss方言
registry.register(OpenGaussDialect, 'opengauss', 'opengauss+psycopg2')


def create_opengauss_engine(db_config=None, config_file=None):
    """创建OpenGauss数据库引擎

    Args:
        db_config: 数据库配置字典，包含host, port, database, user, password
        config_file: 配置文件路径，如果提供则从文件读取配置

    Returns:
        SQLAlchemy Engine对象

    Raises:
        RuntimeError: 当配置无效或连接失败时
    """
    try:
        # 如果提供了配置文件，从文件读取配置
        if config_file and os.path.exists(config_file):
            parser = ConfigParser()
            parser.read(config_file)

            if 'opengauss' not in parser:
                raise RuntimeError(f"配置文件 {config_file} 中未找到 [opengauss] 部分")

            opengauss_config = parser['opengauss']
            db_config = {
                'host': opengauss_config['host'],
                'port': int(opengauss_config['port']),
                'database': opengauss_config['database'],
                'user': opengauss_config['user'],
                'password': opengauss_config['password']
            }
            logger.info(f"从配置文件 {config_file} 读取数据库配置")

        # 如果没有提供配置，使用默认配置
        if db_config is None:
            db_config = {
                "host": "127.0.0.1",
                "port": 18874,
                "database": "testdb",
                "user": "remote_user",
                "password": "dmE43-3654"
            }
            logger.info("使用默认数据库配置")

        # 验证必需的配置项
        required_keys = ['host', 'port', 'database', 'user', 'password']
        for key in required_keys:
            if key not in db_config:
                raise RuntimeError(f"数据库配置缺少必需项: {key}")

        # 对密码进行URL编码
        encoded_password = quote_plus(str(db_config['password']))

        # 构建OpenGauss连接字符串，使用注册的方言
        connection_string = (
            f"opengauss+psycopg2://{db_config['user']}:{encoded_password}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )

        # 创建数据库引擎
        engine = create_engine(connection_string)

        logger.info(f"成功创建OpenGauss数据库引擎，连接到 {db_config['host']}:{db_config['port']}/{db_config['database']}")
        return engine

    except Exception as e:
        logger.error(f"创建OpenGauss数据库引擎失败: {e}")
        raise RuntimeError(f"创建数据库连接失败: {e}")


def get_default_opengauss_engine():
    """获取默认配置的OpenGauss数据库引擎

    Returns:
        SQLAlchemy Engine对象
    """
    return create_opengauss_engine()

