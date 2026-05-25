"""
Database initialization script for Project Oracle MVP.
Project Oracle MVP 数据库初始化脚本。

Running this script creates the `oracle_mvp.db` SQLite file in the project
root directory, with all four tables defined in db/models.py.
运行本脚本会在项目根目录生成 oracle_mvp.db 文件，自动建好蓝图中定义的四张表。

Usage / 用法:
    python init_db.py
"""

from pathlib import Path

from sqlalchemy import create_engine, inspect

from db.models import Base

# Database file lives in the project root alongside this script.
# 数据库文件放在项目根目录，和本脚本同级。
DB_PATH = Path(__file__).resolve().parent / "oracle_mvp.db"
DB_URL = f"sqlite:///{DB_PATH}"


def init_database() -> None:
    """
    Create all ORM tables in the SQLite database.
    在 SQLite 数据库中创建所有 ORM 定义的表。

    Why create_all? 为什么用 create_all？
    MVP 阶段直接用 create_all 建表即可，后续上线时可以切换到 Alembic 做 migration。
    """
    engine = create_engine(DB_URL, echo=False)
    Base.metadata.create_all(engine)

    # Verify: list all created tables.
    # 验证：打印所有已创建的表名。
    inspector = inspect(engine)
    tables = inspector.get_table_names()

    print(f"[Project Oracle] 数据库已创建: {DB_PATH}")
    print(f"[Project Oracle] Database created at: {DB_PATH}")
    print(f"[Project Oracle] 包含 {len(tables)} 张表 / {len(tables)} tables:")
    for t in tables:
        print(f"  - {t}")


if __name__ == "__main__":
    init_database()
