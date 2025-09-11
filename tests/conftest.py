# tests/conftest.py
import os
import pytest
from alembic.config import Config
from alembic import command
from sqlalchemy import delete
from app.models.civilization import Civilization
from app.models.event import Event
from app.models.source import Source, CivSourceLink, EventSourceLink

# Set auth token for tests (matches your app's header check)
TEST_TOKEN = "test-token"

@pytest.fixture(scope="session")
def db_url(tmp_path_factory):
    db_file = tmp_path_factory.mktemp("db") / "app.db"
    return f"sqlite:///{db_file}"

@pytest.fixture(scope="session", autouse=True)
def env_and_migrate(db_url):
    # Set env vars for the app (avoid monkeypatch to dodge scope issues)
    os.environ.setdefault("API_TOKEN", TEST_TOKEN)
    os.environ["DATABASE_URL"] = db_url

    # Run alembic migrations against the temp DB
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", db_url)
    command.upgrade(cfg, "head")
    yield

@pytest.fixture
def session():
    # DB session for DB-level tests
    from app.db.session import get_session
    with get_session() as s:
        yield s

@pytest.fixture(scope="session")
def client(env_and_migrate):
    # Seed once for API tests
    from scripts import ingest_seeds
    ingest_seeds.main()

    from fastapi.testclient import TestClient
    from app.main import app
    tc = TestClient(app)
    # Add default auth header so tests don't have to pass it each time
    tc.headers.update({"X-API-Key": TEST_TOKEN})
    return tc

@pytest.fixture(autouse=True, scope="function")
def _clean_db(session):
    """
    Ensure each test starts with empty tables. Order matters: link tables first.
    """
    # link tables first (FKs to events/civilizations/sources)
    session.exec(delete(EventSourceLink))
    session.exec(delete(CivSourceLink))

    # base tables
    session.exec(delete(Event))
    session.exec(delete(Civilization))
    session.exec(delete(Source))

    session.commit()


