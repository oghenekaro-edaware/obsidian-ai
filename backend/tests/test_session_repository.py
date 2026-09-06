import pytest
from unittest.mock import MagicMock, AsyncMock
from repositories.session_repository import (
    BaseSessionRepository,
    SQLiteSessionRepository,
    MongoSessionRepository,
    get_session_repository,
)


@pytest.mark.asyncio
async def test_sqlite_session_repository_crud():
    mock_db = MagicMock()
    mock_session = MagicMock()
    mock_session.id = 1
    mock_session.user_id = "user123"
    mock_session.title = "Test Session"
    mock_session.agent_id = None
    mock_session.created_at = "2025-01-01"
    mock_session.updated_at = "2025-01-01"

    # Filter mock chaining
    filter_mock = MagicMock()
    filter_mock.filter.return_value = filter_mock
    filter_mock.first.return_value = mock_session
    filter_mock.all.return_value = []

    mock_db.query.return_value = filter_mock

    repo = SQLiteSessionRepository(mock_db)

    # Get session
    session = await repo.get_session(1, user_id="user123")
    assert session is not None
    assert session["id"] == "1"
    assert session["user_id"] == "user123"

    # Invalid numeric ID
    invalid_session = await repo.get_session("not_an_int")
    assert invalid_session is None


@pytest.mark.asyncio
async def test_mongo_session_repository_crud(monkeypatch):
    mock_mongo_db = MagicMock()

    mock_doc = {
        "_id": "507f1f77bcf86cd799439011",
        "user_id": "user123",
        "title": "Mongo Session",
        "agent_id": None,
        "mode": "agent",
    }

    find_by_id_mock = AsyncMock(return_value=mock_doc)
    create_mock = AsyncMock(return_value=mock_doc.copy())
    delete_mock = AsyncMock(return_value=True)

    monkeypatch.setattr("models_mongo.SessionCollection.find_by_id", find_by_id_mock)
    monkeypatch.setattr("models_mongo.SessionCollection.create", create_mock)
    monkeypatch.setattr("models_mongo.SessionCollection.delete", delete_mock)

    repo = MongoSessionRepository(mock_mongo_db)

    session = await repo.get_session("507f1f77bcf86cd799439011", user_id="user123")
    assert session is not None
    assert session["id"] == "507f1f77bcf86cd799439011"

    created = await repo.create_session("user123", title="Mongo Session")
    assert created["id"] == "507f1f77bcf86cd799439011"

    deleted = await repo.delete_session("507f1f77bcf86cd799439011", "user123")
    assert deleted is True


def test_factory_helper(monkeypatch):
    monkeypatch.setattr("repositories.session_repository.DATABASE_TYPE", "sqlite")
    repo_sqlite = get_session_repository(MagicMock())
    assert isinstance(repo_sqlite, SQLiteSessionRepository)

    monkeypatch.setattr("repositories.session_repository.DATABASE_TYPE", "mongo")
    repo_mongo = get_session_repository(MagicMock())
    assert isinstance(repo_mongo, MongoSessionRepository)
