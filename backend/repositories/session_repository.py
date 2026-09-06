"""
Session repository abstraction layer for chat sessions and message operations.
Supports both SQLite (SQLAlchemy) and MongoDB (Motor/Pydantic) database backends.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime

from config import DATABASE_TYPE


class BaseSessionRepository(ABC):
    """Abstract repository interface for chat session and message management."""

    @abstractmethod
    async def get_session(self, session_id: Any, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Fetch a session by ID, optionally validating user ownership."""
        pass

    @abstractmethod
    async def create_session(
        self,
        user_id: str,
        title: Optional[str] = None,
        agent_id: Optional[Any] = None,
        mode: str = "agent",
    ) -> Dict[str, Any]:
        """Create a new session."""
        pass

    @abstractmethod
    async def update_session(self, session_id: Any, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update session fields (e.g. title, updated_at)."""
        pass

    @abstractmethod
    async def delete_session(self, session_id: Any, user_id: str) -> bool:
        """Delete a session and associated messages/attachments."""
        pass

    @abstractmethod
    async def get_messages(self, session_id: Any, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Fetch messages for a session ordered by creation time."""
        pass

    @abstractmethod
    async def add_message(
        self,
        session_id: Any,
        role: str,
        content: str,
        name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Add a message to a chat session."""
        pass


class SQLiteSessionRepository(BaseSessionRepository):
    """SQLite / SQLAlchemy implementation of BaseSessionRepository."""

    def __init__(self, db: Any):
        self.db = db

    async def get_session(self, session_id: Any, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        from models import Session
        try:
            numeric_id = int(session_id)
        except (ValueError, TypeError):
            return None

        query = self.db.query(Session).filter(Session.id == numeric_id)
        if user_id is not None:
            query = query.filter(Session.user_id == user_id)
        session_obj = query.first()
        if not session_obj:
            return None
        return {
            "id": str(session_obj.id),
            "user_id": session_obj.user_id,
            "title": session_obj.title,
            "agent_id": session_obj.agent_id,
            "created_at": session_obj.created_at,
            "updated_at": session_obj.updated_at,
        }

    async def create_session(
        self,
        user_id: str,
        title: Optional[str] = None,
        agent_id: Optional[Any] = None,
        mode: str = "agent",
    ) -> Dict[str, Any]:
        from models import Session
        session_obj = Session(
            user_id=user_id,
            title=title or "New Session",
            agent_id=int(agent_id) if agent_id is not None else None,
            mode=mode,
        )
        self.db.add(session_obj)
        self.db.commit()
        self.db.refresh(session_obj)
        return {
            "id": str(session_obj.id),
            "user_id": session_obj.user_id,
            "title": session_obj.title,
            "agent_id": session_obj.agent_id,
            "created_at": session_obj.created_at,
            "updated_at": session_obj.updated_at,
        }

    async def update_session(self, session_id: Any, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        from models import Session
        try:
            numeric_id = int(session_id)
        except (ValueError, TypeError):
            return None

        session_obj = self.db.query(Session).filter(Session.id == numeric_id).first()
        if not session_obj:
            return None

        for key, value in updates.items():
            if hasattr(session_obj, key):
                setattr(session_obj, key, value)
        session_obj.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(session_obj)
        return {
            "id": str(session_obj.id),
            "user_id": session_obj.user_id,
            "title": session_obj.title,
            "agent_id": session_obj.agent_id,
            "created_at": session_obj.created_at,
            "updated_at": session_obj.updated_at,
        }

    async def delete_session(self, session_id: Any, user_id: str) -> bool:
        from models import Session, Message, FileAttachment
        try:
            numeric_id = int(session_id)
        except (ValueError, TypeError):
            return False

        session_obj = self.db.query(Session).filter(
            Session.id == numeric_id,
            Session.user_id == user_id,
        ).first()

        if not session_obj:
            return False

        self.db.query(Message).filter(Message.session_id == numeric_id).delete()
        self.db.query(FileAttachment).filter(FileAttachment.session_id == numeric_id).delete()
        self.db.delete(session_obj)
        self.db.commit()
        return True

    async def get_messages(self, session_id: Any, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        from models import Message
        try:
            numeric_id = int(session_id)
        except (ValueError, TypeError):
            return []

        messages = (
            self.db.query(Message)
            .filter(Message.session_id == numeric_id)
            .order_by(Message.created_at.asc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        return [
            {
                "id": str(m.id),
                "session_id": str(m.session_id),
                "role": m.role,
                "content": m.content,
                "name": m.name,
                "tool_call_id": m.tool_call_id,
                "tool_calls": m.tool_calls,
                "created_at": m.created_at,
            }
            for m in messages
        ]

    async def add_message(
        self,
        session_id: Any,
        role: str,
        content: str,
        name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        from models import Message
        try:
            numeric_id = int(session_id)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid session_id for SQLite: {session_id}")

        msg = Message(
            session_id=numeric_id,
            role=role,
            content=content,
            name=name,
            tool_call_id=tool_call_id,
            tool_calls=tool_calls,
        )
        self.db.add(msg)
        self.db.commit()
        self.db.refresh(msg)
        return {
            "id": str(msg.id),
            "session_id": str(msg.session_id),
            "role": msg.role,
            "content": msg.content,
            "name": msg.name,
            "tool_call_id": msg.tool_call_id,
            "tool_calls": msg.tool_calls,
            "created_at": msg.created_at,
        }


class MongoSessionRepository(BaseSessionRepository):
    """MongoDB / Motor implementation of BaseSessionRepository."""

    def __init__(self, mongo_db: Any):
        self.db = mongo_db

    async def get_session(self, session_id: Any, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        from models_mongo import SessionCollection
        doc = await SessionCollection.find_by_id(self.db, str(session_id))
        if not doc:
            return None
        if user_id is not None and doc.get("user_id") != user_id:
            return None
        doc["id"] = str(doc.get("_id"))
        return doc

    async def create_session(
        self,
        user_id: str,
        title: Optional[str] = None,
        agent_id: Optional[Any] = None,
        mode: str = "agent",
    ) -> Dict[str, Any]:
        from models_mongo import SessionCollection
        doc = await SessionCollection.create(
            self.db,
            {
                "user_id": user_id,
                "title": title or "New Session",
                "agent_id": str(agent_id) if agent_id is not None else None,
                "mode": mode,
            },
        )
        doc["id"] = str(doc.get("_id"))
        return doc

    async def update_session(self, session_id: Any, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        from models_mongo import SessionCollection
        doc = await SessionCollection.update(self.db, str(session_id), updates)
        if doc:
            doc["id"] = str(doc.get("_id"))
        return doc

    async def delete_session(self, session_id: Any, user_id: str) -> bool:
        from models_mongo import SessionCollection
        return await SessionCollection.delete(self.db, str(session_id), user_id)

    async def get_messages(self, session_id: Any, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        from models_mongo import MessageCollection
        docs = await MessageCollection.find_by_session(self.db, str(session_id), limit=limit, offset=offset)
        for doc in docs:
            doc["id"] = str(doc.get("_id"))
        return docs

    async def add_message(
        self,
        session_id: Any,
        role: str,
        content: str,
        name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        from models_mongo import MessageCollection
        doc = await MessageCollection.create(
            self.db,
            {
                "session_id": str(session_id),
                "role": role,
                "content": content,
                "name": name,
                "tool_call_id": tool_call_id,
                "tool_calls": tool_calls,
            },
        )
        doc["id"] = str(doc.get("_id"))
        return doc


def get_session_repository(db_handle: Any) -> BaseSessionRepository:
    """Factory helper returning SQLiteSessionRepository or MongoSessionRepository."""
    if DATABASE_TYPE == "mongo":
        return MongoSessionRepository(db_handle)
    return SQLiteSessionRepository(db_handle)
