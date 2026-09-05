import logging
from typing import Optional, Tuple
from fastapi import HTTPException
from sqlalchemy.orm import Session

from config import DATABASE_TYPE
from encryption import decrypt_api_key

if DATABASE_TYPE == "mongo":
    from database_mongo import get_database
    from models_mongo import UserSecretCollection, LLMProviderCollection

logger = logging.getLogger(__name__)


async def resolve_embedding_credentials(
    user_id: str,
    kb_config: dict,
    db: Optional[Session] = None
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Resolve dynamic embedding credentials (provider, api_key, model) for a user and KB configuration.

    Parameters:
      user_id: string ID of the owning user.
      kb_config: dict containing optional 'secret_id', 'embedding_provider', and 'embedding_model'.
      db: SQLAlchemy Session (used in SQLite mode).

    Returns:
      Tuple of (embedding_provider, decrypted_api_key, embedding_model).
    """
    provider = kb_config.get("embedding_provider") or "google"
    model = kb_config.get("embedding_model") or "gemini-embedding-2"
    secret_id = kb_config.get("secret_id")

    api_key = None

    if secret_id:
        if DATABASE_TYPE == "mongo":
            mongo_db = get_database()
            secret = await UserSecretCollection.find_by_id(mongo_db, str(secret_id))
            if secret and str(secret.get("user_id")) == str(user_id):
                api_key = decrypt_api_key(secret["encrypted_value"])
            else:
                # Fallback: check if secret_id references an LLMProvider
                prov = await LLMProviderCollection.find_by_id(mongo_db, str(secret_id))
                if prov and str(prov.get("user_id")) == str(user_id) and prov.get("api_key"):
                    api_key = decrypt_api_key(prov["api_key"])
                else:
                    logger.warning(
                        f"[resolve_embedding_credentials] secret_id '{secret_id}' specified for user '{user_id}' "
                        "was not found or does not belong to the user in UserSecrets or LLMProviders."
                    )
        else:
            if db is not None:
                from models import UserSecret, LLMProvider
                if str(secret_id).isdigit():
                    sec = db.query(UserSecret).filter(
                        UserSecret.id == int(secret_id),
                        UserSecret.user_id == int(user_id) if str(user_id).isdigit() else UserSecret.user_id == user_id
                    ).first()
                    if sec and sec.encrypted_value:
                        api_key = decrypt_api_key(sec.encrypted_value)
                    else:
                        prov = db.query(LLMProvider).filter(
                            LLMProvider.id == int(secret_id),
                            LLMProvider.user_id == int(user_id) if str(user_id).isdigit() else LLMProvider.user_id == user_id
                        ).first()
                        if prov and prov.api_key:
                            api_key = decrypt_api_key(prov.api_key)
                        else:
                            logger.warning(
                                f"[resolve_embedding_credentials] secret_id '{secret_id}' specified for user '{user_id}' "
                                "was not found or does not belong to the user in UserSecrets or LLMProviders."
                            )

    if not api_key:
        logger.info(
            f"[resolve_embedding_credentials] Secret resolution yielded no API key for user '{user_id}'. "
            f"Attempting fallback to active LLMProvider matching provider_type '{provider}'."
        )
        prov_str = (provider or "").lower()
        provider_match_types = [provider] if provider else []
        if prov_str in ("google", "gemini"):
            provider_match_types = ["google", "gemini"]
        elif prov_str in ("nvidia", "nvidia_nim"):
            provider_match_types = ["nvidia", "nvidia_nim"]

        if DATABASE_TYPE == "mongo":
            mongo_db = get_database()
            if mongo_db is not None:
                prov = await mongo_db["llm_providers"].find_one({
                    "user_id": str(user_id),
                    "provider_type": {"$in": provider_match_types},
                    "is_active": True
                })
                if prov and prov.get("api_key"):
                    api_key = decrypt_api_key(prov["api_key"])
        else:
            if db is not None:
                from models import LLMProvider
                prov = db.query(LLMProvider).filter(
                    LLMProvider.user_id == (int(user_id) if str(user_id).isdigit() else user_id),
                    LLMProvider.provider_type.in_(provider_match_types),
                    LLMProvider.is_active == True
                ).first()
                if prov and prov.api_key:
                    api_key = decrypt_api_key(prov.api_key)

    if not api_key:
        logger.warning(
            f"[resolve_embedding_credentials] No API key could be resolved for user '{user_id}' "
            f"and provider '{provider}'. Defaulting to 'dummy_embedding_key'. Vector operations may fail if live API is called."
        )
        api_key = "dummy_embedding_key"

    return provider, api_key, model
