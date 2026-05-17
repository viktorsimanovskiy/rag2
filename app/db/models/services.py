# ============================================================
# File: app/db/models/services.py
# Purpose:
#   ORM model for the concrete public service registry.
#
# Notes:
#   - service_registry replaces the removed coarse measure_code layer;
#   - one row = one concrete service from Актуальный_приказ5.xlsx;
#   - document_registry must be bound to this table through service_key.
# ============================================================

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import Boolean, CheckConstraint, DateTime, Index, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


# ============================================================
# Helpers
# ============================================================

def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ============================================================
# Service registry
# ============================================================

class ServiceRegistry(Base):
    """
    Concrete public service registry.

    The registry is imported from Актуальный_приказ5.xlsx, sheet "Для ИИ".
    This table is the stable service identity layer for retrieval and runtime.
    It must not be mixed with the old coarse measure groups such as edv/subsidy.
    """

    __tablename__ = "service_registry"
    __table_args__ = (
        UniqueConstraint("service_key", name="uq_service_registry_service_key"),
        UniqueConstraint("raw_filename", name="uq_service_registry_raw_filename"),
        UniqueConstraint("cleaned_filename", name="uq_service_registry_cleaned_filename"),
        UniqueConstraint("frgu_1", name="uq_service_registry_frgu_1"),
        UniqueConstraint("frgu_3", name="uq_service_registry_frgu_3"),
        CheckConstraint("service_key <> ''", name="chk_service_registry_service_key_not_empty"),
        CheckConstraint("service_name_full <> ''", name="chk_service_registry_name_full_not_empty"),
        CheckConstraint("service_name_short <> ''", name="chk_service_registry_name_short_not_empty"),
        CheckConstraint("raw_filename <> ''", name="chk_service_registry_raw_filename_not_empty"),
        CheckConstraint("cleaned_filename <> ''", name="chk_service_registry_cleaned_filename_not_empty"),
        Index("idx_service_registry_name_short", "service_name_short"),
        Index("idx_service_registry_is_active", "is_active"),
    )

    service_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default=text("gen_random_uuid()"),
    )

    service_key: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    service_name_full: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    service_name_short: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    frgu_1: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    frgu_3: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    order_details: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    raw_filename: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    cleaned_filename: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    aliases_json: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
    )

    note: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=text("TRUE"),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        server_default=text("NOW()"),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        onupdate=utcnow,
        server_default=text("NOW()"),
    )

    documents = relationship(
        "DocumentRegistry",
        back_populates="service",
        lazy="selectin",
    )
