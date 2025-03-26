from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Text, JSON, DateTime, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from utils.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    personas = relationship("Persona", back_populates="user")
    transaction_batches = relationship("TransactionBatch", back_populates="user")


class Persona(Base):
    __tablename__ = "personas"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String)
    description = Column(Text)
    config_json = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="personas")
    transaction_batches = relationship("TransactionBatch", back_populates="persona")


class TransactionBatch(Base):
    __tablename__ = "transaction_batches"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    persona_id = Column(Integer, ForeignKey("personas.id"))
    name = Column(String)
    summary_json = Column(JSON, nullable=True)
    preview_json = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="transaction_batches")
    persona = relationship("Persona", back_populates="transaction_batches")
    transactions = relationship("Transaction", back_populates="batch")


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(Integer, ForeignKey("transaction_batches.id"))
    
    # Transaction fields
    transaction_id = Column(String, index=True)
    booking_date_time = Column(DateTime(timezone=True))
    value_date_time = Column(DateTime(timezone=True))
    amount = Column(Float)
    currency = Column(String)
    creditor_name = Column(String)
    creditor_account_iban = Column(String)
    debtor_name = Column(String)
    debtor_account_iban = Column(String)
    remittance_information_unstructured = Column(Text)
    category = Column(String)
    
    edited = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    batch = relationship("TransactionBatch", back_populates="transactions") 