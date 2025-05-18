from fastapi import FastAPI, Request, HTTPException, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import List, Optional
import numpy as np
import logging
import torch
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from models.wgan_gp import WGAN_GP
from utils.data_processor import TransactionDataProcessor
from utils.database import get_db
from models.database_models import (
    User, Persona, TransactionBatch, Transaction, 
    AuditLog, ActionType, TransactionExplanation, BatchExplanation
)
from models.explanation_service import ExplanationService

from passlib.context import CryptContext
from jose import JWTError, jwt

SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

PERSONAS = {
    "crypto_enthusiast": {"name": "Crypto Enthusiast", "description": "A tech-savvy individual who primarily invests in cryptocurrencies", "dataset_path": "testing_datasets/crypto_enthusiast.json"},
    "shopping_addict": {"name": "Shopping Addict", "description": "Someone who frequently shops online and at retail stores", "dataset_path": "testing_datasets/shopping_addict.json"},
    "gambling_addict": {"name": "Gambling Addict", "description": "A person with a regular gambling habit", "dataset_path": "testing_datasets/gambling_addict.json"},
    "money_mule": {"name": "Money Mule", "description": "Individual involved in moving illegally acquired money", "dataset_path": "testing_datasets/money_mule.json"}
}

models = {}
data_processor = TransactionDataProcessor()

def verify_password(plain_password, hashed_password):
    logger.debug(f"Verifying password. Plain: {plain_password}, Hash: {hashed_password}")
    result = pwd_context.verify(plain_password, hashed_password)
    logger.debug(f"Password verification result: {result}")
    return result

def get_password_hash(password):
    return pwd_context.hash(password)

def authenticate_user(db: Session, username: str, password: str):
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.password_hash):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(status_code=401, detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user

@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password", headers={"WWW-Authenticate": "Bearer"})
    access_token = create_access_token(data={"sub": user.username}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/register")
async def register_user(username: str, password: str, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(password)
    new_user = User(username=username, password_hash=hashed_password)
    db.add(new_user)
    db.flush()  # Flush to get the user ID
    
    # Update persona configurations to use S3 paths
    s3_personas = {
        "crypto_enthusiast": {
            "name": "Crypto Enthusiast",
            "description": "A tech-savvy individual who primarily invests in cryptocurrencies",
            "dataset_path": "s3://synthetic-personas-training-datasets/testing_datasets/crypto_enthusiast.json"
        },
        "shopping_addict": {
            "name": "Shopping Addict",
            "description": "Someone who frequently shops online and at retail stores",
            "dataset_path": "s3://synthetic-personas-training-datasets/testing_datasets/shopping_addict.json"
        },
        "gambling_addict": {
            "name": "Gambling Addict",
            "description": "A person with a regular gambling habit",
            "dataset_path": "s3://synthetic-personas-training-datasets/testing_datasets/gambling_addict.json"
        },
        "money_mule": {
            "name": "Money Mule",
            "description": "Individual involved in moving illegally acquired money",
            "dataset_path": "s3://synthetic-personas-training-datasets/testing_datasets/money_mule.json"
        }
    }
    
    for persona_id, persona_data in s3_personas.items():
        persona = Persona(
            user_id=new_user.id,
            name=persona_data["name"],
            description=persona_data["description"],
            config_json={"dataset_path": persona_data["dataset_path"]}
        )
        db.add(persona)
    
    db.commit()
    return {"message": "User registered successfully"}

@app.post("/ensure-personas")
async def ensure_personas(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Ensure all personas exist for the user with correct S3 config"""
    existing_personas = db.query(Persona).filter(Persona.user_id == user.id).all()
    existing_names = {p.name.lower() for p in existing_personas}
    
    s3_personas = {
        "crypto_enthusiast": {
            "name": "Crypto Enthusiast",
            "description": "A tech-savvy individual who primarily invests in cryptocurrencies",
            "dataset_path": "s3://synthetic-personas-training-datasets/testing_datasets/crypto_enthusiast.json"
        },
        "shopping_addict": {
            "name": "Shopping Addict",
            "description": "Someone who frequently shops online and at retail stores",
            "dataset_path": "s3://synthetic-personas-training-datasets/testing_datasets/shopping_addict.json"
        },
        "gambling_addict": {
            "name": "Gambling Addict",
            "description": "A person with a regular gambling habit",
            "dataset_path": "s3://synthetic-personas-training-datasets/testing_datasets/gambling_addict.json"
        },
        "money_mule": {
            "name": "Money Mule",
            "description": "Individual involved in moving illegally acquired money",
            "dataset_path": "s3://synthetic-personas-training-datasets/testing_datasets/money_mule.json"
        }
    }
    
    added = []
    for persona_id, data in s3_personas.items():
        if data["name"].lower() not in existing_names:
            new_persona = Persona(
                user_id=user.id,
                name=data["name"],
                description=data["description"],
                config_json={"dataset_path": data["dataset_path"]}
            )
            db.add(new_persona)
            added.append(data["name"])
    
    if added:
        db.commit()
        return {"message": f"Added missing personas: {', '.join(added)}"}
    
    # Update existing personas with correct S3 config
    updated = []
    for persona in existing_personas:
        for persona_id, data in s3_personas.items():
            if persona.name == data["name"]:
                if persona.config_json.get("dataset_path") != data["dataset_path"]:
                    persona.config_json = {"dataset_path": data["dataset_path"]}
                    updated.append(persona.name)
                break
    
    if updated:
        db.commit()
        return {"message": f"Updated personas with S3 paths: {', '.join(updated)}"}
    
    return {"message": "All personas exist and are up to date"}

@app.get("/personas")
async def get_personas(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    personas = db.query(Persona).filter(Persona.user_id == user.id).all()
    return {
        "personas": [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description
            } for p in personas
        ]
    }

@app.get("/batches")
async def get_batches(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    batches = (
        db.query(TransactionBatch)
        .filter(TransactionBatch.user_id == user.id)
        .order_by(TransactionBatch.created_at.desc())
        .all()
    )

    result = []
    for batch in batches:
        persona = db.query(Persona).filter(Persona.id == batch.persona_id).first()
        result.append({
            "id": batch.id,
            "name": batch.name,
            "persona_id": batch.persona_id,
            "persona_name": persona.name if persona else "Unknown",
            "created_at": batch.created_at.isoformat(),
            "transaction_count": len(batch.transactions) if batch.transactions else 0,
            "preview": batch.preview_json,
            "months": batch.months
        })

    return {"batches": result}

@app.get("/batches/{batch_id}")
async def get_batch(batch_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get a specific batch and its transactions"""
    batch = (
        db.query(TransactionBatch)
        .filter(TransactionBatch.id == batch_id, TransactionBatch.user_id == user.id)
        .first()
    )
    
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    # Get the persona name
    persona = db.query(Persona).filter(Persona.id == batch.persona_id).first()
    
    # Get all transactions for this batch
    transactions = (
        db.query(Transaction)
        .filter(Transaction.batch_id == batch_id)
        .order_by(Transaction.booking_date_time)
        .all()
    )
    
    # Convert transactions to the expected format
    formatted_transactions = []
    for tx in transactions:
        formatted_transactions.append({
            "transactionId": tx.transaction_id,
            "bookingDateTime": tx.booking_date_time.isoformat(),
            "valueDateTime": tx.value_date_time.isoformat(),
            "transactionAmount": {
                "amount": f"{tx.amount:.2f}",
                "currency": tx.currency
            },
            "creditorName": tx.creditor_name,
            "creditorAccount": {
                "iban": tx.creditor_account_iban
            },
            "debtorName": tx.debtor_name,
            "debtorAccount": {
                "iban": tx.debtor_account_iban
            },
            "remittanceInformationUnstructured": tx.remittance_information_unstructured,
            "category": tx.category
        })
    
    return {
        "id": batch.id,
        "name": batch.name,
        "persona_id": batch.persona_id,
        "persona_name": persona.name if persona else "Unknown",
        "created_at": batch.created_at.isoformat(),
        "transaction_count": len(transactions),
        "preview": batch.preview_json,
        "transactions": formatted_transactions,
        "months": batch.months
    }

@app.post("/generate/{persona_id}")
async def generate_transactions(
    persona_id: int, 
    months: int = 3, 
    batch_name: str = None,
    user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    persona = db.query(Persona).filter(Persona.id == persona_id, Persona.user_id == user.id).first()
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")
    
    dataset_path = persona.config_json.get("dataset_path")
    if not dataset_path:
        # For distribution-only personas, use a default dataset based on the categories
        if persona.config_json.get("custom_distribution"):
            # Use shopping_addict as default base dataset since it has a good variety of transaction types
            dataset_path = "s3://synthetic-personas-training-datasets/testing_datasets/shopping_addict.json"
            logger.info(f"Using default dataset for distribution-only persona: {dataset_path}")
        else:
            raise HTTPException(status_code=500, detail="No dataset path configured")
    
    try:
        # Get or create model key based on dataset path and persona
        model_key = f"persona_{persona_id}_{hash(dataset_path)}"
        model = models.get(model_key)
        
        if not model:
            X, C = data_processor.load_data(dataset_path)
            input_dim = X.shape[1]  # Number of numerical features
            condition_dim = C.shape[1]  # Number of categories
            tensor_X, tensor_C = torch.FloatTensor(X), torch.FloatTensor(C)
            
            # Create model with optimized parameters
            # Note: Generator expects input_dim + condition_dim for the first layer
            model = WGAN_GP(
                input_dim=input_dim,  # Numerical features
                output_dim=input_dim,  # Output same size as input
                condition_dim=condition_dim,  # Number of categories
                device=data_processor.device
            )
            
            # Minimal training for speed
            n_epochs = 15  # Further reduced from 25
            batch_size = 256  # Increased from 128
            n_batches = len(X) // batch_size
            
            # Train with larger batches and fewer iterations
            for epoch in range(n_epochs):
                total_loss = 0
                for i in range(0, len(X), batch_size):
                    batch_X = tensor_X[i:i + batch_size]
                    batch_C = tensor_C[i:i + batch_size]
                    stats = model.train_step(batch_X, batch_C)
                    total_loss += stats['g_loss']
                
                if epoch % 5 == 0:
                    avg_loss = total_loss / n_batches
                    logger.info(f"Epoch {epoch}: avg_loss = {avg_loss:.4f}")
            
            models[model_key] = model

        # Generate data
        n_samples = 50 * months
        categories = data_processor.category_columns
        
        # Get target distribution for the persona
        target_distribution = None
        if persona.config_json.get("custom_distribution"):
            # Use custom distribution if available
            target_distribution = {
                cat: {"min": pct, "max": pct}
                for cat, pct in persona.config_json["custom_distribution"].items()
            }
        else:
            # Fall back to default distribution
            target_distribution = data_processor.ensure_category_distribution(persona.name)
        
        if not target_distribution:
            chosen_categories = random.choices(list(categories), k=n_samples)
        else:
            category_weights = {
                cat: (rng["min"] + rng["max"]) / 2 for cat, rng in target_distribution.items()
            }
            total_weight = sum(category_weights.values())
            category_probs = {cat: w / total_weight for cat, w in category_weights.items()}
            
            chosen_categories = np.random.choice(
                list(category_probs.keys()),
                size=n_samples,
                p=list(category_probs.values())
            )

        # Generate in larger batches
        batch_size = 100  # Increased from 50
        n_batches = (n_samples + batch_size - 1) // batch_size
        all_generated_data = []
        
        # Pre-compute category indices for faster lookup
        category_indices = {}
        for cat in set(chosen_categories):
            # Handle both prefixed and unprefixed category names
            if cat.startswith('category_'):
                category_name = cat
            else:
                category_name = f"category_{cat}"
            
            try:
                category_indices[cat] = list(categories).index(category_name)
            except ValueError:
                # If not found with prefix, try without prefix
                try:
                    category_indices[cat] = list(categories).index(cat)
                except ValueError:
                    logger.error(f"Could not find category index for {cat}")
                    raise ValueError(f"Invalid category: {cat}")
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_categories = chosen_categories[start_idx:end_idx]
            
            condition_matrix = np.zeros((len(batch_categories), len(categories)))
            for j, cat in enumerate(batch_categories):
                idx = category_indices.get(cat)
                if idx is not None:
                    condition_matrix[j, idx] = 1
                else:
                    condition_matrix[j, random.randrange(len(categories))] = 1
            
            batch_generated = model.generate(len(batch_categories), condition_matrix)
            batch_combined = np.hstack([batch_generated, condition_matrix])
            all_generated_data.append(batch_combined)
        
        combined_data = np.vstack(all_generated_data)
        transactions = data_processor.inverse_transform(combined_data)
        
        # Bulk insert transactions
        default_name = f"Batch {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        batch = TransactionBatch(
            user_id=user.id,
            persona_id=persona_id,
            name=batch_name if batch_name else default_name,
            preview_json={"count": len(transactions)},
            months=months  # Store the months value
        )
        db.add(batch)
        db.flush()
        
        # Prepare all transactions at once
        db_transactions = []
        for tx in transactions:
            # Parse the date string - handle both formats
            try:
                booking_date = datetime.strptime(tx["bookingDateTime"], "%d/%m/%Y %H:%M:%S")
            except ValueError:
                try:
                    booking_date = datetime.fromisoformat(tx["bookingDateTime"])
                except ValueError:
                    logger.error(f"Invalid date format: {tx['bookingDateTime']}")
                    raise HTTPException(status_code=500, detail=f"Invalid date format: {tx['bookingDateTime']}")
            
            value_date = booking_date  # Use same date for value_date
            
            db_transactions.append(
                Transaction(
                    batch_id=batch.id,
                    transaction_id=tx["transactionId"],
                    booking_date_time=booking_date,
                    value_date_time=value_date,
                    amount=float(tx["transactionAmount"]["amount"]),
                    currency=tx["transactionAmount"]["currency"],
                    creditor_name=tx["creditorName"],
                    creditor_account_iban=tx["creditorAccount"]["iban"],
                    debtor_name=tx["debtorName"],
                    debtor_account_iban=tx["debtorAccount"]["iban"],
                    remittance_information_unstructured=tx["remittanceInformationUnstructured"],
                    category=tx["category"]
                )
            )
        
        # Bulk insert all transactions
        db.bulk_save_objects(db_transactions)
        db.commit()
        
        # Generate feature importances and explanations
        feature_importances = []
        for tx in transactions:
            tx_tensor, condition_tensor = data_processor.transaction_to_tensor(tx)
            with torch.no_grad():
                _ = model.generator(tx_tensor, condition_tensor)
                importance = model.generator.get_feature_importance()
                # Convert importance tensor to dictionary format
                importance_dict = {
                    'amount': float(importance[0][0]),
                    'day_of_month': float(importance[0][1])
                }
                # Add category importances
                for i, cat in enumerate(data_processor.category_columns):
                    category_name = cat.replace('category_', '')
                    importance_dict[category_name] = float(importance[0][i + 2])  # Start at index 2 since we have 2 numerical features
                feature_importances.append(importance_dict)
        
        # Generate explanations using the explanation service
        explanation_service = ExplanationService(db)
        explanation_service.process_batch(batch, feature_importances)
        
        # Create audit log for batch generation
        audit_log = AuditLog(
            user_id=user.id,
            action_type=ActionType.BATCH_GENERATED,
            entity_type="batch",
            entity_id=str(batch.id),
            details={
                "persona_id": persona_id,
                "months": months,
                "transaction_count": len(transactions),
                "explanations_generated": True
            }
        )
        db.add(audit_log)
        db.commit()
        
        return {
            "count": len(transactions),
            "transactions": transactions,
            "batch_id": batch.id
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error generating transactions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating transactions: {str(e)}")

@app.post("/update-personas")
async def update_personas(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Update existing personas with correct config_json"""
    updated = []
    for persona in user.personas:
        for persona_id, data in PERSONAS.items():
            if persona.name == data["name"]:
                persona.config_json = {"dataset_path": data["dataset_path"]}
                updated.append(persona.name)
                break
    
    if updated:
        db.commit()
        return {"message": f"Updated personas: {', '.join(updated)}"}
    
    return {"message": "No personas needed updating"}

@app.delete("/batches/{batch_id}")
async def delete_batch(
    batch_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Find the batch and verify ownership
    batch = db.query(TransactionBatch).filter(
        TransactionBatch.id == batch_id,
        TransactionBatch.user_id == user.id
    ).first()
    
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    try:
        # Create audit log before deletion
        audit_log = AuditLog(
            user_id=user.id,
            action_type=ActionType.BATCH_DELETED,
            entity_type="batch",
            entity_id=str(batch_id),
            details={
                "batch_name": batch.name,
                "persona_id": batch.persona_id,
                "transaction_count": len(batch.transactions)
            }
        )
        db.add(audit_log)
        
        # Delete associated transactions first
        db.query(Transaction).filter(Transaction.batch_id == batch_id).delete()
        
        # Delete the batch
        db.delete(batch)
        db.commit()
        
        return {"message": "Batch deleted successfully"}
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting batch: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting batch: {str(e)}")

@app.delete("/transactions/{transaction_id}")
async def delete_transaction(
    transaction_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Find the transaction and verify ownership through batch
    transaction = (
        db.query(Transaction)
        .join(TransactionBatch)
        .filter(
            Transaction.transaction_id == transaction_id,
            TransactionBatch.user_id == user.id
        )
        .first()
    )
    
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    try:
        # Create audit log before deletion
        audit_log = AuditLog(
            user_id=user.id,
            action_type=ActionType.TRANSACTION_DELETED,
            entity_type="transaction",
            entity_id=transaction_id,
            details={
                "batch_id": transaction.batch_id,
                "amount": transaction.amount,
                "category": transaction.category,
                "description": transaction.remittance_information_unstructured,
                "creditor_name": transaction.creditor_name
            }
        )
        db.add(audit_log)
        
        # Delete the transaction
        db.delete(transaction)
        db.commit()
        
        return {"message": "Transaction deleted successfully"}
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting transaction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting transaction: {str(e)}")

@app.patch("/batches/{batch_id}")
async def update_batch(
    batch_id: int,
    name: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Find the batch and verify ownership
    batch = db.query(TransactionBatch).filter(
        TransactionBatch.id == batch_id,
        TransactionBatch.user_id == user.id
    ).first()
    
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    try:
        old_name = batch.name
        # Update the batch name
        batch.name = name
        
        # Create audit log for the name change
        audit_log = AuditLog(
            user_id=user.id,
            action_type=ActionType.BATCH_NAME_EDITED,
            entity_type="batch",
            entity_id=str(batch_id),
            details={
                "old_name": old_name,
                "new_name": name
            }
        )
        db.add(audit_log)
        db.commit()
        
        return {"message": "Batch updated successfully", "name": name}
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating batch: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating batch: {str(e)}")

@app.get("/batches/{batch_id}/download/{format}")
async def download_batch(
    batch_id: int,
    format: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if format not in ["csv", "json", "excel"]:
        raise HTTPException(status_code=400, detail="Invalid format. Must be csv, json, or excel")
    
    # Find the batch and verify ownership
    batch = (
        db.query(TransactionBatch)
        .filter(TransactionBatch.id == batch_id, TransactionBatch.user_id == user.id)
        .first()
    )
    
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    try:
        # Get all transactions for this batch
        transactions = (
            db.query(Transaction)
            .filter(Transaction.batch_id == batch_id)
            .order_by(Transaction.booking_date_time)
            .all()
        )
        
        # Format transactions for export
        formatted_transactions = [{
            "transaction_id": tx.transaction_id,
            "booking_date_time": tx.booking_date_time.isoformat(),
            "value_date_time": tx.value_date_time.isoformat(),
            "amount": f"{tx.amount:.2f}",
            "currency": tx.currency,
            "creditor_name": tx.creditor_name,
            "creditor_account_iban": tx.creditor_account_iban,
            "debtor_name": tx.debtor_name,
            "debtor_account_iban": tx.debtor_account_iban,
            "description": tx.remittance_information_unstructured,
            "category": tx.category,
            "edited": tx.edited
        } for tx in transactions]
        
        # Create audit log for the download
        action_type = getattr(ActionType, f"BATCH_DOWNLOADED_{format.upper()}")
        audit_log = AuditLog(
            user_id=user.id,
            action_type=action_type,
            entity_type="batch",
            entity_id=str(batch_id),
            details={
                "format": format,
                "transaction_count": len(transactions),
                "batch_name": batch.name
            }
        )
        db.add(audit_log)
        db.commit()
        
        # Generate the appropriate format
        if format == "json":
            return formatted_transactions
        elif format == "csv":
            import csv
            from io import StringIO
            
            output = StringIO()
            writer = csv.DictWriter(output, fieldnames=formatted_transactions[0].keys())
            writer.writeheader()
            writer.writerows(formatted_transactions)
            
            return Response(
                content=output.getvalue(),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f'attachment; filename="{batch.name}_{datetime.now().strftime("%Y%m%d")}.csv"'
                }
            )
        else:  # excel
            import pandas as pd
            from io import BytesIO
            
            df = pd.DataFrame(formatted_transactions)
            output = BytesIO()
            df.to_excel(output, index=False, engine='openpyxl')
            output.seek(0)
            
            return Response(
                content=output.getvalue(),
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={
                    "Content-Disposition": f'attachment; filename="{batch.name}_{datetime.now().strftime("%Y%m%d")}.xlsx"'
                }
            )
            
    except Exception as e:
        db.rollback()
        logger.error(f"Error downloading batch: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error downloading batch: {str(e)}")

@app.patch("/personas/{persona_id}/distribution")
async def update_persona_distribution(
    persona_id: int,
    distribution: dict,
    batch_id: Optional[int] = None,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    persona = db.query(Persona).filter(
        Persona.id == persona_id,
        Persona.user_id == user.id
    ).first()
    
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")
    
    try:
        # Get the useForTraining flag from the request body
        use_for_training = distribution.pop("useForTraining", True)
        
        # Validate distribution values
        try:
            # Convert each value to float explicitly and log any errors
            validated_distribution = {}
            for category, value in distribution.items():
                try:
                    float_value = float(value)
                    validated_distribution[category] = float_value
                except (TypeError, ValueError) as e:
                    logger.error(f"Error converting value for {category}: {value} (type: {type(value)})")
                    raise ValueError(f"Invalid number for category {category}: {value}")
            
            total = sum(validated_distribution.values())
            if not (0.99 <= total <= 1.01):  # Allow small rounding errors
                raise ValueError(f"Distribution percentages must sum to 100% (got {total * 100}%)")
            
            distribution = validated_distribution  # Use the validated distribution
            
        except (TypeError, ValueError) as e:
            logger.error(f"Error in distribution validation: {str(e)}")
            raise ValueError("Distribution values must be numbers")
        
        # Store original distribution for audit
        original_distribution = persona.config_json.get("custom_distribution", {}) if persona.config_json else {}
        
        # Update persona config
        config = persona.config_json or {}
        config["custom_distribution"] = distribution
        persona.config_json = config
        
        # Create audit log for distribution update
        audit_log = AuditLog(
            user_id=user.id,
            action_type=ActionType.DISTRIBUTION_UPDATED,
            entity_type="persona",
            entity_id=str(persona_id),
            details={
                "new_distribution": distribution,
                "original_distribution": original_distribution,
                "use_for_training": use_for_training,
                "batch_id": batch_id
            }
        )
        db.add(audit_log)
        
        # If useForTraining is True, add to training data
        if use_for_training:
            training_data = TrainingData(
                user_id=user.id,
                data_type="distribution_update",
                data={
                    "persona_id": persona_id,
                    "original_distribution": original_distribution,
                    "new_distribution": distribution,
                    "batch_id": batch_id
                }
            )
            db.add(training_data)
        
        if batch_id:
            # Regenerate batch with new distribution
            # ... existing batch regeneration code ...
            pass
        
        db.commit()
        return {"message": "Distribution updated successfully", "batch_regenerated": batch_id is not None}
        
    except ValueError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating distribution: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating distribution: {str(e)}")

@app.post("/personas")
async def create_persona(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        body = await request.json()
        name = body.get("name")
        distribution = body.get("distribution")
        save_for_training = body.get("save_for_training", False)
        
        if not name or not distribution:
            raise HTTPException(status_code=400, detail="Name and distribution are required")
        
        # Validate the distribution
        try:
            data_processor.validate_custom_distribution(distribution)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Create new persona
        new_persona = Persona(
            user_id=user.id,
            name=name,
            description=f"Custom persona with defined distribution",
            config_json={
                "custom_distribution": distribution,
                "use_for_training": save_for_training
            }
        )
        
        db.add(new_persona)
        db.commit()
        
        return {"message": "Persona created successfully", "id": new_persona.id}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating persona: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating persona: {str(e)}")

@app.post("/personas/dataset")
async def create_persona_with_dataset(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        body = await request.json()
        name = body.get("name")
        description = body.get("description", "")
        dataset = body.get("dataset")
        
        if not name or not dataset:
            raise HTTPException(status_code=400, detail="Name and dataset are required")
        
        # Upload dataset to S3
        try:
            s3_url = data_processor.upload_dataset_to_s3(dataset, user.username, name)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error uploading dataset: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error uploading dataset: {str(e)}")
        
        # Create new persona
        new_persona = Persona(
            user_id=user.id,
            name=name,
            description=description or f"Custom persona with uploaded dataset",
            config_json={
                "dataset_path": s3_url,
                "use_for_training": True  # Always use uploaded datasets for training
            }
        )
        
        db.add(new_persona)
        db.commit()
        
        return {"message": "Persona created successfully", "id": new_persona.id}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating persona: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating persona: {str(e)}")

@app.patch("/transactions/{transaction_id}")
async def update_transaction(
    transaction_id: str,
    transaction_update: dict,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Find the transaction and verify ownership through batch
    transaction = (
        db.query(Transaction)
        .join(TransactionBatch)
        .filter(
            Transaction.transaction_id == transaction_id,
            TransactionBatch.user_id == user.id
        )
        .first()
    )
    
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    try:
        # Store original values for audit log
        original_values = {
            "amount": transaction.amount,
            "category": transaction.category,
            "description": transaction.remittance_information_unstructured,
            "creditor_name": transaction.creditor_name
        }
        
        # Update allowed fields
        if "transactionAmount" in transaction_update:
            transaction.amount = float(transaction_update["transactionAmount"]["amount"])
        
        if "remittanceInformationUnstructured" in transaction_update:
            transaction.remittance_information_unstructured = transaction_update["remittanceInformationUnstructured"]
        
        if "category" in transaction_update:
            transaction.category = transaction_update["category"]
        
        if "creditorName" in transaction_update:
            transaction.creditor_name = transaction_update["creditorName"]
        
        transaction.edited = True
        
        # Get the useForTraining flag
        use_for_training = transaction_update.get("useForTraining", True)
        
        # Create audit log for the edit
        audit_log = AuditLog(
            user_id=user.id,
            action_type=ActionType.TRANSACTION_EDITED,
            entity_type="transaction",
            entity_id=transaction_id,
            details={
                "batch_id": transaction.batch_id,
                "original_values": original_values,
                "new_values": {
                    "amount": transaction.amount,
                    "category": transaction.category,
                    "description": transaction.remittance_information_unstructured,
                    "creditor_name": transaction.creditor_name
                },
                "use_for_training": use_for_training
            }
        )
        db.add(audit_log)
        
        # If useForTraining is True, update the training data
        if use_for_training:
            # Add the transaction to the training dataset
            training_data = TrainingData(
                transaction_id=transaction_id,
                batch_id=transaction.batch_id,
                user_id=user.id,
                data_type="transaction_edit",
                data={
                    "amount": transaction.amount,
                    "category": transaction.category,
                    "description": transaction.remittance_information_unstructured,
                    "creditor_name": transaction.creditor_name,
                    "original_values": original_values
                }
            )
            db.add(training_data)
        
        db.commit()
        
        return {
            "message": "Transaction updated successfully",
            "transaction": {
                "transactionId": transaction.transaction_id,
                "bookingDateTime": transaction.booking_date_time.isoformat(),
                "valueDateTime": transaction.value_date_time.isoformat(),
                "transactionAmount": {
                    "amount": f"{transaction.amount:.2f}",
                    "currency": transaction.currency
                },
                "creditorName": transaction.creditor_name,
                "creditorAccount": {
                    "iban": transaction.creditor_account_iban
                },
                "debtorName": transaction.debtor_name,
                "debtorAccount": {
                    "iban": transaction.debtor_account_iban
                },
                "remittanceInformationUnstructured": transaction.remittance_information_unstructured,
                "category": transaction.category,
                "edited": transaction.edited
            }
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating transaction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating transaction: {str(e)}")

@app.get("/audit-logs")
async def get_audit_logs(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 100,
    offset: int = 0,
    action_type: Optional[str] = None,
    entity_type: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None
):
    query = db.query(AuditLog).filter(AuditLog.user_id == user.id)
    
    if action_type:
        query = query.filter(AuditLog.action_type == ActionType[action_type])
    if entity_type:
        query = query.filter(AuditLog.entity_type == entity_type)
    if from_date:
        query = query.filter(AuditLog.timestamp >= datetime.fromisoformat(from_date))
    if to_date:
        query = query.filter(AuditLog.timestamp <= datetime.fromisoformat(to_date))
    
    total = query.count()
    logs = query.order_by(AuditLog.timestamp.desc()).offset(offset).limit(limit).all()
    
    return {
        "total": total,
        "logs": [{
            "id": log.id,
            "action_type": log.action_type.value,
            "timestamp": log.timestamp.isoformat(),
            "entity_type": log.entity_type,
            "entity_id": log.entity_id,
            "details": log.details
        } for log in logs]
    }

@app.get("/batches/{batch_id}/explanation")
async def get_batch_explanation(
    batch_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get explanation for a batch of transactions."""
    # Find the batch and verify ownership
    batch = db.query(TransactionBatch).filter(
        TransactionBatch.id == batch_id,
        TransactionBatch.user_id == user.id
    ).first()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    explanation = db.query(BatchExplanation).filter(
        BatchExplanation.batch_id == batch_id
    ).first()
    
    if not explanation:
        raise HTTPException(status_code=404, detail="No explanation found for this batch")
    
    return {
        "batch_id": batch_id,
        "distribution_explanation": explanation.distribution_explanation,
        "temporal_patterns": explanation.temporal_patterns,
        "amount_patterns": explanation.amount_patterns,
        "anomalies": explanation.anomalies,
        "summary_text": explanation.summary_text
    }

@app.get("/transactions/{transaction_id}/explanation")
async def get_transaction_explanation(
    transaction_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get explanation for a specific transaction."""
    # Find the transaction and verify ownership through batch
    transaction = db.query(Transaction).filter(Transaction.transaction_id == transaction_id).first()
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    batch = db.query(TransactionBatch).filter(
        TransactionBatch.id == transaction.batch_id,
        TransactionBatch.user_id == user.id
    ).first()
    if not batch:
        raise HTTPException(status_code=403, detail="Not authorized to access this transaction")
    
    explanation = db.query(TransactionExplanation).filter(
        TransactionExplanation.transaction_id == transaction_id
    ).first()
    
    if not explanation:
        raise HTTPException(status_code=404, detail="No explanation found for this transaction")
    
    return {
        "transaction_id": transaction_id,
        "feature_importance": explanation.feature_importance,
        "applied_patterns": explanation.applied_patterns,
        "explanation_text": explanation.explanation_text,
        "confidence_score": explanation.confidence_score,
        "meta_info": explanation.meta_info
    }

@app.get("/debug/check-admin")
async def check_admin(db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == "admin").first()
    if user:
        return {
            "found": True,
            "id": user.id,
            "username": user.username,
            "password_hash": user.password_hash,
            "created_at": user.created_at.isoformat() if user.created_at else None
        }
    return {"found": False}

@app.post("/debug/recreate-admin")
async def recreate_admin(db: Session = Depends(get_db)):
    # Delete existing admin if exists
    db.query(User).filter(User.username == "admin").delete()
    
    # Create new admin user with fresh hash
    password = "admin"
    hashed_password = pwd_context.hash(password)
    
    new_user = User(
        username="admin",
        password_hash=hashed_password
    )
    
    db.add(new_user)
    db.commit()
    
    # Verify the password works
    verification = verify_password("admin", hashed_password)
    
    return {
        "message": "Admin user recreated",
        "id": new_user.id,
        "username": new_user.username,
        "password_hash": hashed_password,
        "verification_test": verification
    }

def transaction_to_tensor(self, tx):
    """Convert a transaction to a tensor format suitable for the model."""
    # Extract numerical features
    try:
        if isinstance(tx, dict):
            amount = float(tx['transactionAmount']['amount'])
        else:
            amount = float(tx.amount)
    except (KeyError, AttributeError):
        logger.error(f"Invalid transaction format: {tx}")
        raise ValueError("Invalid transaction format - missing amount field")

    # Extract temporal features
    try:
        if isinstance(tx, dict):
            booking_date = pd.to_datetime(tx['bookingDateTime'])
        else:
            booking_date = pd.to_datetime(tx.booking_date_time)
    except (KeyError, AttributeError):
        logger.error(f"Invalid transaction format: {tx}")
        raise ValueError("Invalid transaction format - missing booking date field")

    day_of_month = booking_date.day / 31.0
    day_of_week = booking_date.dayofweek / 6.0
    
    # Create condition vector for category
    try:
        if isinstance(tx, dict):
            category = tx['category']
        else:
            category = tx.category
    except (KeyError, AttributeError):
        logger.error(f"Invalid transaction format: {tx}")
        raise ValueError("Invalid transaction format - missing category field")

    category_vector = np.zeros(len(self.categorical_columns))
    try:
        if category.startswith('category_'):
            category_name = category
        else:
            category_name = f"category_{category}"
        category_idx = list(self.categorical_columns).index(category_name)
        category_vector[category_idx] = 1
    except ValueError:
        # If category not found, use a random category
        category_vector[random.randrange(len(self.categorical_columns))] = 1
    
    # Combine features
    features = np.array([amount, day_of_month, day_of_week])
    
    # Normalize numerical features
    if not hasattr(self, '_feature_scaler'):
        self._feature_scaler = MinMaxScaler()
        self._feature_scaler.fit(np.array([[0, 0, 0], [10000, 1, 1]]))
    features = self._feature_scaler.transform(features.reshape(1, -1))[0]
    
    # Create tensors on CPU first
    input_tensor = torch.FloatTensor(features).unsqueeze(0)
    condition_tensor = torch.FloatTensor(category_vector).unsqueeze(0)
    
    # Move tensors to the correct device
    input_tensor = input_tensor.to(self.device)
    condition_tensor = condition_tensor.to(self.device)
    
    # Synchronize if using MPS device
    if self.device.type == 'mps':
        torch.mps.synchronize()
    
    return input_tensor, condition_tensor
