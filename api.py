from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import List, Optional
import numpy as np
import logging
import torch
import random

from models.wgan_gp import WGAN_GP
from utils.data_processor import TransactionDataProcessor
from utils.database import get_db
from models.database_models import User, Persona, TransactionBatch, Transaction

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    return pwd_context.verify(plain_password, hashed_password)

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
    
    for persona_id, persona_data in PERSONAS.items():
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
    """Ensure all personas exist for the user with correct config"""
    existing_personas = db.query(Persona).filter(Persona.user_id == user.id).all()
    existing_names = {p.name.lower() for p in existing_personas}
    
    added = []
    for persona_id, data in PERSONAS.items():
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
    
    # Update existing personas with correct config
    updated = []
    for persona in existing_personas:
        for persona_id, data in PERSONAS.items():
            if persona.name == data["name"]:
                if persona.config_json != {"dataset_path": data["dataset_path"]}:
                    persona.config_json = {"dataset_path": data["dataset_path"]}
                    updated.append(persona.name)
                break
    
    if updated:
        db.commit()
        return {"message": f"Updated personas: {', '.join(updated)}"}
    
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

@app.get("/generate/{persona_id}")
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
        raise HTTPException(status_code=500, detail="No dataset path configured")
    
    try:
        # Get or create model key based on dataset path and persona
        model_key = f"persona_{persona_id}_{hash(dataset_path)}"
        model = models.get(model_key)
        
        if not model:
            X, C = data_processor.load_data(dataset_path)
            input_dim, condition_dim = X.shape[1], C.shape[1]
            tensor_X, tensor_C = torch.FloatTensor(X), torch.FloatTensor(C)
            
            # Create model with optimized parameters
            model = WGAN_GP(input_dim=input_dim, output_dim=input_dim, condition_dim=condition_dim)
            
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
        target_distribution = data_processor.ensure_category_distribution(persona.name)
        if not target_distribution:
            chosen_categories = random.choices(list(categories), k=n_samples)
        else:
            category_weights = {
                cat: (rng['min'] + rng['max']) / 2 for cat, rng in target_distribution.items()
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
        category_indices = {cat: list(categories).index(f"category_{cat}") for cat in set(chosen_categories)}
        
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
        db_transactions = [
            Transaction(
                batch_id=batch.id,
                transaction_id=tx["transactionId"],
                booking_date_time=datetime.fromisoformat(tx["bookingDateTime"]),
                value_date_time=datetime.fromisoformat(tx["valueDateTime"]),
                amount=float(tx["transactionAmount"]["amount"]),
                currency=tx["transactionAmount"]["currency"],
                creditor_name=tx["creditorName"],
                creditor_account_iban=tx["creditorAccount"]["iban"],
                debtor_name=tx["debtorName"],
                debtor_account_iban=tx["debtorAccount"]["iban"],
                remittance_information_unstructured=tx["remittanceInformationUnstructured"],
                category=tx["category"]
            )
            for tx in transactions
        ]
        
        # Bulk insert all transactions
        db.bulk_save_objects(db_transactions)
        db.commit()
        
        return {"count": len(transactions), "transactions": transactions}
        
    except Exception as e:
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
    
    # Delete associated transactions first
    db.query(Transaction).filter(Transaction.batch_id == batch_id).delete()
    
    # Delete the batch
    db.delete(batch)
    db.commit()
    
    return {"message": "Batch deleted successfully"}

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
    
    # Delete the transaction
    db.delete(transaction)
    db.commit()
    
    return {"message": "Transaction deleted successfully"}

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
    
    # Update the batch name
    batch.name = name
    db.commit()
    
    return {"message": "Batch updated successfully", "name": name}
