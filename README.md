# BetaBank

**BetaBank** is a synthetic banking platform for generating, managing, and analyzing synthetic user personas and their financial transactions. It is designed to support compliant fintech product testing and validation, as described in the thesis _"Creating Synthetic User Personas to Support Compliant Fintech Product Testing and Validation"_.

---

## 🚀 Quick Start

### 1. **Backend Setup (FastAPI + PostgreSQL)**

#### **Requirements**
- Python 3.9+
- PostgreSQL (running locally)
- (Recommended) Virtual environment

#### **Install Python dependencies**

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### **Configure Database**

- Ensure PostgreSQL is running and a database named `betabank` exists.
- Update `backend/alembic.ini` and `backend/utils/database.py` if your DB credentials differ.

#### **Run Alembic Migrations**

```bash
alembic upgrade head
```

#### **Add Training Datasets**

- Place all persona training datasets (e.g., `crypto_enthusiast.json`, etc.) in:
  ```
  backend/training-datasets/testing-datasets/
  ```

#### **Start the Backend**

```bash
uvicorn api:app --reload
```
- The API will be available at: [http://localhost:8000](http://localhost:8000)

---

### 2. **Frontend Setup (Next.js)**

#### **Requirements**
- Node.js 18+
- pnpm (or npm/yarn)

#### **Install Node dependencies**

```bash
cd frontend
pnpm install
# or
npm install
```

#### **Start the Frontend**

```bash
pnpm dev
# or
npm run dev
```
- The app will be available at: [http://localhost:3000](http://localhost:3000)

---

## 🏦 Core Functionalities

### 1. **User Authentication & Management**
- Secure registration and login (passwords hashed with bcrypt, JWT-based sessions).
- Each user has a private workspace and data scope.

### 2. **Persona-Based Synthetic Transaction Generation**
- Choose from predefined personas (e.g., Crypto Enthusiast, Shopping Addict, Gambling Addict, Money Mule).
- Create custom personas with user-defined category distributions.
- Transactions are generated using a WGAN-GP AI model, based on persona configuration and real-world-like patterns.

### 3. **Transaction Batching and Management**
- Transactions are grouped into batches for traceability.
- Full CRUD (create, read, update, delete) operations on batches and individual transactions.
- Assign custom batch names.

### 4. **Data Export**
- Export transaction batches in JSON, CSV, or Excel format for further analysis or integration.

### 5. **AI-Powered Explanation Service**
- Analyze and explain generated data using a combination of rules and machine learning (GMM, KDE).
- Provides:
  - Temporal, amount-based, and categorical pattern analysis.
  - Batch-level summaries and transaction-level insights.

### 6. **Audit Logging**
- All user actions (edits, deletions, exports, etc.) are logged for traceability.
- Queryable audit log system with filters.

---

## 💻 Technical Stack

- **Backend:** FastAPI, SQLAlchemy, Alembic, PostgreSQL, PyTorch, scikit-learn
- **Frontend:** Next.js (React), TailwindCSS, shadcn/ui, Axios, Chart.js
- **AI/ML:** WGAN-GP for synthetic data generation, GMM/KDE for pattern analysis
- **Other:** JWT authentication, bcrypt password hashing

---

## 📝 Notes

- **No real user data is used or stored.** All data is synthetic and scoped per user.
- **Training datasets** for personas must be present in `backend/training-datasets/testing-datasets/`.
- **No AWS/S3 is required** for local operation; all data is loaded from the local filesystem.

---

## 🛠️ Extending BetaBank

- Add new personas by placing new training datasets in the training directory and updating persona configs.
- The codebase is modular and can be extended for new analytics, export formats, or UI features.

---

## 📚 Further Reading

- See the thesis _"Creating Synthetic User Personas to Support Compliant Fintech Product Testing and Validation"_ for the research context and design rationale.

