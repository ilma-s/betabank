"""update transaction relationships

Revision ID: update_transaction_relationships
Revises: 922c0072074b
Create Date: 2024-03-22 11:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'update_transaction_relationships'
down_revision = '922c0072074b'
branch_labels = None
depends_on = None

def upgrade():
    # Make transaction_id unique in transactions table
    op.create_unique_constraint('uq_transactions_transaction_id', 'transactions', ['transaction_id'])
    
    # Add foreign key from transaction_explanations to transactions
    op.create_foreign_key(
        'fk_transaction_explanations_transaction_id_transactions',
        'transaction_explanations',
        'transactions',
        ['transaction_id'],
        ['transaction_id'],
        ondelete='CASCADE'
    )

def downgrade():
    # Drop the foreign key
    op.drop_constraint('fk_transaction_explanations_transaction_id_transactions', 'transaction_explanations', type_='foreignkey')
    
    # Drop the unique constraint
    op.drop_constraint('uq_transactions_transaction_id', 'transactions', type_='unique') 