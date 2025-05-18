"""cleanup feedback tables

Revision ID: cleanup_feedback_tables
Revises: update_transaction_relationships
Create Date: 2024-03-22 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.exc import ProgrammingError

# revision identifiers, used by Alembic.
revision = 'cleanup_feedback_tables'
down_revision = 'update_transaction_relationships'
branch_labels = None
depends_on = None

def upgrade():
    # Drop any remaining feedback-related tables
    try:
        op.drop_table('explanation_feedback')
    except (sa.exc.OperationalError, ProgrammingError):
        pass  # Table doesn't exist, which is fine

def downgrade():
    pass  # No downgrade needed since we're removing feedback functionality 