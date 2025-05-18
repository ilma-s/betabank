"""add explanation feedback table

Revision ID: add_explanation_feedback
Create Date: 2024-03-21 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON

# revision identifiers, used by Alembic.
revision = 'add_explanation_feedback'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        'explanation_feedback',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('explanation_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('rating', sa.Integer(), nullable=False),
        sa.Column('feedback_text', sa.Text(), nullable=True),
        sa.Column('pattern_annotations', JSON, nullable=True),
        sa.Column('is_helpful', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.text('now()')),
        sa.ForeignKeyConstraint(['explanation_id'], ['batch_explanations.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Add indexes for common queries
    op.create_index(
        'ix_explanation_feedback_explanation_id',
        'explanation_feedback',
        ['explanation_id']
    )
    op.create_index(
        'ix_explanation_feedback_user_id',
        'explanation_feedback',
        ['user_id']
    )
    op.create_index(
        'ix_explanation_feedback_created_at',
        'explanation_feedback',
        ['created_at']
    )

def downgrade():
    op.drop_index('ix_explanation_feedback_created_at')
    op.drop_index('ix_explanation_feedback_user_id')
    op.drop_index('ix_explanation_feedback_explanation_id')
    op.drop_table('explanation_feedback') 