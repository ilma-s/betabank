"""initial_schema

Revision ID: 922c0072074b
Revises: None
Create Date: 2024-03-21

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '922c0072074b'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Create users table
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('username', sa.String(), nullable=False),
        sa.Column('password_hash', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('username')
    )

    # Create personas table
    op.create_table(
        'personas',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('config_json', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE')
    )

    # Create transaction_batches table
    op.create_table(
        'transaction_batches',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('persona_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('preview_json', sa.JSON(), nullable=True),
        sa.Column('months', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['persona_id'], ['personas.id'], ondelete='CASCADE')
    )

    # Create transactions table with unique transaction_id
    op.create_table(
        'transactions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('transaction_id', sa.String(), nullable=False),
        sa.Column('batch_id', sa.Integer(), nullable=False),
        sa.Column('booking_date_time', sa.DateTime(), nullable=False),
        sa.Column('value_date_time', sa.DateTime(), nullable=False),
        sa.Column('amount', sa.Float(), nullable=False),
        sa.Column('currency', sa.String(), nullable=False),
        sa.Column('creditor_name', sa.String(), nullable=False),
        sa.Column('creditor_account_iban', sa.String(), nullable=False),
        sa.Column('debtor_name', sa.String(), nullable=False),
        sa.Column('debtor_account_iban', sa.String(), nullable=False),
        sa.Column('remittance_information_unstructured', sa.Text(), nullable=True),
        sa.Column('category', sa.String(), nullable=False),
        sa.Column('edited', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('transaction_id'),
        sa.ForeignKeyConstraint(['batch_id'], ['transaction_batches.id'], ondelete='CASCADE')
    )

    # Create audit_logs table
    op.create_table(
        'audit_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('action_type', sa.String(), nullable=False),
        sa.Column('entity_type', sa.String(), nullable=False),
        sa.Column('entity_id', sa.String(), nullable=False),
        sa.Column('details', sa.JSON(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE')
    )

    # Create pattern_library table
    op.create_table(
        'pattern_library',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('pattern_type', sa.String(), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('rules', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.CheckConstraint(
            "pattern_type IN ('temporal', 'amount', 'category', 'distribution')",
            name='valid_pattern_type'
        )
    )
    op.create_index('ix_pattern_library_pattern_type', 'pattern_library', ['pattern_type'], unique=False)

    # Create transaction_explanations table
    op.create_table(
        'transaction_explanations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('transaction_id', sa.String(), nullable=False),
        sa.Column('batch_id', sa.Integer(), nullable=False),
        sa.Column('feature_importance', sa.JSON(), nullable=False),
        sa.Column('applied_patterns', sa.JSON(), nullable=False),
        sa.Column('explanation_text', sa.Text(), nullable=False),
        sa.Column('confidence_score', sa.Float(), nullable=False),
        sa.Column('meta_info', sa.JSON(), nullable=True),  # Changed from metadata to meta_info
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['batch_id'], ['transaction_batches.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['transaction_id'], ['transactions.transaction_id'], ondelete='CASCADE')
    )
    op.create_index('ix_transaction_explanations_transaction_id', 'transaction_explanations', ['transaction_id'], unique=True)
    op.create_index('ix_transaction_explanations_batch_id', 'transaction_explanations', ['batch_id'], unique=False)

    # Create batch_explanations table
    op.create_table(
        'batch_explanations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('batch_id', sa.Integer(), nullable=False),
        sa.Column('distribution_explanation', sa.JSON(), nullable=False),
        sa.Column('temporal_patterns', sa.JSON(), nullable=False),
        sa.Column('amount_patterns', sa.JSON(), nullable=False),
        sa.Column('anomalies', sa.JSON(), nullable=True),
        sa.Column('summary_text', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['batch_id'], ['transaction_batches.id'], ondelete='CASCADE')
    )
    op.create_index('ix_batch_explanations_batch_id', 'batch_explanations', ['batch_id'], unique=True)

def downgrade():
    # Drop tables in reverse order
    op.drop_table('batch_explanations')
    op.drop_table('transaction_explanations')
    op.drop_table('pattern_library')
    op.drop_table('audit_logs')
    op.drop_table('transactions')
    op.drop_table('transaction_batches')
    op.drop_table('personas')
    op.drop_table('users')
