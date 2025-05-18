"""add_audit_logs_table

Revision ID: d2e46cb982ed
Revises: 7b85a0ea0447
Create Date: 2024-03-21

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'd2e46cb982ed'
down_revision = '7b85a0ea0447'
branch_labels = None
depends_on = None

def upgrade():
    # Create audit_logs table
    op.create_table(
        'audit_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('action_type', sa.String(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('entity_type', sa.String(), nullable=False),
        sa.Column('entity_id', sa.String(), nullable=False),
        sa.Column('details', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.CheckConstraint(
            "action_type IN ('batch_generated', 'batch_deleted', 'batch_name_edited', "
            "'transaction_edited', 'transaction_deleted', 'distribution_updated', "
            "'persona_created', 'persona_updated', 'batch_downloaded_csv', "
            "'batch_downloaded_json', 'batch_downloaded_excel')",
            name='valid_action_type'
        )
    )
    op.create_index(op.f('ix_audit_logs_user_id'), 'audit_logs', ['user_id'], unique=False)
    op.create_index(op.f('ix_audit_logs_timestamp'), 'audit_logs', ['timestamp'], unique=False)
    op.create_index(op.f('ix_audit_logs_entity_type'), 'audit_logs', ['entity_type'], unique=False)
    op.create_index(op.f('ix_audit_logs_action_type'), 'audit_logs', ['action_type'], unique=False)

def downgrade():
    op.drop_index(op.f('ix_audit_logs_action_type'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_entity_type'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_timestamp'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_user_id'), table_name='audit_logs')
    op.drop_table('audit_logs')
