"""Create articles table

Revision ID: create_articles_table
Revises: 
Create Date: 2025-03-02 01:00:00

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'create_articles_table'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    op.create_table(
        'articles',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('content', sa.Text, nullable=False),
        sa.Column('summary', sa.Text, nullable=True),
        sa.Column('images', sa.Text, nullable=True),
        sa.Column('url', sa.String(length=255), nullable=True),
        sa.Column('source_id', sa.Integer, nullable=True),
        sa.Column('published_at', sa.DateTime, nullable=True),
        sa.Column('discovered_at', sa.DateTime, nullable=True),
        sa.Column('author', sa.String(length=100), nullable=True),
        sa.Column('topic', sa.String(length=100), nullable=True),
        sa.Column('category', sa.String(length=50), nullable=True),
        sa.Column('language', sa.String(length=10), nullable=True),
        sa.Column('sentiment_score', sa.Float, nullable=True),
        sa.Column('readability_score', sa.Float, nullable=True),
        sa.Column('trending_topic_id', sa.Integer, nullable=True)
    )

def downgrade() -> None:
    op.drop_table('articles')
