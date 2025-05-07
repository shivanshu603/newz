"""Merge heads

Revision ID: merge_heads
Revises: create_articles_table, f4b6d35ba7bc
Create Date: 2025-03-02 01:30:00

"""
from typing import Sequence, Union
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'merge_heads'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # Merge all heads into a single revision
    pass


def downgrade() -> None:
    # Downgrade logic if needed
    pass
