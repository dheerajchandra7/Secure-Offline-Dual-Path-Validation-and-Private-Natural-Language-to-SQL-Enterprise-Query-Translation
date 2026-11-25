"""Create initial platform schema."""
from alembic import op
import sqlalchemy as sa

revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None
user_role = sa.Enum("USER", "ADMIN", name="user_role")
user_status = sa.Enum("PENDING", "ACTIVE", "REJECTED", "DISABLED", name="user_status")
model_family = sa.Enum("TRADITIONAL_ML", "TFIDF_DL", "TRANSFORMER_CLASSIFIER", "EMBEDDING_CLASSIFIER", name="model_family")
job_status = sa.Enum("QUEUED", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED", name="job_status")

def upgrade() -> None:
    bind = op.get_bind()
    for enum_type in (user_role, user_status, model_family, job_status):
        enum_type.create(bind, checkfirst=True)
    op.create_table("users",
        sa.Column("id", sa.Integer(), primary_key=True), sa.Column("username", sa.String(40), nullable=False, unique=True),
        sa.Column("email", sa.String(254), nullable=False, unique=True), sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("full_name", sa.String(120), nullable=False), sa.Column("role", user_role, nullable=False),
        sa.Column("status", user_status, nullable=False), sa.Column("approved_by", sa.Integer(), nullable=True),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False))
    op.create_index("ix_users_username", "users", ["username"], unique=True)
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_index("ix_users_status", "users", ["status"])
    op.create_table("model_definitions",
        sa.Column("id", sa.Integer(), primary_key=True), sa.Column("key", sa.String(100), nullable=False, unique=True),
        sa.Column("display_name", sa.String(160), nullable=False), sa.Column("family", model_family, nullable=False),
        sa.Column("description", sa.Text(), nullable=False), sa.Column("artifact_path", sa.String(500), nullable=True),
        sa.Column("adapter_key", sa.String(100), nullable=False), sa.Column("target_column", sa.String(100), nullable=False),
        sa.Column("input_schema", sa.JSON(), nullable=False), sa.Column("is_enabled", sa.Boolean(), nullable=False),
        sa.Column("is_proposed", sa.Boolean(), nullable=False), sa.Column("version", sa.String(40), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False))
    op.create_index("ix_model_definitions_key", "model_definitions", ["key"], unique=True)
    op.create_index("ix_model_definitions_is_enabled", "model_definitions", ["is_enabled"])
    op.create_table("inference_jobs",
        sa.Column("id", sa.Integer(), primary_key=True), sa.Column("public_id", sa.String(36), nullable=False, unique=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("model_id", sa.Integer(), sa.ForeignKey("model_definitions.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("status", job_status, nullable=False), sa.Column("input_path", sa.String(500), nullable=True),
        sa.Column("result_path", sa.String(500), nullable=True), sa.Column("summary", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True), sa.Column("row_count", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True), sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_inference_jobs_public_id", "inference_jobs", ["public_id"], unique=True)
    op.create_index("ix_inference_jobs_user_id", "inference_jobs", ["user_id"])
    op.create_index("ix_inference_jobs_status", "inference_jobs", ["status"])
    op.create_table("audit_logs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("actor_user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("action", sa.String(100), nullable=False), sa.Column("resource_type", sa.String(80), nullable=False),
        sa.Column("resource_id", sa.String(100), nullable=True), sa.Column("details", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False))
    op.create_index("ix_audit_logs_actor_user_id", "audit_logs", ["actor_user_id"])
    op.create_index("ix_audit_logs_action", "audit_logs", ["action"])
    op.create_index("ix_audit_logs_created_at", "audit_logs", ["created_at"])

def downgrade() -> None:
    for table in ("audit_logs", "inference_jobs", "model_definitions", "users"):
        op.drop_table(table)
    bind = op.get_bind()
    for enum_type in (job_status, model_family, user_status, user_role):
        enum_type.drop(bind, checkfirst=True)
