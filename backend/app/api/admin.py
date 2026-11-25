from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.api.deps import require_admin
from app.db.session import get_db
from app.models.audit_log import AuditLog
from app.models.user import User, UserRole, UserStatus
from app.schemas.auth import MessageResponse, UserResponse

router = APIRouter(prefix="/admin", tags=["administration"])

@router.get("/users/pending", response_model=list[UserResponse])
def pending_users(_: User = Depends(require_admin), db: Session = Depends(get_db)):
    return list(db.scalars(select(User).where(User.status == UserStatus.PENDING).order_by(User.created_at)))

@router.post("/users/{user_id}/approve", response_model=MessageResponse)
def approve_user(user_id: int, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    user = db.get(User, user_id)
    if not user: raise HTTPException(status_code=404, detail="User not found")
    if user.role == UserRole.ADMIN: raise HTTPException(status_code=400, detail="Administrator accounts cannot be approved here")
    user.status = UserStatus.ACTIVE
    user.approved_by = admin.id
    user.approved_at = datetime.now(timezone.utc)
    db.add(AuditLog(actor_user_id=admin.id, action="user.approved", resource_type="user", resource_id=str(user.id)))
    db.commit()
    return MessageResponse(message="User approved")

@router.post("/users/{user_id}/reject", response_model=MessageResponse)
def reject_user(user_id: int, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    user = db.get(User, user_id)
    if not user: raise HTTPException(status_code=404, detail="User not found")
    if user.role == UserRole.ADMIN: raise HTTPException(status_code=400, detail="Administrator accounts cannot be rejected here")
    user.status = UserStatus.REJECTED
    db.add(AuditLog(actor_user_id=admin.id, action="user.rejected", resource_type="user", resource_id=str(user.id)))
    db.commit()
    return MessageResponse(message="User rejected")
