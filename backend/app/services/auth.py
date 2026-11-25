from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.core.security import hash_password, verify_password
from app.models.user import User, UserRole, UserStatus
from app.schemas.auth import RegisterRequest


class RegistrationConflict(Exception):
    pass


class InvalidCredentials(Exception):
    pass


class InactiveAccount(Exception):
    pass


def register_user(db: Session, payload: RegisterRequest) -> User:
    existing = db.scalar(
        select(User).where(
            or_(User.username == payload.username, User.email == str(payload.email).lower())
        )
    )
    if existing:
        raise RegistrationConflict("Username or email is already registered")

    user = User(
        username=payload.username,
        email=str(payload.email).lower(),
        full_name=payload.full_name,
        password_hash=hash_password(payload.password),
        role=UserRole.USER,
        status=UserStatus.PENDING,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, identity: str, password: str) -> User:
    normalized = identity.strip()
    user = db.scalar(
        select(User).where(
            or_(User.username == normalized, User.email == normalized.lower())
        )
    )
    if not user or not verify_password(password, user.password_hash):
        raise InvalidCredentials()
    if user.status != UserStatus.ACTIVE:
        raise InactiveAccount(user.status.value)
    return user


def bootstrap_admin(db: Session, email: str, username: str, password: str) -> User:
    existing = db.scalar(select(User).where(User.role == UserRole.ADMIN))
    if existing:
        return existing
    admin = User(
        username=username,
        email=email.lower(),
        full_name="Platform Administrator",
        password_hash=hash_password(password),
        role=UserRole.ADMIN,
        status=UserStatus.ACTIVE,
    )
    db.add(admin)
    db.commit()
    db.refresh(admin)
    return admin
