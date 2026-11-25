import secrets
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session
from app.api.deps import get_current_user
from app.core.config import settings
from app.core.security import create_access_token, new_csrf_token
from app.db.session import get_db
from app.models.user import User
from app.schemas.auth import LoginRequest, MessageResponse, RegisterRequest, UserResponse
from app.services.auth import InactiveAccount, InvalidCredentials, RegistrationConflict, authenticate_user, register_user

router = APIRouter(prefix="/auth", tags=["authentication"])

def _set_session_cookies(response: Response, user: User) -> None:
    common = {"secure": settings.cookie_secure, "samesite": settings.cookie_samesite, "path": "/", "max_age": settings.access_token_minutes * 60}
    response.set_cookie("access_token", create_access_token(str(user.id), user.role.value), httponly=True, **common)
    response.set_cookie("csrf_token", new_csrf_token(), httponly=False, **common)

@router.post("/register", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    try:
        register_user(db, payload)
    except RegistrationConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return MessageResponse(message="Registration submitted. Administrator approval is required.")

@router.post("/login", response_model=UserResponse)
def login(payload: LoginRequest, response: Response, db: Session = Depends(get_db)):
    try:
        user = authenticate_user(db, payload.username_or_email, payload.password)
    except InvalidCredentials:
        raise HTTPException(status_code=401, detail="Invalid username/email or password")
    except InactiveAccount as exc:
        message = "Account is pending administrator approval" if str(exc) == "pending" else "Account is not active"
        raise HTTPException(status_code=403, detail=message)
    _set_session_cookies(response, user)
    return user

@router.get("/me", response_model=UserResponse)
def me(user: User = Depends(get_current_user)):
    return user

@router.post("/logout", response_model=MessageResponse)
def logout(response: Response, request: Request):
    csrf_cookie = request.cookies.get("csrf_token")
    csrf_header = request.headers.get("X-CSRF-Token")
    if csrf_cookie and (not csrf_header or not secrets.compare_digest(csrf_cookie, csrf_header)):
        raise HTTPException(status_code=403, detail="CSRF validation failed")
    response.delete_cookie("access_token", path="/", secure=settings.cookie_secure, samesite=settings.cookie_samesite)
    response.delete_cookie("csrf_token", path="/", secure=settings.cookie_secure, samesite=settings.cookie_samesite)
    return MessageResponse(message="Logged out")
