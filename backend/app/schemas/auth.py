from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field, StringConstraints
from typing import Annotated


Username = Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=40, pattern=r"^[A-Za-z0-9_.-]+$")]
FullName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=120)]


class RegisterRequest(BaseModel):
    username: Username
    email: EmailStr
    full_name: FullName
    password: str = Field(min_length=12, max_length=128)


class LoginRequest(BaseModel):
    username_or_email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=128)


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    email: EmailStr
    full_name: str
    role: str
    status: str
    created_at: datetime


class MessageResponse(BaseModel):
    message: str
