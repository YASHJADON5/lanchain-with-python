from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class Student(BaseModel):
    name: str ="Yash"
    age: Optional[int] = None
    email: EmailStr
    cgpa: float=Field(gt=0, lt=10, default = 5, description="cnvert this decimal value to int")


# pydantic also handles implicit type conversion wheneverit is possible
# pydantic is like zod for python

new_student = {"age":24, "email":"asdf@gmail.com","cgpa":5.00}

student = Student(**new_student)


print(student)