from pydantic import BaseModel, Field


class PersonSeed(BaseModel):
    balance: float = Field(default=0, ge=0, description="Account balance")
    mpc: float = Field(
        default=0.4, gt=0, le=1.0, description="Marginal propensity to consume"
    )


class CorpSeed(BaseModel):
    balance: float = Field(default=0, ge=0, description="Account balance")
    salary: float = Field(default=0, ge=0, description="Salary level")
    nr_of_employees: int = Field(default=0, ge=0, description="Number of employees")
    upe: int = Field(default=0, ge=0, description="Unit per employee")
    price: float = Field(default=0, ge=0, description="Price per product")
