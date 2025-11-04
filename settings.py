from pydantic import BaseModel, Field


class SimConfig(BaseModel):
    nr_of_ticks: int = Field(default=0, gt=0, description="Number of ticks")
    nr_of_people: int = Field(default=0, gt=0, description="Number of persons")
    nr_of_corps: int = Field(default=0, gt=0, description="Number of corps")
    min_base_salary: int = Field(
        default=0, gt=0, description="Min recurring salary every step"
    )
    max_base_salary: int = Field(
        default=0, gt=0, description="Max recurring salary every step"
    )
