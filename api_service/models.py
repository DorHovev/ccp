from pydantic import BaseModel, field_validator, ValidationInfo, model_validator

class PredictionFeatures(BaseModel):
    TotalCharges: float
    Month_to_month: int
    One_year: int
    Two_year: int
    PhoneService: int # Expecting 0 or 1
    tenure: int

    @field_validator('PhoneService')
    @classmethod
    def phone_service_must_be_binary(cls, v: int) -> int:
        if v not in [0, 1]:
            raise ValueError('PhoneService must be 0 (No) or 1 (Yes)')
        return v

    @field_validator('Month_to_month', 'One_year', 'Two_year')
    @classmethod
    def contract_type_must_be_binary(cls, v: int, info: ValidationInfo) -> int:
        if v not in [0, 1]:
            raise ValueError(f'{info.field_name} must be 0 or 1 (one-hot encoded)')
        return v
    
    @field_validator('tenure')
    @classmethod
    def tenure_must_be_positive(cls,v: int) -> int:
        if v < 0:
            raise ValueError('tenure must be non-negative')
        return v

    @classmethod
    def validate_contract_one_hot(cls, values):
        contract_fields = [values.Month_to_month, values.One_year, values.Two_year]
        if contract_fields.count(1) != 1:
            raise ValueError('Exactly one of Month_to_month, One_year, or Two_year must be 1 (one-hot encoded)')
        return values

    _validate_contract_one_hot = model_validator(mode="after")(validate_contract_one_hot) 