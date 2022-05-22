from typing import List
from pydantic import BaseModel

class OneThesisPredictionRequest(BaseModel):
        thesis_id: int
        text: str
	

class ThesesPredictionRequest(BaseModel):
	books: List[OneThesisPredictionRequest] = []