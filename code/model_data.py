from pydantic import BaseModel
from typing import List

class PreferenceInfoRes(BaseModel):
    bmId: int
    spViewTime: int
    isViewed: bool
    isInvested: bool

class MemberPreferenceInfoRes(BaseModel):
    memberId: int
    preferenceInfoResList: List[PreferenceInfoRes]