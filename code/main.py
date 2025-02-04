import pandas as pd
import uvicorn
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel
from category_recommendation import recommend_top_companies
from Recommendation_through_viewtime import ViewTimePredictor,load_model
import os
import logging
from typing import List
from model_data import MemberPreferenceInfoRes, PreferenceInfoRes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base_dir = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(os.path.join(base_dir, "data/ratings_small.csv"), encoding='utf-8')

app = FastAPI()
class RecommendationRequest(BaseModel):
    memberId: int
    num_recommendations: int
# ---------------------------------------------------------------------
# AI load
# ---------------------------------------------------------------------

X_columns = [col for col in data.columns if col.startswith('viewed time')]

hidden_dim = 128

model = None
df_mapping = None

@app.get("/")
def root():
    return {"message": "Welcome to the Recommendation API"}

@app.post("/models")
def load_and_save_model(infos: List[MemberPreferenceInfoRes]):
    # 데이터 잘 받아와지는지 확인하는 코드로, 실제로 삭제해도 무방합니다.
    # for info in infos:
    #     print(info)
    #     for preference_info in info.preferenceInfoResList:
    #         print(preference_info)
            
    global model, df_mapping
    # model = load_model(infos)
    model, X_train_tensor, y_train_tensor, df_mapping = load_model(infos)

    with torch.no_grad():
        predictions = model(X_train_tensor)  # Get predictions for training data
        df_mapping.loc[:len(X_train_tensor)-1, "prediction"] = predictions.numpy().flatten()  # Assign only to training rows

        # Print predicted values per user & item
        print(df_mapping[["memberId", "bmId", "prediction"]])


async def get_recommendation(memberId: int, num_recommendations: int):
    try:
        logger.info(f"Processing request for memberId: {memberId}")

        # Get category-based recommendations
        recommendations_by_category = recommend_top_companies(memberId, num_recommendations)

        # Handle empty recommendations
        if isinstance(recommendations_by_category, list):
            recommendations_by_category = pd.DataFrame(recommendations_by_category)

        if recommendations_by_category.empty:
            logger.warning("No category-based recommendations found")
            recommendations_by_category = pd.DataFrame(columns=['bmId', 'score'])

        # Get AI predictions
        X = torch.tensor(data[X_columns].iloc[memberId - 1].values, dtype=torch.float32)
        with torch.no_grad():
            ai_scores = model(X).detach().numpy()

        # Merge scores
        ai_df = pd.DataFrame({'bmId': range(len(ai_scores)), 'ai_score': ai_scores})
        merged = pd.merge(
            recommendations_by_category,
            ai_df,
            on='bmId',
            how='left'
        ).fillna(0)

        # Calculate final score
        merged['score'] = merged['final_score'] + merged['ai_score']

        # Final cleanup
        merged = merged.sort_values('score', ascending=False)
        merged = merged[~pd.isna(merged['bmId'])]
        merged['bmId'] = merged['bmId'].fillna(0).astype(int)

        return merged.head(num_recommendations)['bmId'].tolist()

    except Exception as e:
        logger.error(f"Error for memberId {memberId}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@app.post("/recommendations/")
async def recommend(request: RecommendationRequest):
    try:
        memberId = request.memberId
        num_recommendations = request.num_recommendations
        if memberId < 1 or memberId > len(data):
            raise HTTPException(status_code=400, detail="Invalid memberId")

        recommend_bm = await get_recommendation(memberId, num_recommendations)

        # ✅ Fix: Remove any remaining NaN values
        recommend_bm = [int(x) for x in recommend_bm if not pd.isna(x)]

        return {
            "memberId": memberId,
            "num_recommendations": num_recommendations,
            "bmId_recommendations": recommend_bm,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
