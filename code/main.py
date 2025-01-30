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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "/Users/kimjinha/Documents/GitHub/youngflow/pitchain 2/data/ratings_small.csv")  # Adjust the relative path

app = FastAPI()
class RecommendationRequest(BaseModel):
    memberId: int
    num_recommendations: int
# ---------------------------------------------------------------------
# AI load
# ---------------------------------------------------------------------
data = pd.read_csv(data_path,encoding='utf-8')

X_columns = [col for col in data.columns if col.startswith('viewed time')]
#X = data[X_columns].iloc[memberId]
#X = torch.tensor(X, dtype=torch.float32)
hidden_dim = 128
model = ViewTimePredictor(input_dim=len(X_columns), hidden_dim=hidden_dim, output_dim=len(X_columns))
model = load_model(model,load_path = "/Users/kimjinha/Documents/GitHub/youngflow/pitchain 2/code/model_weights.pth")
model.eval()



@app.get("/")
def root():
    return {"message": "Welcome to the Recommendation API"}


def get_recommendation(memberId: int, num_recommendations: int):
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "/Users/kimjinha/Documents/GitHub/youngflow/pitchain 2/data/ratings_small.csv")  # Adjust the relative path

app = FastAPI()
class RecommendationRequest(BaseModel):
    memberId: int
    num_recommendations: int
# ---------------------------------------------------------------------
# AI load
# ---------------------------------------------------------------------
data = pd.read_csv(data_path,encoding='utf-8')

X_columns = [col for col in data.columns if col.startswith('viewed time')]
#X = data[X_columns].iloc[memberId]
#X = torch.tensor(X, dtype=torch.float32)
hidden_dim = 128
model = ViewTimePredictor(input_dim=len(X_columns), hidden_dim=hidden_dim, output_dim=len(X_columns))
model = load_model(model,load_path = "/Users/kimjinha/Documents/GitHub/youngflow/pitchain 2/code/model_weights.pth")
model.eval()



@app.get("/")
def root():
    return {"message": "Welcome to the Recommendation API"}


def get_recommendation(memberId: int, num_recommendations: int):
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

# -----------------------------------------------------------------------------------------------------
# def get_recommendation(memberId: int, num_recommendations:int):
#     recommendations_by_category = recommend_top_companies(memberId, num_recommendations)
#     print(f"recommendations_by_category: {recommendations_by_category}")
#     if isinstance(recommendations_by_category, list):
#         recommendations_by_category = pd.DataFrame(recommendations_by_category)
#
#     X = torch.tensor(data[X_columns].iloc[memberId - 1].values, dtype=torch.float32)
#     with torch.no_grad():
#         recommendations_by_AI = model(X)
#     recommendations_by_category = recommendations_by_category.reset_index(drop=True)
#     final_score = recommendations_by_category['final_score']
#     AI_calculation = pd.Series(recommendations_by_AI.detach().numpy()).loc[recommendations_by_category['bmId']].reset_index(drop=True)
#
#     recommendations_by_category['score'] = final_score+AI_calculation
#     recommendations_by_category = recommendations_by_category.sort_values('score',ascending=False)
#     # print(recommendations_by_category['score'])
#     # print(recommendations_by_category)
#     return recommendations_by_category

@app.post("/recommendations/")
def recommend(request: RecommendationRequest):
    try:
        memberId = request.memberId
        num_recommendations = request.num_recommendations
        if memberId < 1 or memberId > len(data):
            raise HTTPException(status_code=400, detail="Invalid memberId")

        recommend_bm = get_recommendation(memberId, num_recommendations)

        # ✅ Fix: Remove any remaining NaN values
        recommend_bm = [int(x) for x in recommend_bm if not pd.isna(x)]

        return {
            "memberId": memberId,
            "num_recommendations": num_recommendations,
            "bmId_recommendations": recommend_bm,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

# recommendations_by_category = recommend_top_companies(user_id, num_recommendations)
# print(recommendations_by_category)
# recommendations_by_AI = model(X)
# recommendations_by_category = recommendations_by_category.reset_index(drop=True)
# final_score = recommendations_by_category['final_score']
# AI_calculation = pd.Series(recommendations_by_AI.detach().numpy()).loc[recommendations_by_category['bmId']].reset_index(drop=True)
# AI_calculation = pd.Series(recommendations_by_AI.detach().numpy()).loc[recommendations_by_category['bmId']].reset_index(drop=True)
#
# recommendations_by_category['score'] = final_score+AI_calculation
# recommendations_by_category = recommendations_by_category.sort_values('score',ascending=False)
# print(recommendations_by_category['score'])
# print(recommendations_by_category)




if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# -----------------------------------------------------------------------------------------------------
# def get_recommendation(memberId: int, num_recommendations:int):
#     recommendations_by_category = recommend_top_companies(memberId, num_recommendations)
#     print(f"recommendations_by_category: {recommendations_by_category}")
#     if isinstance(recommendations_by_category, list):
#         recommendations_by_category = pd.DataFrame(recommendations_by_category)
#
#     X = torch.tensor(data[X_columns].iloc[memberId - 1].values, dtype=torch.float32)
#     with torch.no_grad():
#         recommendations_by_AI = model(X)
#     recommendations_by_category = recommendations_by_category.reset_index(drop=True)
#     final_score = recommendations_by_category['final_score']
#     AI_calculation = pd.Series(recommendations_by_AI.detach().numpy()).loc[recommendations_by_category['bmId']].reset_index(drop=True)
#
#     recommendations_by_category['score'] = final_score+AI_calculation
#     recommendations_by_category = recommendations_by_category.sort_values('score',ascending=False)
#     # print(recommendations_by_category['score'])
#     # print(recommendations_by_category)
#     return recommendations_by_category

@app.post("/recommendations/")
def recommend(request: RecommendationRequest):
    try:
        memberId = request.memberId
        num_recommendations = request.num_recommendations
        if memberId < 1 or memberId > len(data):
            raise HTTPException(status_code=400, detail="Invalid memberId")

        recommend_bm = get_recommendation(memberId, num_recommendations)

        # ✅ Fix: Remove any remaining NaN values
        recommend_bm = [int(x) for x in recommend_bm if not pd.isna(x)]

        return {
            "memberId": memberId,
            "num_recommendations": num_recommendations,
            "bmId_recommendations": recommend_bm,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

# recommendations_by_category = recommend_top_companies(user_id, num_recommendations)
# print(recommendations_by_category)
# recommendations_by_AI = model(X)
# recommendations_by_category = recommendations_by_category.reset_index(drop=True)
# final_score = recommendations_by_category['final_score']
# AI_calculation = pd.Series(recommendations_by_AI.detach().numpy()).loc[recommendations_by_category['bmId']].reset_index(drop=True)
# AI_calculation = pd.Series(recommendations_by_AI.detach().numpy()).loc[recommendations_by_category['bmId']].reset_index(drop=True)
#
# recommendations_by_category['score'] = final_score+AI_calculation
# recommendations_by_category = recommendations_by_category.sort_values('score',ascending=False)
# print(recommendations_by_category['score'])
# print(recommendations_by_category)




if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
