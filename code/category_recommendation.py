import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import warnings
import torch
warnings.filterwarnings(action='ignore')
from categories import categories  # Your custom categories mapping
import os

# ---------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
os.path.join(base_dir, "data", "ratings_small.csv")
user_invest_data = pd.read_csv(os.path.join(base_dir, "data/investment_preference.csv"), encoding='utf-8')
company_data = pd.read_csv(os.path.join(base_dir, "data/company_key_information.csv"), encoding='utf-8')

# ---------------------------------------------------------------------
# 2. Data Preprocessing
# ---------------------------------------------------------------------

scaler = MinMaxScaler()

company_data['investmentGoal(Unscaled)'] = company_data['investmentGoal']
company_data['risk'] = company_data['risk'] / 100
company_data[['risk', 'investmentGoal']] = scaler.fit_transform(
    company_data[['risk', 'investmentGoal']].fillna(0)
)
company_data.fillna({'category2':0},inplace=True)

risk_mapping = {'high': 0.7, 'medium': 0.5, 'low': 0.25}
user_invest_data['risk_tolerance_numeric'] = user_invest_data['risk_tolerance'].map(risk_mapping)

# String Data to Int
def parse_categories(row):
    return [int(x) for x in row if not pd.isnull(x)]

company_data['categories'] = company_data[['category1', 'category2']].apply(parse_categories, axis=1)
user_invest_data['categories'] = user_invest_data[['user_category1', 'user_category2', 'user_category3']].apply(
    parse_categories, axis=1)

#user data must be sorted by userid
user_invest_data.sort_values(by='userId')

risk_weights = {
    'high': {'alpha': 0.5, 'beta': 0.4, 'gamma': 0.3, 'delta': -0.2},
    'medium': {'alpha': 0.4, 'beta': 0.3, 'gamma': 0.2, 'delta': -0.1},
    'low': {'alpha': 0.3, 'beta': 0.2, 'gamma': 0.1, 'delta': -0.05}
}

# ---------------------------------------------------------------------
# 3. Caculate score
# ---------------------------------------------------------------------

def recommend_top_companies(user_id, num_recommendations=3):
    user_idx = user_id - 1
    user_data = user_invest_data.iloc[user_idx]
    
    #Extract user info
    user_categories = user_data['categories']
    user_risk_level = user_data['risk_tolerance']

    weights = risk_weights[user_risk_level]
    alpha = weights['alpha']
    beta = weights['beta']
    gamma = weights['gamma']
    delta = weights['delta']
    
    company_selected = []
    
    for idx,row in company_data.iterrows():
        company_categories = row[['category1','category2']]
        is_overlabed = set(company_categories) & set(user_categories)
        if len(is_overlabed) > 0:
            company_selected.append(row.to_frame().T)
    
    if company_selected:
        company_selected = pd.concat(company_selected,ignore_index=True)
    else:
        print("No Selected Company.")
        return []   
    
    company_selected['final_score'] = (
            0.1*(alpha * company_selected['risk'] +
            beta * company_selected['investmentGoal'] +
            gamma * scaler.fit_transform(company_selected[['Fully Diluted Shares']].fillna(0))[:, 0] +
            delta * scaler.fit_transform(company_selected[['price per share']].fillna(0))[:, 0])
    )

    company_selected['final_score'] = pd.to_numeric(company_selected['final_score'], errors='coerce')
    
    top_company_indices = company_selected['final_score'].nlargest(num_recommendations).index
    recommendations = company_selected.iloc[top_company_indices][
        ['category1','category2', 'name', 'investmentGoal(Unscaled)', 'price per share','bmId','final_score']
    ]
    category_dict = {i + 1: category for i, category in enumerate(categories)}
    recommendations['category1'] = recommendations['category1'].map(category_dict)
    recommendations['category2'] = recommendations['category2'].map(category_dict)
    
    recommendations['price per share'] = recommendations['price per share'].apply(
        lambda x: f"₩{int(x):,}"
    )

    return recommendations
