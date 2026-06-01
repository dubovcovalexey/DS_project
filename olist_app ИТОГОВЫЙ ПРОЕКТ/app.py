import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from catboost import CatBoostRanker
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Olist RecSys", layout="wide")

st.markdown("""
<style>
    .reportview-container .main .block-container { padding-top: 1rem; }
    .dataframe { font-size: 13px !important; }
</style>
""", unsafe_allow_html=True)


# Neural network architectural framework for Two-Tower inference
class TwoTowerNetwork(nn.Module):
    def __init__(self, item_feat_dim, embedding_dim=64):
        super(TwoTowerNetwork, self).__init__()
        self.user_fc = nn.Sequential(nn.Linear(item_feat_dim, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, embedding_dim))
        self.item_fc = nn.Sequential(nn.Linear(item_feat_dim, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, embedding_dim))

    def forward_user(self, history, item_features_matrix):
        hist_features = item_features_matrix[history]
        mask = (history > 0).float().unsqueeze(-1)
        sum_features = torch.sum(hist_features * mask, dim=1)
        denom = torch.sum(mask, dim=1).clamp(min=1.0)
        mean_features = sum_features / denom
        return F.normalize(self.user_fc(mean_features), p=2, dim=1)

    def forward_item(self, item_features):
        return F.normalize(self.item_fc(item_features), p=2, dim=1)



# Full assembly and in-memory caching of models on the cloud server
@st.cache_resource
def load_all_models_and_assets():
    with open("streamlit_data.pkl", "rb") as f:
        data = pickle.load(f)
        
    x_dense = data["x_content"].toarray()
    x_item_tensor = torch.tensor(x_dense, dtype=torch.float32)
    
    padding_vector = torch.zeros((1, x_item_tensor.shape[1]), dtype=torch.float32)
    x_item_tensor_extended = torch.cat([padding_vector, x_item_tensor], dim=0)
    
    net = TwoTowerNetwork(x_item_tensor_extended.shape[1], embedding_dim=64)
    try:
        net.load_state_dict(torch.load("two_tower.pt", map_location=torch.device('cpu')))
    except:
        pass
    net.eval()
    
    with torch.no_grad():
        all_item_embeddings = net.forward_item(x_item_tensor_extended)
        
    cb = CatBoostRanker()
    try:
        cb.load_model("catboost_ranker.cbm")
    except:
        cb = None
        
    return data, net, x_item_tensor_extended, all_item_embeddings, cb

assets, model_pytorch, x_item_tensor_extended, all_item_embeddings, cb_model = load_all_models_and_assets()


full_history = assets["full_history"]
df_products = assets["df_products"]
swing_matrix = assets["swing_matrix"]
x_content = assets["x_content"]
product_list = assets["product_list"]
pid_to_idx = assets["pid_to_idx"]
pid_to_idx_shifted = assets["pid_to_idx_shifted"]
idx_to_pid_shifted = assets["idx_to_pid_shifted"]


# Sidebar for managing filtration modes and three-level item selection
st.sidebar.header("Settings")
mode = st.sidebar.radio("Search Mode:", ["User (User ID)", "Product (Product ID)"])

if mode == "User (User ID)":
    selected_user = st.sidebar.number_input("Customer Number (1 - 91979):", min_value=1, max_value=len(full_history), value=None, step=1)
    available_models = [
        "Model 1: Alibaba Swing (Graph-based)",
        "Model 2: Content-Based (Cosine Similarity)",
        "Model 3: Score Fusion (Additive Hybrid)",
        "Model 4: Two-Tower (DSSM Neural Network)",
        "Model 5: CatBoostRanker (Two-Stage Boosting)"
    ]
else:
    st.sidebar.subheader("Catalog Product Search")
    unique_macro_groups = sorted(df_products['category_group'].unique().tolist())
    selected_macro = st.sidebar.selectbox("Step 1: Macro-group:", unique_macro_groups)
    
    filtered_df_by_macro = df_products[df_products['category_group'] == selected_macro]
    # Filter out empty values, cast to strings, and safely call sorted()
    raw_unique = filtered_df_by_macro['product_category_name_english'].dropna().astype(str).unique().tolist()
    unique_micro_groups = sorted(raw_unique) if raw_unique else ['other']

    selected_micro = st.sidebar.selectbox("Step 2: Micro-category:", unique_micro_groups)
    
    filtered_products_by_micro = filtered_df_by_macro[filtered_df_by_macro['product_category_name_english'] == selected_micro].index.tolist()
    selected_product = st.sidebar.selectbox("Step 3: Target Product:", sorted(filtered_products_by_micro))
    
    available_models = ["Model 2: Content-Based (Cosine Similarity)"]

selected_model = st.sidebar.selectbox("Model:", available_models)
k_recs = st.sidebar.slider("Number of Recommendations:", 5, 15, 10)



def show_product_features_with_scores(title, recs_with_scores):
    st.subheader(title)
    
    # RESTORE YOUR ORIGINAL WORKING LINE:
    pids = [item[0] for item in recs_with_scores]
    
    # Safely add the microgroup column if it is missing from the source df_products 
    # to avoid hidden columns or missing key errors
    if 'microgroup' not in df_products.columns:
        df_products['microgroup'] = 0  # or any default value for testing
    
    cols_to_show = [
        'microgroup', 'category_group', 'product_category_name_english', 'avg_unit_price', 
        'avg_review_score', 'avg_delivery_days', 'total_sold',
        'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm',
        'seller_region_top1'
    ]
    sub_df = df_products.loc[pids, [c for c in cols_to_show if c in df_products.columns]].copy()
    
    sub_df = sub_df.rename(columns={
        'microgroup': 'Cluster',
        'category_group': 'Macro-group',
        'product_category_name_english': 'Micro-category',
        'avg_unit_price': 'Price',
        'avg_review_score': 'Rating',
        'avg_delivery_days': 'Delivery (days)',
        'total_sold': 'Sold (units)',
        'product_weight_g': 'Weight (g)',
        'product_length_cm': 'Length (cm)',
        'product_height_cm': 'Height (cm)',
        'product_width_cm': 'Width (cm)',
        'seller_region_top1': 'Seller Region'
    })
    
    score_map = dict(recs_with_scores)
    sub_df.insert(0, 'Score', sub_df.index.map(score_map))
    sub_df.insert(0, 'Product', sub_df.index)
    
    sub_df = sub_df.sort_values(by='Score', ascending=False)

    
    # --- Formatting and CSS Styles (white-space: nowrap prevents text wrapping, shrinking the table) ---
    styled_df = sub_df.style.format({
        'Score': '{:.4f}', 'Price': '{:.2f}', 'Rating': '{:.2f}', 'Delivery (days)': '{:.1f}', 
        'Sold (units)': '{:.0f}', 'Weight (g)': '{:.0f}', 'Length (cm)': '{:.0f}', 
        'Height (cm)': '{:.0f}', 'Width (cm)': '{:.0f}'
    }).set_table_styles([
        {'selector': 'th', 'props': [('padding', '6px'), ('white-space', 'nowrap')]},
        {'selector': 'td', 'props': [('padding', '6px'), ('white-space', 'nowrap')]}
    ])
    
    st.dataframe(styled_df, hide_index=True, use_container_width=True)


# --- Live offline inference of all five approaches for the selected numeric buyer ---
if mode == "User (User ID)":
    if selected_user is None:
        st.write("⬅️ Please enter the customer number in the sidebar to compute recommendations.")
    elif selected_user in full_history:
        history = full_history[selected_user]
        st.info(f"Customer #{selected_user} | Items in history: {len(history)}")
        
        hist_with_dummy_scores = [(p, 1.0) for p in history]
        show_product_features_with_scores("Purchase History", hist_with_dummy_scores)
        
        final_recs_with_scores = []
        
        # Model 1: Swing
        if "Model 1: Alibaba Swing" in selected_model:
            candidates = {}
            for item in history:
                if item in swing_matrix:
                    for sim_item, score in swing_matrix[item].items():
                        if sim_item not in history:
                            candidates[sim_item] = candidates.get(sim_item, 0) + score
            final_recs_with_scores = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:k_recs]

        # Model 2: Calculating cosine similarity of item feature vectors
        elif "Model 2: Content-Based" in selected_model:
            hist_idxs = [pid_to_idx[p] for p in history if p in pid_to_idx]
            if hist_idxs:
                user_sims = np.max(cosine_similarity(x_content[hist_idxs], x_content), axis=0)
                for p in history:
                    if p in pid_to_idx: user_sims[pid_to_idx[p]] = -1.0
                top_idxs = np.argsort(user_sims)[::-1][:k_recs]
                final_recs_with_scores = [(product_list[idx], float(user_sims[idx])) for idx in top_idxs]

        # Model 3: Score Fusion hybrid
        elif "Model 3: Score Fusion" in selected_model:
            hist_idxs = [pid_to_idx[p] for p in history if p in pid_to_idx]
            if hist_idxs:
                user_sims = np.max(cosine_similarity(x_content[hist_idxs], x_content), axis=0)
                s_min, s_max = user_sims.min(), user_sims.max()
                if s_max - s_min > 0: user_sims = (user_sims - s_min) / (s_max - s_min)
                for item in history:
                    if item in swing_matrix:
                        for target_pid, swing_weight in swing_matrix[item].items():
                            if target_pid in pid_to_idx: user_sims[pid_to_idx[target_pid]] += swing_weight * 3.0
                for p in history:
                    if p in pid_to_idx: user_sims[pid_to_idx[p]] = -1.0
                top_idxs = np.argsort(user_sims)[::-1][:k_recs]
                final_recs_with_scores = [(product_list[idx], float(user_sims[idx])) for idx in top_idxs]

        # Model 4: Two-Tower via PyTorch tensors
        elif "Model 4: Two-Tower" in selected_model:
            rem_indices_shifted = [pid_to_idx_shifted[p] for p in history if p in pid_to_idx_shifted]
            if rem_indices_shifted:
                hist_input = rem_indices_shifted[-10:]
                if len(hist_input) < 10:
                    hist_input = list(hist_input) + [0] * (10 - len(hist_input))
                    
                with torch.no_grad():
                    user_embed = model_pytorch.forward_user(torch.tensor([hist_input], dtype=torch.long), x_item_tensor_extended)
                    scores = torch.matmul(user_embed, all_item_embeddings.T).squeeze(0).cpu().numpy()
                    
                for p in history:
                    if p in pid_to_idx_shifted: scores[pid_to_idx_shifted[p]] = -1.0
                scores[0] = -1.0
                top_idxs = np.argsort(scores)[::-1][:k_recs]
                final_recs_with_scores = [(idx_to_pid_shifted[idx], float(scores[idx])) for idx in top_idxs if idx in idx_to_pid_shifted]

        # Model 5: CatBoostRanker using the feature matrix
        elif "Model 5: CatBoostRanker" in selected_model:
            hist_idxs = [pid_to_idx[p] for p in history if p in pid_to_idx]
            if hist_idxs:
                user_sims = np.max(cosine_similarity(x_content[hist_idxs], x_content), axis=0)
                for p in history:
                    if p in pid_to_idx: user_sims[pid_to_idx[p]] = -1.0
                # Step 1: Standard selection of 100 first-level candidates
                top_100_idxs = np.argsort(user_sims)[::-1][:100]
                candidate_pids = [product_list[idx] for idx in top_100_idxs]
                
                if cb_model is None:
                    # If the .cbm weights file is missing, return the baseline top list
                    final_recs_with_scores = [(product_list[idx], float(user_sims[idx] * 1.5)) for idx in top_100_idxs[:k_recs]]
                else:
                    features_list = [
                        'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm',
                        'total_sold', 'total_revenue', 'sales_per_month', 'avg_unit_price', 
                        'avg_review_score', 'avg_delivery_days', 'seller_region_top1', 'customer_region_top1'
                    ]
                    # Extract candidate rows and retain only the required features in the correct column order
                    X_cand = df_products.loc[candidate_pids, [c for c in features_list if c in df_products.columns]].copy()
                    
                    # Rename region columns to match the Pool structure used during training
                    X_cand = X_cand.rename(columns={'seller_region_top1': 'seller_region', 'customer_region_top1': 'customer_region'})
                    
                    # Call the original YetiRank tree-based prediction method from CatBoost
                    preds = cb_model.predict(X_cand)
                    
                    # Map the model predictions to product names and extract Top-K
                    cb_scores = list(zip(candidate_pids, [float(p) for p in preds]))
                    final_recs_with_scores = sorted(cb_scores, key=lambda x: x[1], reverse=True)[:k_recs]

        if final_recs_with_scores:
            show_product_features_with_scores("Recommended Complementary Products", final_recs_with_scores)
    else:
        st.warning(f"Customer #{selected_user} not found")

# --- Offline score calculation based on cosine distance for the product search mode ---
else:
    st.info(f"Selected anchor product: {selected_product}")
    show_product_features_with_scores("Product Characteristics", [(selected_product, 1.0)])
    
    final_recs_with_scores = []
    if selected_product in pid_to_idx:
        prod_idx = pid_to_idx[selected_product]
        prod_sims = cosine_similarity(x_content[[prod_idx]], x_content).flatten()
        prod_sims[prod_idx] = -1.0
        
        top_idxs = np.argsort(prod_sims)[::-1][:k_recs]
        final_recs_with_scores = [(product_list[idx], float(prod_sims[idx])) for idx in top_idxs]
        
    if final_recs_with_scores:
        show_product_features_with_scores("Similar Products", final_recs_with_scores)


