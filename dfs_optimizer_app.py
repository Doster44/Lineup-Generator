# dfs_optimizer_app.py
import streamlit as st
import pandas as pd
import numpy as np
import random
import io

st.set_page_config(page_title="NFL DFS Optimizer", layout="wide")

# -----------------------------
# Helper functions
# -----------------------------
ROSTER_SLOTS = ["QB","RB1","RB2","WR1","WR2","WR3","TE","FLEX","DEF"]

def normalize_position(pos: str) -> str:
    if not isinstance(pos,str):
        return ""
    pos = pos.upper()
    if "QB" in pos: return "QB"
    if "RB" in pos: return "RB"
    if "WR" in pos: return "WR"
    if "TE" in pos: return "TE"
    if "D" in pos: return "DEF"
    return pos

def build_lineups(players, num_lineups, salary_cap,
                  exposure_cap, min_unique,
                  randomness, stack_size, bring_back):
    results = []
    counts = {pid:0 for pid in players["id"]}
    player_dict = players.set_index("id").to_dict(orient="index")

    def eligible(slot, p):
        if slot.startswith("RB"): return p["position"]=="RB"
        if slot.startswith("WR"): return p["position"]=="WR"
        if slot=="QB": return p["position"]=="QB"
        if slot=="TE": return p["position"]=="TE"
        if slot=="DEF": return p["position"]=="DEF"
        if slot=="FLEX": return p["position"] in ["RB","WR","TE"]
        return False

    for i in range(num_lineups):
        tmp = players.copy()
        tmp["rand_proj"] = tmp["projection"] * np.random.uniform(1-randomness,1+randomness,len(tmp))
        tmp = tmp.sort_values("rand_proj", ascending=False)

        lineup = {}
        used_ids = set()
        salary = 0
        for slot in ROSTER_SLOTS:
            pool = [p for _,p in tmp.iterrows()
                    if p["id"] not in used_ids and eligible(slot,p)
                    and counts[p["id"]] < exposure_cap*num_lineups]
            if not pool:
                break
            choice = pool[0]
            lineup[slot] = choice["id"]
            used_ids.add(choice["id"])
            salary += choice["salary"]

        if len(lineup)==len(ROSTER_SLOTS) and salary<=salary_cap:
            ok=True
            for prev in results:
                overlap = sum([1 for s in ROSTER_SLOTS if prev[s]==lineup[s]])
                if overlap > len(ROSTER_SLOTS)-min_unique:
                    ok=False
                    break
            if ok:
                results.append(lineup)
                for pid in used_ids:
                    counts[pid]+=1

    return results

# -----------------------------
# UI
# -----------------------------
st.title("🏈 NFL DFS Optimizer (FanDuel)")

uploaded_file = st.sidebar.file_uploader("📂 Upload FanDuel Player CSV", type=["csv"])

num_lineups = st.sidebar.slider("Number of Lineups",1,150,20)
salary_cap = st.sidebar.number_input("Salary Cap", value=60000, step=500)
exposure_cap = st.sidebar.slider("Max Exposure (fraction)",0.1,1.0,0.4)
min_unique = st.sidebar.slider("Min Unique Players",0,9,3)
randomness = st.sidebar.slider("Randomness",0.0,0.2,0.08)
stack_size = st.sidebar.selectbox("QB + Pass Catchers", [0,1,2,3], index=1)
bring_back = st.sidebar.selectbox("Bring Back (opponent WR/TE)", [0,1,2], index=0)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if "id" not in df.columns:
        df = df.rename(columns={df.columns[0]:"id"})
    df["position"] = df["position"].apply(normalize_position)
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce").fillna(0).astype(int)
    df["projection"] = pd.to_numeric(df["projection"], errors="coerce").fillna(0.0)

    st.subheader("Player Pool (first 20)")
    st.dataframe(df.head(20))

    if st.button("🚀 Generate Lineups"):
        lineups = build_lineups(df,num_lineups,
                                salary_cap,exposure_cap,
                                min_unique,randomness,
                                stack_size,bring_back)

        if not lineups:
            st.error("No valid lineups built. Try loosening constraints.")
        else:
            st.success(f"Built {len(lineups)} lineups!")

            upload_df = pd.DataFrame(lineups)[ROSTER_SLOTS]
            buf1 = io.StringIO()
            upload_df.to_csv(buf1,index=False)
            st.download_button("⬇️ Download FanDuel Upload CSV",buf1.getvalue(),
                               file_name="lineups_upload.csv",mime="text/csv")

            id_to_row = df.set_index("id").to_dict(orient="index")
            review_rows=[]
            for lp in lineups:
                row={}
                for s in ROSTER_SLOTS:
                    pid=lp[s]
                    pdata=id_to_row.get(pid,{})
                    row[s]=f"{pdata.get('name','')} ({pid})"
                row["Salary"]=sum(id_to_row[pid]["salary"] for pid in lp.values())
                row["Projection"]=round(sum(id_to_row[pid]["projection"] for pid in lp.values()),2)
                review_rows.append(row)
            review_df=pd.DataFrame(review_rows)

            buf2=io.StringIO()
            review_df.to_csv(buf2,index=False)
            st.download_button("⬇️ Download Review CSV",buf2.getvalue(),
                               file_name="lineups_review.csv",mime="text/csv")

            st.subheader("Preview of Review CSV")
            st.dataframe(review_df.head())
