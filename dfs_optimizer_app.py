# dfs_optimizer_app.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import math
from collections import Counter, defaultdict

st.set_page_config(page_title="NFL DFS Optimizer", layout="wide")

ROSTER_SLOTS = ["QB","RB1","RB2","WR1","WR2","WR3","TE","FLEX","DEF"]
FD_SALARY_CAP = 60000

# ------------ helpers ------------
def pick_col(df: pd.DataFrame, candidates, required=False, label=""):
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(f"Missing required column for {label}. Looked for: {', '.join(candidates)}.")
    return None

def normalize_position(pos: str) -> str:
    if not isinstance(pos, str):
        return ""
    p = pos.upper().strip()
    if p == "QB": return "QB"
    if p == "RB": return "RB"
    if p == "WR": return "WR"
    if p == "TE": return "TE"
    if p in ("DEF","DST","D/ST"): return "DEF"
    return p

def eligible(slot, pos):
    if slot == "QB": return pos == "QB"
    if slot in ("RB1","RB2"): return pos == "RB"
    if slot in ("WR1","WR2","WR3"): return pos == "WR"
    if slot == "TE": return pos == "TE"
    if slot == "DEF": return pos == "DEF"
    if slot == "FLEX": return pos in ("RB","WR","TE")
    return False

def weighted_sample(df: pd.DataFrame, weight_col: str):
    if df.empty:
        return None
    w = df[weight_col].to_numpy()
    w = np.maximum(w, 1e-6)
    sw = w.sum()
    if sw <= 0:
        return df.sample(1).iloc[0]
    probs = w / sw
    idx = np.random.choice(np.arange(len(df)), p=probs)
    return df.iloc[idx]

def cap_ok(lineup_ids, id_to_row, cap):
    total = sum(int(id_to_row[i]["salary"]) for i in lineup_ids if i in id_to_row)
    return total <= cap

def team_count_ok(lineup_ids, id_to_row, max_team):
    cnt = Counter(id_to_row[i]["team"] for i in lineup_ids if i in id_to_row and id_to_row[i]["team"])
    return all(v <= max_team for v in cnt.values())

def overlap_ok(lineup_ids, prev_ids_sets, min_unique):
    need_diff = min_unique
    for s in prev_ids_sets:
        overlap = len(set(lineup_ids) & s)
        if overlap > (len(lineup_ids) - need_diff):
            return False
    return True

def build_lineups(players: pd.DataFrame, *, num_lineups: int, salary_cap: int,
                  exposure_cap: float, min_unique: int, randomness: float,
                  max_team: int, qb_stack: int, bring_back: int,
                  locks: set, bans: set, min_proj: float):
    p = players.copy()
    p = p[~p["id"].isin(bans)]
    p = p[p["projection"] >= float(min_proj)]
    if p.empty:
        return []

    if randomness > 0:
        noise = np.random.uniform(1 - randomness, 1 + randomness, len(p))
        p = p.assign(rand_proj=p["projection"].to_numpy() * noise)
    else:
        p = p.assign(rand_proj=p["projection"].to_numpy())

    id_to_row = p.set_index("id").to_dict(orient="index")
    counts = Counter()
    max_per_player = max(1, int(exposure_cap * num_lineups))

    pos_groups = {pos: p[p["position"] == pos] for pos in ["QB","RB","WR","TE","DEF"]}
    team_index = defaultdict(lambda: defaultdict(pd.DataFrame))
    for pos in ["QB","RB","WR","TE","DEF"]:
        dfp = pos_groups[pos]
        for t, block in dfp.groupby("team"):
            team_index[pos][t] = block

    results = []
    prev_ids_sets = []

    attempts = 0
    target_attempts = max(2000, num_lineups * 200)

    while len(results) < num_lineups and attempts < target_attempts:
        attempts += 1
        lineup = {}
        used_ids = set()

        # Locks
        locks_list = list(locks & set(p["id"]))
        for pid in locks_list:
            pos = id_to_row[pid]["position"]
            placed = False
            for slot in ROSTER_SLOTS:
                if slot not in lineup and eligible(slot, pos) and pid not in used_ids:
                    lineup[slot] = pid
                    used_ids.add(pid)
                    placed = True
                    break
            if not placed:
                lineup = {}
                used_ids = set()
                break
        if not lineup and locks_list:
            continue

        def get_pool(pos):
            dfpos = pos_groups[pos]
            mask = (~dfpos["id"].isin(used_ids)) & (dfpos["id"].apply(lambda i: counts[i] < max_per_player))
            return dfpos[mask]

        # QB
        qb_pool = get_pool("QB")
        if qb_pool.empty:
            qb_pool = pos_groups["QB"][~pos_groups["QB"]["id"].isin(used_ids)]
        if qb_pool.empty:
            continue
        qb_pick = weighted_sample(qb_pool, "rand_proj")
        lineup["QB"] = qb_pick["id"]; used_ids.add(qb_pick["id"])

        qb = id_to_row[lineup["QB"]]
        qb_team = qb.get("team","")
        opp_team = qb.get("opp","")

        # Stack WR/TE
        needed_stack = max(0, qb_stack)
        if needed_stack > 0:
            cand_wr = team_index["WR"].get(qb_team, pd.DataFrame())
            cand_te = team_index["TE"].get(qb_team, pd.DataFrame())
            cand = pd.concat([cand_wr, cand_te]).drop_duplicates(subset=["id"])
            cand = cand[~cand["id"].isin(used_ids)]
            if cand.empty and qb_stack > 0:
                continue
            for _ in range(needed_stack):
                sub = cand[~cand["id"].isin(used_ids)]
                if sub.empty:
                    break
                pick = weighted_sample(sub, "rand_proj")
                placed = False
                for slot in ["WR1","WR2","WR3","TE"]:
                    if slot not in lineup and eligible(slot, id_to_row[pick["id"]]["position"]):
                        lineup[slot] = pick["id"]; used_ids.add(pick["id"]); placed = True; break
                if not placed and "FLEX" not in lineup:
                    lineup["FLEX"] = pick["id"]; used_ids.add(pick["id"])

        # Bring-back
        for _ in range(max(0, bring_back)):
            opp_cand_wr = team_index["WR"].get(opp_team, pd.DataFrame())
            opp_cand_te = team_index["TE"].get(opp_team, pd.DataFrame())
            ob = pd.concat([opp_cand_wr, opp_cand_te]).drop_duplicates(subset=["id"])
            ob = ob[~ob["id"].isin(used_ids)]
            if ob.empty:
                break
            pick = weighted_sample(ob, "rand_proj")
            placed = False
            for slot in ["WR1","WR2","WR3","TE","FLEX"]:
                if slot not in lineup and eligible(slot, id_to_row[pick["id"]]["position"]):
                    lineup[slot] = pick["id"]; used_ids.add(pick["id"]); placed = True; break
            if not placed:
                break

        # Fill remaining
        for slot in ROSTER_SLOTS:
            if slot in lineup:
                continue
            if slot == "QB": pos_needed = "QB"
            elif slot.startswith("RB"): pos_needed = "RB"
            elif slot.startswith("WR"): pos_needed = "WR"
            elif slot == "TE": pos_needed = "TE"
            elif slot == "DEF": pos_needed = "DEF"
            else: pos_needed = None
            if pos_needed:
                pool = get_pool(pos_needed)
            else:
                pool = pd.concat([get_pool("RB"), get_pool("WR"), get_pool("TE")]).drop_duplicates(subset=["id"])
            if pool.empty:
                continue
            pick = weighted_sample(pool, "rand_proj")
            lineup[slot] = pick["id"]; used_ids.add(pick["id"])

        if len(lineup) != len(ROSTER_SLOTS):
            continue

        lineup_ids = [lineup[s] for s in ROSTER_SLOTS]
        if not cap_ok(lineup_ids, id_to_row, salary_cap):
            continue
        if not team_count_ok(lineup_ids, id_to_row, max_team):
            continue
        if not overlap_ok(lineup_ids, prev_ids_sets, min_unique):
            continue

        results.append(lineup)
        prev_ids_sets.append(set(lineup_ids))
        for pid in lineup_ids:
            counts[pid] += 1

    return results

# ------------ UI ------------
st.title("🏈 NFL DFS Optimizer (FanDuel-style)")

uploaded = st.sidebar.file_uploader("📂 Upload FanDuel Player CSV", type=["csv"])

st.sidebar.subheader("Global Settings")
num_lineups = st.sidebar.slider("Number of Lineups", 1, 150, 20)
salary_cap = st.sidebar.number_input("Salary Cap", value=FD_SALARY_CAP, step=500)
randomness = st.sidebar.slider("Randomness (0–0.5)", 0.0, 0.5, 0.15)
exposure_cap = st.sidebar.slider("Max Exposure (fraction)", 0.1, 1.0, 0.5)
min_unique = st.sidebar.slider("Min Unique Players", 0, 9, 3)
max_team = st.sidebar.slider("Max Players per Team", 2, 9, 4)

st.sidebar.subheader("Stacking")
qb_stack = st.sidebar.selectbox("QB + WR/TE", [0,1,2,3], index=1)
bring_back = st.sidebar.selectbox("Bring-back (opp WR/TE)", [0,1,2], index=0)

st.sidebar.subheader("Filters")
min_proj = st.sidebar.number_input("Min Projection", value=0.0, step=0.5)
locks_input = st.sidebar.text_area("Lock IDs (comma-separated)", value="")
bans_input = st.sidebar.text_area("Ban IDs (comma-separated)", value="")

locks = set([x.strip() for x in locks_input.split(",") if x.strip()])
bans = set([x.strip() for x in bans_input.split(",") if x.strip()])

if uploaded is not None:
    df_raw = pd.read_csv(uploaded)
    id_col   = pick_col(df_raw, ["id","Id","ID","Player ID","PlayerID","FID","fd_id"], required=True)
    pos_col  = pick_col(df_raw, ["position","Position","Roster Position","RosterPosition","Pos","POS"], required=True)
    team_col = pick_col(df_raw, ["team","Team","TeamAbbrev"], required=False) or ""
    opp_col  = pick_col(df_raw, ["opp","Opp","Opponent"], required=False) or ""
    sal_col  = pick_col(df_raw, ["salary","Salary","SAL","FD Salary"], required=True)
    prj_col  = pick_col(df_raw, ["projection","Projection","Proj","FPPG"], required=True)
    name_col = pick_col(df_raw, ["name","Name","Player","Player Name"], required=False)

    df = pd.DataFrame({
        "id": df_raw[id_col].astype(str),
        "position": df_raw[pos_col].astype(str).map(normalize_position),
        "team": df_raw[team_col].astype(str).str.upper() if team_col else "",
        "opp": df_raw[opp_col].astype(str).str.upper() if opp_col else "",
        "salary": pd.to_numeric(df_raw[sal_col], errors="coerce").fillna(0).astype(int),
        "projection": pd.to_numeric(df_raw[prj_col], errors="coerce").fillna(0.0),
        "name": df_raw[name_col] if name_col else ""
    })

    df = df[df["position"].isin(["QB","RB","WR","TE","DEF"]) & (df["salary"] > 0) & (df["projection"] > 0)]

    st.subheader("Player Pool (preview only — full pool used)")
    st.dataframe(df.sample(min(len(df), 20)) if len(df) > 0 else df)

    if st.button("🚀 Generate Lineups"):
        lineups = build_lineups(
            df,
            num_lineups=num_lineups,
            salary_cap=salary_cap,
            exposure_cap=exposure_cap,
            min_unique=min_unique,
            randomness=randomness,
            max_team=max_team,
            qb_stack=qb_stack,
            bring_back=bring_back,
            locks=locks,
            bans=bans,
            min_proj=min_proj,
        )
        if not lineups:
            st.error("Could not build lineups. Loosen constraints or check your CSV.")
        else:
            st.success(f"Built {len(lineups)} lineups!")

            # Upload CSV = IDs only
            upload_df = pd.DataFrame(lineups)[ROSTER_SLOTS]
            buf1 = io.StringIO()
            upload_df.to_csv(buf1, index=False)
            st.download_button("⬇️ Download FanDuel Upload CSV", buf1.getvalue(),
                               file_name="lineups_upload.csv", mime="text/csv")

            # Review CSV = names + IDs
            id_lu = df.set_index("id").to_dict(orient="index")
            review_rows = []
            for lp in lineups:
                row = {}
                for s in ROSTER_SLOTS:
                    pid = lp[s]
                    nm = id_lu.get(pid, {}).get("name", "")
                    row[s] = pid if not nm else f"{nm} ({pid})"
                row["Salary"] = int(sum(id_lu.get(pid,{}).get("salary",0) for pid in lp.values()))
                row["Projection"] = round(sum(id_lu.get(pid,{}).get("projection",0.0) for pid in lp.values()), 2)
                review_rows.append(row)
            review_df = pd.DataFrame(review_rows)
            buf2 = io.StringIO()
            review_df.to_csv(buf2, index=False)
            st.download_button("⬇️ Download Review CSV", buf2.getvalue(),
                               file_name="lineups_review.csv", mime="text/csv")

            st.subheader("Preview of Review CSV")
            st.dataframe(review_df.head())

