# dfs_optimizer_app.py
import streamlit as st
import pandas as pd
import numpy as np
import io

st.set_page_config(page_title="NFL DFS Optimizer", layout="wide")

ROSTER_SLOTS = ["QB","RB1","RB2","WR1","WR2","WR3","TE","FLEX","DEF"]

# ---------- helpers ----------
def pick_col(df: pd.DataFrame, candidates, required=False, label=""):
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(
            f"Missing required column for {label}. "
            f"Looked for: {', '.join(candidates)}. "
            f"Check your CSV headers."
        )
    return None

def normalize_position(pos: str) -> str:
    if not isinstance(pos, str):
        return ""
    p = pos.upper().strip()
    if p in ["QB"]: return "QB"
    if p in ["RB"]: return "RB"
    if p in ["WR"]: return "WR"
    if p in ["TE"]: return "TE"
    if p in ["DEF","DST","D/ST"]: return "DEF"
    return p

def build_lineups(players: pd.DataFrame, num_lineups: int,
                  salary_cap: int, exposure_cap: float,
                  min_unique: int, randomness: float):
    results = []
    max_per_player = max(1, int(exposure_cap * num_lineups))
    counts = {pid: 0 for pid in players["id"]}

    def eligible(slot, pos):
        if slot == "QB": return pos == "QB"
        if slot in ("RB1","RB2"): return pos == "RB"
        if slot in ("WR1","WR2","WR3"): return pos == "WR"
        if slot == "TE": return pos == "TE"
        if slot == "DEF": return pos == "DEF"
        if slot == "FLEX": return pos in ("RB","WR","TE")
        return False

    for _ in range(num_lineups):
        tmp = players.copy()
        tmp["rand_proj"] = tmp["projection"] * np.random.uniform(1 - randomness, 1 + randomness, len(tmp))
        tmp.sort_values("rand_proj", ascending=False, inplace=True)

        lineup = {}
        used_ids = set()
        salary = 0

        for slot in ROSTER_SLOTS:
            pool = [r for _, r in tmp.iterrows()
                    if r["id"] not in used_ids
                    and counts[r["id"]] < max_per_player
                    and eligible(slot, r["position"])]

            if not pool:  # relax exposure
                pool = [r for _, r in tmp.iterrows()
                        if r["id"] not in used_ids
                        and eligible(slot, r["position"])]

            if not pool:  # absolute fallback
                pool = [r for _, r in tmp.iterrows() if r["id"] not in used_ids]

            pick = pool[0]
            pid = pick["id"]
            lineup[slot] = pid
            used_ids.add(pid)
            salary += int(pick["salary"])

        # uniqueness check
        ok = True
        for prev in results:
            overlap = sum(1 for s in ROSTER_SLOTS if prev[s] == lineup[s])
            if overlap > len(ROSTER_SLOTS) - min_unique:
                ok = False
                break
        if not ok:
            continue

        lineup["__salary__"] = salary
        results.append(lineup)
        for pid in used_ids:
            counts[pid] += 1

    return results

# ---------- UI ----------
st.title("🏈 NFL DFS Optimizer (FanDuel)")

uploaded = st.sidebar.file_uploader("📂 Upload FanDuel Player CSV", type=["csv"])
num_lineups = st.sidebar.slider("Number of Lineups", 1, 150, 20)
salary_cap = st.sidebar.number_input("Salary Cap", value=60000, step=500)
exposure_cap = st.sidebar.slider("Max Exposure (fraction)", 0.1, 1.0, 0.4)
min_unique = st.sidebar.slider("Min Unique Players", 0, 9, 3)
randomness = st.sidebar.slider("Randomness", 0.0, 0.25, 0.08)

if uploaded is not None:
    df_raw = pd.read_csv(uploaded)

    # Map common FD headings -> required fields
    id_col   = pick_col(df_raw, ["id","Id","ID","Player ID","PlayerID","FID","fd_id"], required=True,  label="player ID")
    pos_col  = pick_col(df_raw, ["position","Position","Roster Position","RosterPosition","Pos","POS"], required=True,  label="position")
    sal_col  = pick_col(df_raw, ["salary","Salary","SAL","Sal","FD Salary","FDSalary"], required=True,  label="salary")
    prj_col  = pick_col(df_raw, ["projection","Projection","Proj","FPPG","AvgPointsPerGame","Points","ProjPoints"], required=True, label="projection")

    # Optional: player display name
    name_col = pick_col(df_raw, ["name","Name","Player","Player Name","Nickname"], required=False)
    if not name_col:
        first = pick_col(df_raw, ["First Name","First","first_name","first"], required=False)
        last  = pick_col(df_raw, ["Last Name","Last","last_name","last"],    required=False)
        if first and last:
            df_raw["__tmp_name__"] = (df_raw[first].fillna("").astype(str) + " " +
                                      df_raw[last].fillna("").astype(str)).str.strip()
            name_col = "__tmp_name__"

    df = pd.DataFrame({
        "id": df_raw[id_col].astype(str),
        "position": df_raw[pos_col].astype(str).map(normalize_position),
        "salary": pd.to_numeric(df_raw[sal_col], errors="coerce").fillna(0).astype(int),
        "projection": pd.to_numeric(df_raw[prj_col], errors="coerce").fillna(0.0),
        "name": df_raw[name_col] if name_col else ""
    })
    df = df[df["position"].isin(["QB","RB","WR","TE","DEF"]) & (df["salary"] > 0) & (df["projection"] > 0)]

    st.subheader("Detected columns")
    st.write({
        "id": id_col, "position": pos_col, "salary": sal_col, "projection": prj_col,
        "name": name_col if name_col else "(none)"
    })

    st.subheader("Player Pool (preview only — full pool used)")
    st.dataframe(df.head(20))

    if st.button("🚀 Generate Lineups"):
        if df.empty:
            st.error("No valid players after parsing. Check your CSV formatting.")
        else:
            lineups = build_lineups(df, num_lineups, salary_cap, exposure_cap, min_unique, randomness)
            if not lineups:
                st.error("No valid lineups built. Loosen constraints or check your CSV.")
            else:
                st.success(f"Built {len(lineups)} lineups!")

                # FanDuel upload CSV (IDs only)
                upload_df = pd.DataFrame(lineups)[ROSTER_SLOTS]
                buf1 = io.StringIO()
                upload_df.to_csv(buf1, index=False)
                st.download_button("⬇️ Download FanDuel Upload CSV", buf1.getvalue(),
                                   file_name="lineups_upload.csv", mime="text/csv")

                # Review CSV (names + IDs + salary + projection)
                id_lu = df.set_index("id").to_dict(orient="index")
                review_rows = []
                for lp in lineups:
                    row = {}
                    for s in ROSTER_SLOTS:
                        pid = lp[s]
                        pdata = id_lu.get(pid, {})
                        disp = pdata.get("name", "")
                        row[s] = pid if not disp else f"{disp} ({pid})"
                    row["Salary"] = int(sum(id_lu[pid]["salary"] for pid in lp.values() if pid in id_lu))
                    row["Projection"] = round(sum(id_lu[pid]["projection"] for pid in lp.values() if pid in id_lu), 2)
                    review_rows.append(row)
                review_df = pd.DataFrame(review_rows)
                buf2 = io.StringIO()
                review_df.to_csv(buf2, index=False)
                st.download_button("⬇️ Download Review CSV", buf2.getvalue(),
                                   file_name="lineups_review.csv", mime="text/csv")

                st.subheader("Preview of Review CSV")
                st.dataframe(review_df.head())
