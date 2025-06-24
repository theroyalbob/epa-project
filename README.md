# ============================================================
#  NFL State-Space Power-Ratings – ENGINEERING SPECIFICATION
#  Data source: nfl_data_py (play-by-play)  ➜ 1999-present
# ============================================================

──────────────────────────────────────────────────────────────
1 · ENVIRONMENT & DEPENDENCIES
──────────────────────────────────────────────────────────────
• Python 3.10+
• poetry or pip-tools for locking     
• Core libs
    pandas>=2.2
    numpy>=1.26
    scikit-learn>=1.5
    scipy>=1.13
    nfl_data_py>=0.4           # play-by-play import  :contentReference[oaicite:0]{index=0}
    filterpy>=1.4              # Kalman filter utils
    joblib                     # caching serialized design matrices
• Optional (GPU/Bayes later)
    pymc>=5  OR  stan-py

Create a `pyproject.toml`; include [tool.black], [tool.isort] sections.

──────────────────────────────────────────────────────────────
2 · REPO LAYOUT
──────────────────────────────────────────────────────────────
src/
│
├── data_ingest.py            # downloads + caches pbp, cap tables, snaps
├── continuity.py             # computes multiple continuity metrics
├── srs_ridge.py              # opponent-adjusted additive model (ridge)
├── kalman_model.py           # state-space build + EM fitting
├── train.py                  # blocked CV, hyper-grid, Bayes-opt loop
├── predict.py                # weekly update; emits JSON power ratings
├── cli.py                    # `python -m cli fetch|train|update|score`
│
├── config/
│   ├── paths.yaml            # raw/processed/cache dirs
│   └── hyper_grid.yaml       # φ, σ, γ search bounds
│
└── notebooks/                # EDA, sanity plots (non-prod)

Tests in  `tests/`  (pytest & hypothesis).

──────────────────────────────────────────────────────────────
3 · DATA LAYER
──────────────────────────────────────────────────────────────
3.1  Play-by-play
     pbp = nfl.import_pbp_data(list(range(1999, 2025)))        # nfl_data_py
     Persist season-wise parquet to `data/raw/pbp_{year}.pq`.

3.2  Cap tables
     Pull OTC or Spotrac CSV exports manually for now ➜
     place in `data/raw/cap_hits_{year}.csv`.

3.3  Snap counts
     Use nfl_data_py roster_weekly → aggregate player-week snaps.

3.4  Scheduled caching
     In `data_ingest.fetch_all()`, skip download if parquet exists & hash ok.

──────────────────────────────────────────────────────────────
4 · CONTINUITY METRICS ( continuity.py )
──────────────────────────────────────────────────────────────
For each team-season-unit  u ∈ {QB, OFF_EX_QB, DEF}  build:
    • cap_cont         = returning_cap / unit_cap
    • snap_cont        = returning_snaps / unit_snaps
    • perf_cont        = returning_EPA / unit_EPA
    • coach_flag       = {HC, OC, DC} return booleans
    • scheme_cosine    = cosine_similarity of PROE vectors (t-1 vs t)

Expose helper →  `get_continuity_df(year) -> pd.DataFrame`
All metrics normalized to [0,1].

──────────────────────────────────────────────────────────────
5 · OPPONENT-ADJUSTED EPA ( srs_ridge.py )
──────────────────────────────────────────────────────────────
Per season:
    1. Build game-unit matrix  X  ( +1 for offense, –1 for defense ).
    2. Ridge solve  (λ=0.1 default)  to obtain O_rating, D_rating.
    3. Store weekly rolling ratings to feed Kalman.

Unit split convention:
    qb_epa   = qb_dropback_epa
    off_ex_qb= team_off_epa – qb_epa
    def_epa  = defensive_epa  (neg values are better)

──────────────────────────────────────────────────────────────
6 · STATE-SPACE MODEL ( kalman_model.py )
──────────────────────────────────────────────────────────────
State vector θ_t  length =  1(HFA) + 3*32 units = 97.

Observation:   y_g = X_g θ_t + ε_g,   ε_g ~ N(0, σ_y²)

Process:       θ_t = Φ θ_{t-1} + Γ · cont_index + η_t
               η_t ~ N(0, Σ_η)   (block-diag 3×32)

• Φ   = diag([φ_QB, φ_OFFXQB, φ_DEF] repeated)   
• Σ_η = diag([σ_QB², σ_OFFXQB², σ_DEF²] repeated)

Continuity enters **additively** via Γ = diag([γ_QB, γ_OFFXQB, γ_DEF]).

Implement:
    fit_em(y, X, cont, Φ_init, Σ_init) -> Φ*, Σ*, Γ*
    kalman_predict_update()   # single-week online step

──────────────────────────────────────────────────────────────
7 · TRAINING / CV ( train.py )
──────────────────────────────────────────────────────────────
• Blocked split:
      1999-2014  → grid search  φ, σ, γ, λ (ridge)
      2015-2019  → validation (choose single hyper-set)
      2020-2024  → hold-out reporting

• Objective: cumulative log-loss vs actual point margin
              secondary: MAE vs Vegas close (import when available).

• Bayes-opt (TPE) over top 10 % grid survivors.

Freeze hyper-set after 2019.

──────────────────────────────────────────────────────────────
8 · LIVE UPDATE ( predict.py )
──────────────────────────────────────────────────────────────
Run weekly:
    a. `fetch_weekly_pbp(week)`  ➜ append parquet
    b. `update_ridge(week)`      ➜ recalc last-season overlap
    c. `kalman_predict_update(week)` ➜ new θ̂ week+1
    d. Output:
         • team_power.json             # total, QB, OFF, DEF means & SDs
         • game_probs_week{n+1}.json   # win prob, spread mean, 68 % CI
Schedules via cron or GitHub Actions.

──────────────────────────────────────────────────────────────
9 · ACCEPTANCE TESTS
──────────────────────────────────────────────────────────────
✔ `pytest -q` must pass end-to-end dummy season (1999 only).  
✔ `train.py` completes full 1999-2024 back-test < 10 min on M2 Mac.  
✔ Live `predict.py` run after Week 4 updates JSON under 30 s.  
✔ Back-test MAE ≤ 5.4 pts (beats naïve Elo baseline by ≥0.3 pp).  
✔ Log-loss on 2020-24 ≤ Vegas + 0.01.

──────────────────────────────────────────────────────────────
10 · EXTENSION HOOKS
──────────────────────────────────────────────────────────────
• Swap additive continuity for process-noise scaling (λ toggle).  
• Plug-in play-caller or OL-pair cohesion as extra state dims.  
• Port to PyMC if posterior intervals required.

# End of spec
