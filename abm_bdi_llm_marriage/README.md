# ABM-BDI-LLM Marriage & Fertility Simulation

A computational social-science platform that simulates **how generative AI diffusion reshapes young people's marriage and fertility behaviour** through three theoretically grounded labour-market mechanisms.

---

## 1. Research Goal

This project investigates the indirect pathway from generative AI diffusion to demographic change:

> Generative AI restructures job tasks, raises entry barriers, accelerates skill churn, and enables platform-based surveillance → youth face delayed career entry, uncertain professional futures, and compressed private time → marriage and fertility decisions shift at the individual level → new aggregate demographic patterns emerge.

The model deliberately avoids the naive assumption that "more AI = lower marriage/fertility rates." Instead it models the *mechanisms*: **employment stability pressure**, **career expectation uncertainty**, and **relationship time compression**, each of which can nonlinearly and heterogeneously affect individual decisions.

---

## 2. Model Logic

```
Generative AI Impact Scenario
  → Individual AI Exposure
    → Job-task restructuring, entry barriers, skill churn,
      platform monitoring, remote collaboration
      → [Mechanism 1] Employment Stability Pressure
      → [Mechanism 2] Career Expectation Uncertainty
      → [Mechanism 3] Relationship Time Compression
        → BDI Belief / Desire / Intention update
        → LLM subjective self-assessment (optional)
          → Marriage-market search
          → Relationship formation
          → Marriage conversion
          → Fertility decision
            → Aggregate demographic patterns emerge
```

---

## 3. Three-Layer Architecture

| Layer | Module | Role |
|-------|--------|------|
| **ABM** | `model.py`, `matching.py`, `fertility.py`, `environment.py` | *How the social world operates* — population dynamics, marriage market encounters, birth events, demographic flow |
| **BDI** | `bdi.py` | *How agents think* — beliefs formed from perceived signals (with memory inertia), desires derived from beliefs (conflicts allowed), intentions selected from competing desires |
| **LLM** | `llm_client.py` | *How agents interpret their situation* — DeepSeek API generates subjective valuations of marriage/fertility; outputs blend with rule-based values; LLM never directly decides behaviour |

**Critical design principle:** The LLM outputs *soft scores* (marriage value, willingness, intention suggestions). These are mixed with rule-based values at configurable weight and feed into probabilistic ABM decision gates. The LLM is a narrative interpreter, not an executive decision-maker.

---

## 4. Three Mechanism Variables

### Mechanism 1 — Employment Stability Pressure (`employment_stability_pressure`)

*AI restructures job tasks and raises entry barriers → stable career start is delayed → economic foundation for family formation is postponed.*

Components: task restructuring exposure, entry barrier level, employment entry delay, employment stability.

**Pathway:** AI diffusion → job-task restructuring & higher entry thresholds → delayed stable career entry → delayed economic basis for marriage/children.

### Mechanism 2 — Career Expectation Uncertainty (`career_expectation_uncertainty`)

*AI accelerates skill obsolescence → career trajectories become unpredictable → long-term life planning (including marriage and children) becomes more cautious.*

Components: reskilling need, career predictability, individual AI exposure.

**Pathway:** AI diffusion → rapid skill churn & opaque career prospects → unstable long-term professional expectations → cautious marriage commitment & childbearing.

### Mechanism 3 — Relationship Time Compression (`relationship_time_compression`)

*AI combines with platform management and remote work to erase work-life boundaries → private and relational time is squeezed → less time and energy for dating, relationship maintenance, and family building.*

Components: monitoring intensity, remote collaboration intensity, work-life boundary blurring, available relationship time.

**Pathway:** AI diffusion + platform monitoring + remote work → continuous availability & hidden labour → eroded private time → reduced dating, marriage, and childbearing probability.

---

## 5. Why Replace "Income Anxiety / Cognitive Anxiety / Time Squeeze"?

The original framing mapped AI shock onto three psychological states. The updated framework shifts to **structural labour-market mechanisms** because:

1. **Employment stability pressure** is not merely "low income" — it captures *whether youth can obtain a stable occupational position at all*, which is a pre-condition for family formation regardless of absolute income level.

2. **Career expectation uncertainty** is not merely "anxiety" — it captures *the objective unpredictability of future career paths and skill advantages*, which rationally makes long-term commitments like marriage and children harder to plan.

3. **Relationship time compression** is not merely "being busy" — it captures *how platform-based management, remote work, and performance surveillance structurally alter the temporal organisation of private life*, not just subjective fatigue.

The updated mechanisms are more directly tied to verifiable empirical indicators and more precisely specify the causal pathway from AI diffusion to demographic change.

---

## 6. Installation

```bash
# Clone or download the repository
cd abm_bdi_llm_marriage

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Install in editable mode
pip install -e ".[dev]"
```

---

## 7. Configuring the DeepSeek API Key

The simulation runs fully in **no-LLM mode** without any API key.

To enable LLM mode:

```bash
cp .env.example .env
# Edit .env and set:
# DEEPSEEK_API_KEY=sk-your-real-key
```

The client uses the OpenAI-compatible endpoint at `https://api.deepseek.com`.

---

## 8. Running in No-LLM Mode (Default)

```bash
python run_experiment.py --no-llm --population-size 500 --years 30 --replications 5 --plot
```

All subjective valuations are generated by heuristic fallback rules. No API key required.

---

## 9. Running in LLM Mode

```bash
python run_experiment.py --use-llm --llm-sample-rate 0.05 --population-size 300 --years 20 --replications 2 --plot
```

- `--llm-sample-rate 0.05` queries only 5 % of agents per tick, controlling API cost.
- Each agent is cached for `llm_refresh_interval` ticks (default 8).
- If the API fails, the system automatically falls back to heuristics and logs the event.
- LLM call statistics (total calls, cache hits, fallbacks, success rate) are printed at the end.

---

## 10. Output Files

| File | Description |
|------|-------------|
| `outputs/scenario_results_long.csv` | Per-tick metrics for every replication and scenario |
| `outputs/scenario_results_summary.csv` | Scenario-level mean ± std of final-tick metrics |
| `outputs/agent_snapshot_final.csv` | Individual agent state at simulation end |
| `outputs/tick_metrics.csv` | Alias of the long-format metrics file |
| `outputs/belief_trajectory.csv` | Tick-level BDI belief trajectory diagnostics |
| `outputs/desire_trajectory.csv` | Tick-level desire strength and conflict diagnostics |
| `outputs/intention_trajectory.csv` | Tick-level intention distribution trajectory |
| `outputs/sensitivity_results.csv` | One-factor sensitivity analysis results |
| `outputs/sensitivity_tornado.png` | Tornado-style chart of scenario-gap sensitivity drivers |
| `outputs/figures/*.png` | Comparison charts (when `--plot` is passed) |

---

## 11. Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--population-size` | 500 | Number of agents |
| `--years` | 30 | Simulation duration (years) |
| `--ticks-per-year` | 4 | Time resolution (4 = quarterly) |
| `--replications` | 3 | Independent runs per scenario |
| `--seed` | 42 | Base random seed (reproducible) |
| `--llm-sample-rate` | 0.05 | Fraction of agents queried by LLM per tick |
| `--llm-model` | `deepseek-chat` | DeepSeek model identifier |
| `delta_school_meeting` | 0.30 | Probability of education-homogamous encounter |
| `beta_commitment` | 0.05 | Per-tick marriage-commitment growth |
| `sigma_age_pressure_female` | 27.0 | Age at which female age-pressure peaks |
| `sigma_age_pressure_male` | 30.0 | Age at which male age-pressure peaks |

---

## 12. Extending the Model

- **New scenarios:** Add a `ScenarioConfig` instance to `src/config.py` and register it in `SCENARIOS`.
- **New mechanisms:** Add variables to `YouthAgent`, update `update_labour_situation()` in `environment.py`, then incorporate them in `bdi.py`, `matching.py`, and `fertility.py`.
- **New metrics:** Add a computation to `compute_metrics()` in `metrics.py` and add the corresponding column to the plotting functions in `plotting.py`.
- **Different LLM provider:** Change `base_url` and `llm_model` in `ModelConfig`; the client uses the OpenAI SDK interface, so any compatible provider works.
- **Policy interventions:** Add tick-conditional logic in `model.py`'s `step()` to simulate policy shocks (e.g., housing subsidies reducing `employment_stability_pressure` after tick N).

---


## 12A. BDI 认知过程与情景平滑校准

### 1) BDI 如何体现认知过程（而非固定规则）

当前 BDI 已从“机制变量→固定加权→意图”升级为动态认知链条：

- **记忆与平滑更新**：belief 更新使用历史窗口滚动感知信号 + `belief_t = (1-rho)*belief_{t-1} + rho*signal_t`。
- **认知解释层**：将客观机制变量转为主观判断（经济成家基础、职业轨迹可预测性、关系投入时间）。
- **愿望冲突保留**：同时保留 marry / delay / career / time-protection 等连续 desire 强度，并输出 `desire_conflict_index`。
- **有限理性意图选择**：加入 intention inertia、reconsider threshold、decision noise、uncertainty tolerance 与 risk preference。
- **反馈更新**：`last_decision_outcome` 与 `cognitive_consistency_score` 反向影响后续感知与意图稳定性。

### 2) 为什么不能让 AI 情景直接大幅改变婚育概率

如果直接把 low/medium/high 映射为大幅概率跳变，会把结构性冲击误当成行为参数重写，导致断崖式结果，且难以与经验数据校准。

### 3) 如何控制情景差异（经验基线 + 渐进冲击 + 有界平滑）

- 所有情景共享同一**经验基线分布**（年龄、教育、收入、职业等初始化）。
- AI 仅作为 baseline 上的**增量 shock**（默认 low=0.00, medium=0.05, high=0.10）。
- 机制影响采用 **exponential damping + clamp**，避免线性无限放大。
- 搜索/约会/结婚/生育概率设定了合理上下限，避免高 AI 情景概率塌缩至 0。
- 机制变量与行为通道采用 `alpha=0.2` 的时间平滑。
- 跨情景比较采用 common random numbers（相同初始种子序列）。
- 结果过度偏离时会触发一次 soft calibration，自动下调惩罚参数。

### 4) 如何查看 BDI 轨迹和敏感性分析

运行实验后查看：

- `outputs/belief_trajectory.csv`
- `outputs/desire_trajectory.csv`
- `outputs/intention_trajectory.csv`
- `outputs/sensitivity_results.csv`
- `outputs/sensitivity_tornado.png`

可据此检查：

- 意图切换是否有惯性、持续期是否合理；
- 愿望冲突是否随机制冲击动态变化；
- 哪些参数导致 low/medium/high 差异过大。

---

## 13. Describing the Model in a Paper

The model can be described as follows:

> We develop an agent-based model (ABM) integrating Belief-Desire-Intention (BDI) cognitive architecture and optional large language model (LLM) assistance to simulate the demographic consequences of generative AI diffusion. Each agent represents a young adult characterised by education, income, AI occupational exposure, and relationship status. At each quarterly tick, a causal environment module updates ten labour-situation variables from a scenario-specific AI-impact parameter set. Three aggregate mechanism variables are derived: **employment stability pressure** (capturing delayed career entry and unstable occupational position), **career expectation uncertainty** (capturing skill churn and opaque career trajectories), and **relationship time compression** (capturing platform-based erosion of private time). A BDI engine updates agents' subjective beliefs, derives competing desires, and selects intentions using a scored deliberation procedure. Optionally, a DeepSeek LLM generates subjective marriage-value and fertility-value scores for a sampled subset of agents, blended with rule-based heuristics at configurable weight. Macro-level demographic outcomes — first marriage rate, first birth rate, average parity, mean ages at first marriage and first birth, and educational assortative mating patterns — emerge from the aggregate of individual probabilistic decisions. Three scenarios (low, medium, high AI diffusion) are compared across multiple stochastic replications using a fixed random seed protocol. Full source code is available at [repository URL].

---

## 14. Running Tests

```bash
# From the abm_bdi_llm_marriage directory
pytest tests/ -v
```

Coverage report:

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

---

## License

MIT
