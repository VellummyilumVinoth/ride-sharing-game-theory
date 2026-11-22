# Urban Ride-Sharing Ecosystem: Game-Theoretic Simulation Dashboard

Game Theory (CS5824), Group I. A Streamlit dashboard for the Colombo ride-sharing ecosystem described in our
phased proposal. Phase 1 (Modules 1A, 1B, 1C, 2A, 2B) is working. Phase 2 (Modules 3 to 6) has a placeholder
tab with each owner's plan.

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py               # opens http://localhost:8501
python -m pytest -q                # solver checks + a full-app render test
```

Python 3.10 or newer.

## What we are doing

Ride-hailing in a city is full of strategic tension. Fleets compete on fares, every driver picks the
fastest road, fleets meet each other every day, and the platform sets the rules. We model one ride-sharing
market in Colombo with three kinds of agents:

- **Fleets** F₁, F₂ set fares, choose pickup zones and enter or skip busy areas.
- **Riders** arrive at Q per hour and move to the cheaper fleet with price sensitivity β.
- **The platform** sets the surge multiplier σ, its commission κ and driver bonuses b (and acts as a
  regulator in Phase 2).

Each module studies one tension with one game-theory tool. All payoffs are generated from real domain
quantities (fares in rupees, rides per hour, minutes of travel time) set by the sidebar sliders, so the
dashboard is a simulator of the market, not a generic matrix calculator.

## Concepts by module

**1A. Normal-form games and IESDS.** Two fleets pick a fare at the same time. A fare is *strictly dominated*
if another fare earns more whatever the rival does; *IESDS* removes such fares again and again. What is left
is solved for *Nash equilibria* (no fleet gains by changing its fare alone) by *support enumeration*: guess
which fares each fleet uses, make the rival indifferent between them, and check no unused fare pays more.
A *zero-sum* zone game (one fleet's market-share gain is the other's loss) is solved by linear programming,
which gives each fleet's safest mixed strategy and the value of the game.

**1B. Extensive-form games and SPNE.** Moves happen in order: the platform sets surge, then fleets decide to
enter an event zone, then the platform decides on a bonus. F₂ does not see F₁'s move, so its two decision
points form an *information set*. A *subgame perfect Nash equilibrium* must be an equilibrium in every
subgame, which rules out empty threats. *Generalized backward induction* solves the smallest subgames first
(a normal-form game where an information set is involved) and folds the results up to the root. Example
finding: the platform never pays the bonus, because once fleets have entered, the bonus is pure cost.

**1C. Congestion games and the Price of Anarchy.** Each driver picks the fastest route, and travel time on a
road grows with traffic (the BPR function `t = t0 [1 + 0.15 (x/k)^4]`). The selfish outcome is the *Wardrop
equilibrium*; a central planner would choose the *social optimum*. Their cost ratio is the *Price of
Anarchy* (PoA). *Braess's paradox*: opening a new flyover can make everyone's trip longer.

**2A. Repeated games: geometry.** The 1A fare game is played every day. A fleet's *minmax* value is the lowest
payoff the rival can force on it (pure mᵢ, or mixed μᵢ via an LP). The *feasible set* V holds every
average payoff the fleets can reach, and the *Folk Theorem region* V* keeps those above both minmax floors.

**2B. Trigger strategies.** Under *Grim Trigger* both fleets keep fares high (collude) and punish any
undercut by playing the stage Nash equilibrium for ever. Collusion survives when the fleets are patient
enough: `δ ≥ (dᵢ − uᵢ(a*)) / (dᵢ − eᵢ)`. Payoffs inside V* are reached by repeating a *K-day cycle* of fare
pairs. The *one-shot deviation principle* checks each day of the cycle in an *augmented stage game*, and a
heatmap shows the minimum patience δ needed for every target payoff.

**Phase 2 (planned).** Learning dynamics (fictitious play, regret matching), Bayesian auctions and
mechanism design (second-price auctions, Pigouvian tolls), coalitional games (the core and Shapley-value
carpool fares), and stress tests for collusion (bidding rings, Q-learning fleets).

## Layout

```
app.py                 entry point: shared sidebar + one tab per module
gt/                    solvers (pure Python, no Streamlit), one file per module
  params.py            DomainParams (the shared sliders) and fare levels
  m1a_normal_form.py   payoff generator, IESDS, support enumeration, zero-sum zone LP, envelopes
  m1b_extensive.py     surge/entry game tree, generalized backward induction (SPNE)
  m1c_congestion.py    road network, BPR latency, Wardrop equilibrium, social optimum, PoA, Braess
  m2a_repeated.py      pure/mixed minmax, feasible set V, Folk region V*
  m2b_trigger.py       Grim Trigger, K-step cycles, augmented stage game, min-delta heatmap
ui/                    one Streamlit page per module (tab_1a.py ... tab_2b.py, phase2.py)
tests/                 test_solvers.py (textbook games) and test_app.py (renders every tab)
```

## Who edits what

| Module | Owner | Solver | Page |
|---|---|---|---|
| 1A Normal form and IESDS | Kulathunga W. P. M. W. | `gt/m1a_normal_form.py` | `ui/tab_1a.py` |
| 1B Extensive form and SPNE | Vishmani P. C. | `gt/m1b_extensive.py` | `ui/tab_1b.py` |
| 1C Congestion and PoA | Wijerathne R. M. N. S. | `gt/m1c_congestion.py` | `ui/tab_1c.py` |
| 2A Repeated game geometry | Karunarathna W. K. | `gt/m2a_repeated.py` | `ui/tab_2a.py` |
| 2B Trigger strategies | Vinoth V. | `gt/m2b_trigger.py` | `ui/tab_2b.py` |
| Dashboard architecture | Wijesekera H. T. J. | `gt/params.py` | `app.py`, `ui/common.py` |
| Phase 2 (3, 4, 5, 6) | see `ui/phase2.py` | add `gt/m3_...py` etc. | add a tab and register it in `app.py` |

To add a Phase 2 module: write the solver in `gt/`, a page with a `render(params)` function in `ui/`, add it to
the `tabs` list in `app.py`, and add a test in `tests/test_solvers.py`.

## How the modules connect

- Every payoff comes from the sidebar (`DomainParams`): demand Q, surge σ, cost c, commission κ, price
  sensitivity β, fleet capacity K, road capacity scale and patience δ.
- The fare game `u_i(a) = ((1 − κ) a_i − c) · min{D_i(a), K}` with `D_i = min{Q, max{0, Q/2 − β(a_i − a_j)}}`
  is built once in 1A and reused as the stage game of 2A and 2B.
- 1B uses the 1A support-enumeration solver for the subgames with information sets.
- 1C's network is where Module 4's Pigouvian tolls will plug in.

## Default values and why

- Fares for a 10 km trip: Low Rs 600, Standard Rs 750, Surge Rs 1,000 (Premium Rs 1,200 in the 4 × 4 game);
  m = 5 or 6 spreads levels from Rs 550 to Rs 1,050. All are multiplied by σ.
- β = 1.0 and K = 400 (equal to Q) make undercutting tempting: (Surge, Surge) maximises joint profit but is
  not a stage equilibrium, so collusion needs patience (Grim Trigger needs δ ≥ 0.135 with (Low, Low) as the
  punishment). With K below Q, (Surge, Surge) becomes a stage equilibrium and 2B has nothing to show.
- The 1C corridor (Maharagama to Fort) has 6 road segments plus an optional flyover. Travel times and
  capacities are illustrative, not survey data. Braess's paradox appears between about 1,500 and 3,900
  drivers per hour.

## Checks against textbook results (tests/test_solvers.py)

- Prisoner's Dilemma: IESDS leaves (D, D); Grim Trigger needs δ ≥ 0.2; the 4-round cycle
  (C,C), (D,D), (D,D), (D,D) needs δ ≥ (1/5)^(1/4) ≈ 0.669.
- Matching Pennies: the unique equilibrium is (½, ½); pure minmax 1, mixed minmax 0.
- Rock-Paper-Scissors LP: uniform strategies, value 0. A 6 × 6 game checks 923 support pairs.
- The 1B tree has 10 terminal nodes, 4 decision levels and 2 imperfect-information sets.

## Known limits

- Support enumeration uses equal-size supports, which finds every equilibrium of a non-degenerate game.
  Degenerate games (ties in payoffs) can have extra equilibria it does not list.
- 1C uses continuous flow (allowed by the brief). An atomic version with individual drivers can be added.
- The min-δ heatmap rounds each target to a K-day cycle, so the cycle's average is close to, not exactly,
  the target.
