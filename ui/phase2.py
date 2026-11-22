"""Phase 2 placeholders. Each owner replaces their section with a real tab once the lecture is covered."""
import streamlit as st

from gt.params import DomainParams

PLAN = [
    ("Module 3: Learning Dynamics", "Wijerathne R. M. N. S.",
     "Fictitious play and regret matching for 2 to 10 fleets over 5,000 to 20,000 days on the 1A fare game. "
     "Plot strategy paths and O(1/√T) regret decay for a converging and a cycling game, and compare learned "
     "payoffs with the 2A minmax floors.",
     "gt/m3_learning.py: fictitious_play(A, B, T), regret_matching(A, B, T) -> history, regret"),
    ("Module 4: Bayesian Games and Central Control", "Karunarathna W. K.",
     "A reverse ride auction with 5 to 20 drivers whose private costs θᵢ are uniform on [θ_min, θ_max]. BNE over "
     "1,000+ Monte Carlo draws, a second-price auction with a truthfulness and individual-rationality plot over "
     "misreports θ̂ᵢ, and Pigouvian tolls τₑ = xₑ* t′ₑ(xₑ*) that remove the 1C Price of Anarchy.",
     "gt/m4_mechanism.py: simulate_bne(n, lo, hi, draws), second_price(bids), pigouvian_tolls(edges, d)"),
    ("Module 5: Coalitional Games and the Shapley Value", "Vinoth V.",
     "Carpool value v(S) = Σ c({i}) − c(S). Test the core with an LP over 2ᴺ − 1 coalition constraints for 8 to 12 "
     "riders, compute exact Shapley fare shares, and use a Monte Carlo permutation estimator (2,000+ permutations) "
     "for 25 to 50 riders. Plot runtime and approximation error.",
     "gt/m5_coalitions.py: carpool_value(S), core_lp(v, N), shapley_exact(v, N), shapley_mc(v, N, samples)"),
    ("Module 6: Adversarial Stress-Testing and RL Collusion", "Wijesekera H. T. J.",
     "Bidding rings against the Module 4 auction, and multi-agent Q-learning fleets on the 2A fare game to test "
     "for algorithmic collusion.",
     "gt/m6_stress.py: bidding_ring(auction, k), q_learning_fleets(A, B, episodes)"),
]


def render(p: DomainParams) -> None:
    st.info("Phase 2 modules are built after the matching lectures. Each owner adds a solver in gt/ and a tab in "
            "ui/, then registers the tab in app.py.")
    for title, who, plan, api in PLAN:
        with st.expander(f"{title}, owner {who}", expanded=False):
            st.write(plan)
            st.code(api, language="text")
