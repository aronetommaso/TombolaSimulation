# 🎄 Tombola Strategy Analysis: A Monte Carlo Study

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Library](https://img.shields.io/badge/Lib-NumPy-orange)
![Type](https://img.shields.io/badge/Project-Monte%20Carlo%20Simulation-green)

> **Author:** Tommaso Arone  
> **Project Type:** Personal Research Project

---

## 🎲 Overview

Tombola is the traditional Italian equivalent of Bingo, played predominantly during Christmas holidays. Unlike commercial gambling, family games typically operate with **Zero House Cut**, meaning 100% of the buy-in money is redistributed as prizes.

This project answers a simple question: **Is there a mathematically optimal strategy for buying cards?**

Does buying 6 cards (a full "cartella") guarantee a better return on investment than buying just 1, or does it simply increase the variance? To answer this, I built a **Vectorized Monte Carlo simulation engine** in Python to simulate over 30,000 games.

---

## ⚙️ Methodology & Rules

The simulation models a realistic family game environment:
* **Players:** 11 total (10 opponents + 1 tracked player).
* **Opponent Behavior:** Opponents buy an average of 4 cards each.
* **House Cut:** 0% (Fair Game).
* **Prize Structure:** Standard Italian rules (Ambo, Terna, Quaterna, Cinquina, Tombola).
* **Scale:** 5,000 iterations per strategy (Total 30,000 games simulated).

### Technical Approach
Instead of standard loops, the engine utilizes **NumPy vectorization** to simulate card generation, number extraction, and prize checking in parallel batches. This allows for high-speed statistical convergence.

---

## 📊 Key Findings

### 1. The "Fair Game" Convergence
The simulation confirmed that in a game with no house cut, **Expected Value (EV) is always €0.00**, regardless of how many cards you buy. 
* **1 Card:** Low investment, low return.
* **6 Cards:** High investment, high return.
* [cite_start]**Net Profit:** Converges to zero for both. [cite: 314]

### 2. Risk vs. Reward (Variance)
While profit expectation is identical, **Risk** is not.
* Buying **1 Card** results in a stable, low-volatility experience (Standard Deviation: €0.70).
* [cite_start]Buying **6 Cards** scales the variance linearly (Standard Deviation: €1.61)[cite: 316]. This means you win big or lose big, but rarely break even.

### 3. Win Rate Reality
Psychologically, buying more cards feels like winning more often. The data backs this up:
* **1 Card Strategy:** You end the game with a profit only **11.5%** of the time.
* [cite_start]**6 Card Strategy:** You end the game with a profit **38.8%** of the time[cite: 316].

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/tombola-simulation.git](https://github.com/yourusername/tombola-simulation.git)
