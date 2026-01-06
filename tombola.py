import numpy as np
import random
from typing import List, Set, Tuple

def generate_layout_for_series(col_counts: np.ndarray) -> np.ndarray:
    """
    Generates a 3x9 binary mask with specific column counts and exactly 5 items per row.
    Uses rejection sampling to ensure row constraint is met.
    
    Args:
        col_counts: Array of length 9 with number of cells to fill per column
        
    Returns:
        3x9 binary matrix where 1 indicates a number should be placed
    """
    for _ in range(10000):
        layout = np.zeros((3, 9), dtype=int)
        
        # Fill columns based on counts
        for col, count in enumerate(col_counts):
            if count > 0:
                rows = random.sample(range(3), count)
                for row in rows:
                    layout[row, col] = 1
        
        # Check if all rows have exactly 5 numbers
        if np.all(np.sum(layout, axis=1) == 5):
            return layout
    
    # Fallback: force valid layout if sampling fails
    layout = np.zeros((3, 9), dtype=int)
    for row in range(3):
        cols = random.sample(range(9), 5)
        layout[row, cols] = 1
    return layout

def generate_card() -> np.ndarray:
    """
    Generates a random Tombola card (3x9 grid).
    Each row has exactly 5 numbers, each column has 1-3 numbers.
    Numbers are sorted within each column.
    
    Returns:
        Numpy array (3, 5) containing the numbers in each row.
    """
    card_matrix = np.zeros((3, 9), dtype=int)
    
    # Define number ranges for each column
    col_ranges = []
    for i in range(9):
        if i == 0:
            col_ranges.append(list(range(1, 10)))  # 1-9
        elif i == 8:
            col_ranges.append(list(range(80, 91)))  # 80-90
        else:
            col_ranges.append(list(range(i * 10, (i + 1) * 10)))  # 10-19, 20-29, etc.

    # Create layout: exactly 5 numbers per row
    layout = np.zeros((3, 9), dtype=int)
    for row in range(3):
        cols = random.sample(range(9), 5)
        layout[row, cols] = 1

    # Fill layout with sorted random numbers from appropriate ranges
    for col in range(9):
        count = int(np.sum(layout[:, col]))
        if count > 0:
            # Select random numbers from this column's range
            values = sorted(random.sample(col_ranges[col], count))
            row_indices = np.where(layout[:, col] == 1)[0]
            for idx, row in enumerate(row_indices):
                card_matrix[row, col] = values[idx]

    # Extract just the numbers for each row (compact representation)
    compact_card = np.zeros((3, 5), dtype=int)
    for r in range(3):
        compact_card[r] = card_matrix[r, card_matrix[r] > 0]
    return compact_card

def generate_series() -> List[np.ndarray]:
    """
    Generates a series of 6 cards containing all numbers 1-90 exactly once.
    Each card has 15 numbers.
    
    Distribution strategy:
    - Column 0 (1-9): 9 numbers → 3 cards with 2, 3 cards with 1
    - Columns 1-7 (10-79): 10 numbers each → 4 cards with 2, 2 cards with 1
    - Column 8 (80-90): 11 numbers → 5 cards with 2, 1 card with 1
    
    Returns:
        List of 6 Numpy arrays (3, 5) representing the card series
    """
    # Define column distributions
    col_distributions = []
    col_distributions.append([2, 2, 2, 1, 1, 1])  # Col 0: 9 numbers total
    for _ in range(7):  # Cols 1-7
        col_distributions.append([2, 2, 2, 2, 1, 1])  # 10 numbers each
    col_distributions.append([2, 2, 2, 2, 2, 1])  # Col 8: 11 numbers total
    
    # Find valid distribution where each card has exactly 15 numbers
    counts_matrix = np.zeros((6, 9), dtype=int)
    max_attempts = 10000
    for attempt in range(max_attempts):
        for col in range(9):
            random.shuffle(col_distributions[col])
            counts_matrix[:, col] = col_distributions[col]
        
        # Check if all cards have 15 numbers (5 per row × 3 rows)
        if np.all(np.sum(counts_matrix, axis=1) == 15):
            break
    else:
        # Fallback if no valid distribution found
        print("Warning: Using fallback distribution for series")

    # Create number pools for each column
    number_pools = []
    for i in range(9):
        if i == 0:
            number_pools.append(list(range(1, 10)))
        elif i == 8:
            number_pools.append(list(range(80, 91)))
        else:
            number_pools.append(list(range(i * 10, (i + 1) * 10)))
    
    # Shuffle pools
    for pool in number_pools:
        random.shuffle(pool)

    # Generate 6 cards
    cards = []
    for card_idx in range(6):
        # Generate layout for this card
        layout = generate_layout_for_series(counts_matrix[card_idx])
        card_matrix = np.zeros((3, 9), dtype=int)
        
        # Fill card with numbers from pools
        for col in range(9):
            count = int(counts_matrix[card_idx, col])
            if count > 0:
                # Pop numbers from pool and sort them
                values = sorted([number_pools[col].pop() for _ in range(count)])
                rows = np.where(layout[:, col] == 1)[0]
                for idx, row in enumerate(rows):
                    card_matrix[row, col] = values[idx]
        
        # Compact format
        compact_card = np.zeros((3, 5), dtype=int)
        for r in range(3):
            compact_card[r] = card_matrix[r, card_matrix[r] > 0]
        cards.append(compact_card)
    
    return cards

def simulate_batch(n_sims: int, num_my_cards: int, num_opponents: int, 
                   opponent_cards_avg: int) -> Tuple[np.ndarray, dict]:
    """
    Simulates a batch of Tombola games using vectorized numpy operations.
    This is significantly faster than looping through extractions.
    """
    # --- 1. Setup Economics ---
    card_cost = 0.20
    total_cards = num_my_cards + (num_opponents * opponent_cards_avg)
    total_pot = total_cards * card_cost
    prize_pool = total_pot
    
    prize_percentages = {
        2: 0.10, 3: 0.15, 4: 0.20, 5: 0.25, 15: 0.30
    }
    prizes = {k: prize_pool * v for k, v in prize_percentages.items()}

    # --- 2. Generate Cards for all simulations ---
    # Shape: (n_sims, num_cards, 3, 5)
    # We generate a flat list first then reshape
    
    # My cards
    my_cards_batch = np.zeros((n_sims, num_my_cards, 3, 5), dtype=int)
    for i in range(n_sims):
        if num_my_cards == 6:
            my_cards_batch[i] = np.array(generate_series())
        else:
            for c in range(num_my_cards):
                my_cards_batch[i, c] = generate_card()

    # Opponent cards
    num_opp_total_cards = num_opponents * opponent_cards_avg
    opp_cards_batch = np.zeros((n_sims, num_opp_total_cards, 3, 5), dtype=int)
    # Optimization: Generate random cards faster or just loop (loop is fine for batch setup)
    for i in range(n_sims):
        for c in range(num_opp_total_cards):
            opp_cards_batch[i, c] = generate_card()

    # --- 3. Generate Extractions ---
    # Generate random permutations of 1..90 for each sim
    # Shape: (n_sims, 90)
    extractions = np.argsort(np.random.rand(n_sims, 90), axis=1) + 1

    # Create a lookup table: extraction_time[sim_idx, number-1] = time_index
    # This tells us WHEN each number is extracted (0 to 89)
    extraction_times = np.argsort(extractions, axis=1)

    # --- 4. Map Cards to Extraction Times ---
    # Replace every number on the cards with the time it gets extracted
    # Numbers are 1-90, so we use number-1 for indexing
    
    # Helper to map (N, C, 3, 5) numbers to times using (N, 90) lookup
    def map_to_times(cards, times_lookup):
        # Advanced indexing: times_lookup[batch_idx, card_number_val - 1]
        # We need to broadcast batch_idx across the card dimensions
        batch_indices = np.arange(n_sims)[:, None, None, None]
        return times_lookup[batch_indices, cards - 1]

    my_times = map_to_times(my_cards_batch, extraction_times)
    opp_times = map_to_times(opp_cards_batch, extraction_times)

    # --- 5. Determine Win Times per Row/Card ---
    # Sort the times in each row. 
    # The 2nd sorted time is when Ambo is achieved. 5th is Cinquina.
    my_sorted = np.sort(my_times, axis=3)
    opp_sorted = np.sort(opp_times, axis=3)

    # Extract specific completion times
    # Shape: (n_sims, n_cards, 3)
    # Index 1 = Ambo (2nd num), Index 2 = Terna (3rd), ..., Index 4 = Cinquina (5th)
    
    # Combine my and opp times for global calculation
    # Shape: (n_sims, total_cards_in_game, 3, 5)
    all_sorted = np.concatenate([my_sorted, opp_sorted], axis=1)
    
    # Calculate earliest win time for each prize level across ALL cards in each game
    # Ambo (2 numbers) -> index 1
    # Terna (3 numbers) -> index 2
    # Quaterna (4 numbers) -> index 3
    # Cinquina (5 numbers) -> index 4
    # Tombola (15 numbers) -> max of Cinquina times across the 3 rows
    
    win_times = {}
    win_times[2] = np.min(all_sorted[:, :, :, 1], axis=(1, 2)) # Min over cards and rows
    win_times[3] = np.min(all_sorted[:, :, :, 2], axis=(1, 2))
    win_times[4] = np.min(all_sorted[:, :, :, 3], axis=(1, 2))
    win_times[5] = np.min(all_sorted[:, :, :, 4], axis=(1, 2))
    
    # Tombola: Max of the 3 rows' completion times gives card completion time
    card_completion_times = np.max(all_sorted[:, :, :, 4], axis=2) # (n_sims, n_cards)
    win_times[15] = np.min(card_completion_times, axis=1)

    # --- 6. Calculate Winnings ---
    my_winnings = np.zeros(n_sims)
    
    # Track stats
    stats_wins = {k: np.zeros(n_sims) for k in prizes}
    
    # For each prize level, see who won at the winning time
    for level in [2, 3, 4, 5]:
        target_time = win_times[level][:, None, None] # (n_sims, 1, 1)
        
        # Check my wins: time matches target
        # my_sorted[:, :, :, level-1] is the time each row achieved the level
        my_hits = (my_sorted[:, :, :, level-1] == target_time)
        my_win_count = np.sum(my_hits, axis=(1, 2)) # Sum over cards and rows
        
        # Check all wins (to calculate share)
        all_hits = (all_sorted[:, :, :, level-1] == target_time)
        total_win_count = np.sum(all_hits, axis=(1, 2))
        
        # Add to winnings (avoid division by zero, though total_win_count >= 1 always)
        share = np.zeros_like(my_winnings)
        mask = total_win_count > 0
        share[mask] = (my_win_count[mask] / total_win_count[mask]) * prizes[level]
        
        my_winnings += share
        stats_wins[level] = my_win_count

    # Tombola calculation
    target_time_tomb = win_times[15][:, None]
    
    # My Tombola wins
    # my_card_completion is max of rows
    my_card_completion = np.max(my_sorted[:, :, :, 4], axis=2)
    my_tomb_hits = (my_card_completion == target_time_tomb)
    my_tomb_count = np.sum(my_tomb_hits, axis=1)
    
    # Total Tombola wins
    all_tomb_hits = (card_completion_times == target_time_tomb)
    total_tomb_count = np.sum(all_tomb_hits, axis=1)
    
    share_tomb = np.zeros_like(my_winnings)
    mask_t = total_tomb_count > 0
    share_tomb[mask_t] = (my_tomb_count[mask_t] / total_tomb_count[mask_t]) * prizes[15]
    
    my_winnings += share_tomb
    stats_wins[15] = my_tomb_count

    # --- 7. Results ---
    my_cost = num_my_cards * card_cost
    net_profits = my_winnings - my_cost
    
    # Aggregate stats for return
    stats = {
        'wins_by_level': stats_wins,
        'total_cards': np.full(n_sims, total_cards)
    }
    
    return net_profits, stats

def run_simulation_study(simulations: int = 1000, num_opponents: int = 10, 
                         opponent_cards_avg: int = 2):
    """
    Runs comprehensive simulation study to determine optimal card purchase strategy.
    Family rules: no house cut, same card can win multiple prizes.
    
    Args:
        simulations: Number of games to simulate per card count
        num_opponents: Number of opponents in each game
        opponent_cards_avg: Average cards per opponent
    """
    print("=" * 80)
    print("TOMBOLA PROFIT SIMULATION STUDY (FAMILY RULES)")
    print("=" * 80)
    print(f"Simulations per strategy: {simulations}")
    print(f"Opponents per game: {num_opponents}")
    print(f"Average cards per opponent: {opponent_cards_avg}")
    print(f"House cut: 0% (family game - all money goes to prizes)")
    print(f"Card cost: €0.20")
    print(f"Rules: Same card can win multiple prizes (Ambo, Terna, etc.)")
    print("=" * 80)
    print()
    
    results = []
    
    for num_cards in range(1, 7):
        print(f"Simulating {num_cards} card{'s' if num_cards > 1 else ''} strategy...", end=" ")
        
        # Run batch simulation
        profits, stats = simulate_batch(simulations, num_cards, num_opponents, opponent_cards_avg)
        
        win_stats = {level: stats['wins_by_level'][level] for level in [2, 3, 4, 5, 15]}
        total_cards_list = stats['total_cards']
        
        # Calculate statistics
        avg_profit = np.mean(profits)
        std_profit = np.std(profits)
        min_profit = np.min(profits)
        max_profit = np.max(profits)
        positive_games = sum(1 for p in profits if p > 0)
        win_rate = (positive_games / simulations) * 100
        avg_total_cards = np.mean(total_cards_list)
        
        cost = num_cards * 0.20
        avg_winnings = avg_profit + cost
        
        # Expected share if fair (no variance)
        expected_share = num_cards / avg_total_cards
        expected_return = expected_share * (avg_total_cards * 0.20)
        expected_profit = expected_return - cost
        
        results.append({
            'cards': num_cards,
            'cost': cost,
            'avg_winnings': avg_winnings,
            'avg_profit': avg_profit,
            'std_profit': std_profit,
            'min_profit': min_profit,
            'max_profit': max_profit,
            'win_rate': win_rate,
            'win_stats': win_stats,
            'expected_profit': expected_profit,
            'avg_total_cards': avg_total_cards
        })
        
        print("✓")
    
    # Display results
    print()
    print("=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80)
    print(f"{'Cards':<7} | {'Cost':>7} | {'Avg Win':>9} | {'Avg Profit':>11} | "
          f"{'Std Dev':>9} | {'Win Rate':>9}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['cards']:<7} | €{r['cost']:>6.2f} | €{r['avg_winnings']:>8.2f} | "
              f"€{r['avg_profit']:>10.2f} | €{r['std_profit']:>8.2f} | {r['win_rate']:>8.1f}%")
    
    print()
    print("=" * 80)
    print("THEORETICAL ANALYSIS")
    print("=" * 80)
    print(f"Average total cards per game: {results[0]['avg_total_cards']:.1f}")
    print(f"\nWith no house cut, expected profit should be ≈€0.00 for all strategies.")
    print(f"Small deviations are due to variance in card quality and prize distribution.")
    print()
    
    for r in results:
        share = (r['cards'] / r['avg_total_cards']) * 100
        print(f"{r['cards']} cards: You own {share:.1f}% of cards → "
              f"Theoretical EV = €{r['expected_profit']:.4f}")
    
    print()
    print("=" * 80)
    print("DETAILED STATISTICS")
    print("=" * 80)
    
    win_names = {2: 'Ambo', 3: 'Terna', 4: 'Quaterna', 5: 'Cinquina', 15: 'Tombola'}
    
    for r in results:
        print(f"\n{r['cards']} Card{'s' if r['cards'] > 1 else ''} Strategy:")
        print(f"  Cost per game: €{r['cost']:.2f}")
        print(f"  Average profit: €{r['avg_profit']:.3f} ± €{r['std_profit']:.2f}")
        print(f"  Profit range: €{r['min_profit']:.2f} to €{r['max_profit']:.2f}")
        print(f"  Win rate: {r['win_rate']:.1f}% of games have positive profit")
        print(f"  Average prizes won per game:")
        
        for level in [2, 3, 4, 5, 15]:
            avg_wins = np.mean(r['win_stats'][level])
            # Calculate what percentage of times you win this prize
            times_won = sum(1 for w in r['win_stats'][level] if w > 0)
            win_pct = (times_won / simulations) * 100
            print(f"    {win_names[level]:>10}: {avg_wins:.3f} cards/game ({win_pct:.1f}% of games)")
    
    # Analysis
    print()
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    
    # Check if profits are close to zero (as expected)
    max_abs_profit = max(abs(r['avg_profit']) for r in results)
    
    if max_abs_profit < 0.05:
        print("✓ Results are REALISTIC: All strategies have expected profit ≈ €0.00")
        print("  This is correct for a fair game with no house cut.")
        print("\n  Strategy choice depends on risk tolerance:")
        print("  - Fewer cards = lower variance (safer, but less exciting)")
        print("  - More cards = higher variance (riskier, but bigger wins possible)")
    else:
        print("⚠ Results show unexpected profits - potential simulation issues:")
        for r in results:
            if abs(r['avg_profit']) > 0.05:
                print(f"  - {r['cards']} cards: €{r['avg_profit']:.2f} profit (should be ≈€0.00)")
    
    print("=" * 80)

if __name__ == "__main__":
    # Run simulation with family rules (no house cut)
    run_simulation_study(simulations=5000, num_opponents=10, opponent_cards_avg=4)