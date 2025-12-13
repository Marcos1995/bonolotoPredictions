"""
Pattern-Based Strategy Optimizer
================================
Recursively tests different parameter combinations for the pattern-based
strategy to find the optimal configuration.

Parameters tested:
- Lookback window (10, 20, 30, 50, 75, 100)
- Range boundaries (different low/mid/high splits)
- Picks per range (1-1-4, 2-2-2, 1-3-2, etc.)
- Odd/even priority weight
"""

import sqliteClass
import pandas as pd
import numpy as np
import colorama
from colorama import Fore, Style
import datetime as dt
from itertools import product
import json
import hashlib

colorama.init(autoreset=True)


class PatternOptimizer:
    """
    Optimize pattern-based lottery predictions by testing parameter combinations.
    """
    
    def __init__(self, dbFileName: str, datasetTable: str, raffle: str = "Bonoloto"):
        self.dbFileName = dbFileName
        self.datasetTable = datasetTable
        self.raffle = raffle
        self.cache_table = "optimizer_cache"
        
        self.sqlite = sqliteClass.db(
            dbFileName=self.dbFileName,
            datasetTable=self.datasetTable,
            predictionsTable=""
        )
        self.sqlite.quiet = True
        
        # Create cache table if not exists
        self._create_cache_table()
        
        print(f"{Fore.CYAN}{'='*80}")
        print(f"{Fore.CYAN}Pattern-Based Strategy Optimizer - {self.raffle}")
        print(f"{Fore.CYAN}Finding the best parameter combination...")
        print(f"{Fore.CYAN}{'='*80}\n")
    
    def _create_cache_table(self):
        """Create the optimizer cache table if it doesn't exist."""
        query = f"""
            CREATE TABLE IF NOT EXISTS {self.cache_table} (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                RAFFLE VARCHAR(30) NOT NULL,
                TEST_DATE DATE NOT NULL,
                PARAMS_HASH VARCHAR(64) NOT NULL,
                PARAMS_JSON TEXT NOT NULL,
                DAYS_TESTED INTEGER NOT NULL,
                AVG_HITS REAL,
                MAX_HITS INTEGER,
                TOTAL_HITS INTEGER,
                PRIZE_SCORE INTEGER,
                HITS_6 INTEGER,
                HITS_5 INTEGER,
                HITS_4 INTEGER,
                HITS_3 INTEGER,
                PRIZE_DRAWS INTEGER,
                HITS_LIST TEXT,
                CREATED_AT DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(RAFFLE, TEST_DATE, PARAMS_HASH, DAYS_TESTED)
            );
        """
        self.sqlite.executeQuery(query)
        
        # Create index for fast lookups
        index_query = f"""
            CREATE INDEX IF NOT EXISTS idx_cache_lookup 
            ON {self.cache_table} (RAFFLE, TEST_DATE, PARAMS_HASH, DAYS_TESTED);
        """
        self.sqlite.executeQuery(index_query)
    
    def _get_params_hash(self, params: dict) -> str:
        """Generate a unique hash for a parameter combination."""
        # Sort keys for consistent hashing
        params_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()
    
    def get_cached_result(self, test_date: str, params: dict, days_to_test: int) -> dict:
        """
        Check if we have a cached result for this parameter combination.
        Returns the cached result dict or None if not found.
        """
        params_hash = self._get_params_hash(params)
        
        query = f"""
            SELECT AVG_HITS, MAX_HITS, TOTAL_HITS, PRIZE_SCORE,
                   HITS_6, HITS_5, HITS_4, HITS_3, PRIZE_DRAWS, HITS_LIST
            FROM {self.cache_table}
            WHERE RAFFLE = '{self.raffle}'
            AND TEST_DATE = '{test_date}'
            AND PARAMS_HASH = '{params_hash}'
            AND DAYS_TESTED = {days_to_test}
        """
        
        df = self.sqlite.executeQuery(query)
        
        if df.empty:
            return None
        
        row = df.iloc[0]
        return {
            'avg_hits': row['AVG_HITS'],
            'max_hits': row['MAX_HITS'],
            'total_hits': row['TOTAL_HITS'],
            'prize_score': row['PRIZE_SCORE'],
            'hits_6': row['HITS_6'],
            'hits_5': row['HITS_5'],
            'hits_4': row['HITS_4'],
            'hits_3': row['HITS_3'],
            'prize_draws': row['PRIZE_DRAWS'],
            'hits_list': json.loads(row['HITS_LIST']) if row['HITS_LIST'] else [],
            'params': params,
            'from_cache': True
        }
    
    def save_cached_result(self, test_date: str, params: dict, days_to_test: int, result: dict):
        """Save a result to the cache."""
        params_hash = self._get_params_hash(params)
        params_json = json.dumps(params, sort_keys=True).replace("'", "''")
        hits_list_json = json.dumps(result.get('hits_list', [])).replace("'", "''")
        
        # Use INSERT OR REPLACE to handle duplicates
        query = f"""
            INSERT OR REPLACE INTO {self.cache_table} 
            (RAFFLE, TEST_DATE, PARAMS_HASH, PARAMS_JSON, DAYS_TESTED,
             AVG_HITS, MAX_HITS, TOTAL_HITS, PRIZE_SCORE,
             HITS_6, HITS_5, HITS_4, HITS_3, PRIZE_DRAWS, HITS_LIST)
            VALUES (
                '{self.raffle}',
                '{test_date}',
                '{params_hash}',
                '{params_json}',
                {days_to_test},
                {result.get('avg_hits', 0)},
                {result.get('max_hits', 0)},
                {result.get('total_hits', 0)},
                {result.get('prize_score', 0)},
                {result.get('hits_6', 0)},
                {result.get('hits_5', 0)},
                {result.get('hits_4', 0)},
                {result.get('hits_3', 0)},
                {result.get('prize_draws', 0)},
                '{hits_list_json}'
            );
        """
        self.sqlite.executeQuery(query)
    
    def get_frequency_data(self, cutoff_date: str, lookback: int) -> pd.DataFrame:
        """Get number frequency data for a given lookback window."""
        query = f"""
            WITH RecentDraws AS (
                SELECT DISTINCT RESULT_DATE
                FROM {self.datasetTable}
                WHERE RAFFLE = '{self.raffle}'
                AND RESULT_DATE < '{cutoff_date}'
                ORDER BY RESULT_DATE DESC
                LIMIT {lookback}
            )
            SELECT NUMBER, COUNT(*) as FREQ
            FROM {self.datasetTable}
            WHERE RAFFLE = '{self.raffle}'
            AND NUMBER_TYPE IN ('N1','N2','N3','N4','N5','N6')
            AND RESULT_DATE IN (SELECT RESULT_DATE FROM RecentDraws)
            GROUP BY NUMBER
            ORDER BY FREQ DESC
        """
        return self.sqlite.executeQuery(query)
    
    def strategy_pattern_optimized(self, cutoff_date: str, params: dict) -> list:
        """
        Pattern-based strategy with configurable parameters.
        
        params:
            lookback: number of draws to consider
            low_max: upper bound for 'low' range (e.g., 16)
            mid_max: upper bound for 'mid' range (e.g., 33)
            picks_low: numbers to pick from low range
            picks_mid: numbers to pick from mid range
            picks_high: numbers to pick from high range
            prefer_odd_even: 'balanced', 'odd', 'even', or 'none'
        """
        df = self.get_frequency_data(cutoff_date, params['lookback'])
        
        if df.empty:
            return []
        
        low_max = params.get('low_max', 16)
        mid_max = params.get('mid_max', 33)
        picks_low = params.get('picks_low', 2)
        picks_mid = params.get('picks_mid', 2)
        picks_high = params.get('picks_high', 2)
        prefer_odd_even = params.get('prefer_odd_even', 'balanced')
        
        # Split into ranges
        df['RANGE'] = df['NUMBER'].apply(
            lambda x: 'low' if x <= low_max else ('mid' if x <= mid_max else 'high')
        )
        df['PARITY'] = df['NUMBER'].apply(lambda x: 'odd' if x % 2 == 1 else 'even')
        
        selected = []
        picks_config = {'low': picks_low, 'mid': picks_mid, 'high': picks_high}
        
        for range_name, num_picks in picks_config.items():
            range_nums = df[df['RANGE'] == range_name].copy()
            
            if range_nums.empty:
                continue
            
            if prefer_odd_even == 'balanced' and num_picks >= 2:
                # Try to get balanced odd/even
                odd_nums = range_nums[range_nums['PARITY'] == 'odd']
                even_nums = range_nums[range_nums['PARITY'] == 'even']
                
                picks_per_parity = num_picks // 2
                remainder = num_picks % 2
                
                if not odd_nums.empty:
                    selected.extend(odd_nums.head(picks_per_parity + remainder)['NUMBER'].tolist())
                if not even_nums.empty:
                    selected.extend(even_nums.head(picks_per_parity)['NUMBER'].tolist())
                
                # Fill if we didn't get enough
                while len([n for n in selected if n in range_nums['NUMBER'].values]) < num_picks:
                    remaining = range_nums[~range_nums['NUMBER'].isin(selected)]
                    if remaining.empty:
                        break
                    selected.append(remaining.iloc[0]['NUMBER'])
            
            elif prefer_odd_even == 'odd':
                odd_nums = range_nums[range_nums['PARITY'] == 'odd']
                selected.extend(odd_nums.head(num_picks)['NUMBER'].tolist())
                if len(selected) < num_picks:
                    even_nums = range_nums[range_nums['PARITY'] == 'even']
                    selected.extend(even_nums.head(num_picks - len(selected))['NUMBER'].tolist())
            
            elif prefer_odd_even == 'even':
                even_nums = range_nums[range_nums['PARITY'] == 'even']
                selected.extend(even_nums.head(num_picks)['NUMBER'].tolist())
                if len(selected) < num_picks:
                    odd_nums = range_nums[range_nums['PARITY'] == 'odd']
                    selected.extend(odd_nums.head(num_picks - len(selected))['NUMBER'].tolist())
            
            else:  # 'none' - just use frequency
                selected.extend(range_nums.head(num_picks)['NUMBER'].tolist())
        
        return list(set(selected))[:6]
    
    def get_actual_numbers(self, date: str) -> set:
        """Get actual lottery numbers for a specific date."""
        query = f"""
            SELECT NUMBER
            FROM {self.datasetTable}
            WHERE RAFFLE = '{self.raffle}'
            AND RESULT_DATE = '{date}'
            AND NUMBER_TYPE IN ('N1','N2','N3','N4','N5','N6')
        """
        df = self.sqlite.executeQuery(query)
        return set(df['NUMBER'].tolist()) if not df.empty else set()
    
    def test_params(self, params: dict, days_to_test: int = 30, test_date: str = None) -> dict:
        """
        Test a specific parameter combination.
        Uses caching to skip recalculation if already computed for same date/params.
        """
        # Use today's date if not provided (for caching key)
        if test_date is None:
            test_date = dt.date.today().strftime('%Y-%m-%d')
        
        # Check cache first
        cached = self.get_cached_result(test_date, params, days_to_test)
        if cached is not None:
            return cached
        
        # Not cached, calculate
        query = f"""
            SELECT DISTINCT RESULT_DATE
            FROM {self.datasetTable}
            WHERE RAFFLE = '{self.raffle}'
            ORDER BY RESULT_DATE DESC
            LIMIT {days_to_test}
        """
        test_dates = self.sqlite.executeQuery(query)['RESULT_DATE'].tolist()
        test_dates.reverse()
        
        hits_list = []
        for td in test_dates:
            actual = self.get_actual_numbers(td)
            if not actual:
                continue
            
            predicted = self.strategy_pattern_optimized(td, params)
            hits = len(set(predicted).intersection(actual))
            hits_list.append(hits)
        
        if not hits_list:
            return {'avg_hits': 0, 'max_hits': 0, 'total_hits': 0, 'prize_score': 0,
                    'hits_6': 0, 'hits_5': 0, 'hits_4': 0, 'hits_3': 0, 'prize_draws': 0}
        
        # Count hits by category
        hits_6 = sum(1 for h in hits_list if h == 6)
        hits_5 = sum(1 for h in hits_list if h == 5)
        hits_4 = sum(1 for h in hits_list if h == 4)
        hits_3 = sum(1 for h in hits_list if h == 3)
        prize_draws = hits_6 + hits_5 + hits_4 + hits_3
        
        # Weighted prize score: prioritize bigger prizes exponentially
        # 6 hits (jackpot) = 1000 pts, 5 hits = 100 pts, 4 hits = 10 pts, 3 hits = 1 pt
        prize_score = (hits_6 * 1000) + (hits_5 * 100) + (hits_4 * 10) + (hits_3 * 1)
        
        result = {
            'avg_hits': np.mean(hits_list),
            'max_hits': max(hits_list),
            'total_hits': sum(hits_list),
            'prize_score': prize_score,  # Weighted score prioritizing bigger prizes
            'hits_6': hits_6,
            'hits_5': hits_5,
            'hits_4': hits_4,
            'hits_3': hits_3,
            'prize_draws': prize_draws,
            'hits_list': hits_list
        }
        
        # Save to cache
        self.save_cached_result(test_date, params, days_to_test, result)
        
        return result
    
    def optimize(self, days_to_test: int = 30):
        """Run optimization across all parameter combinations."""
        
        # Define parameter search space
        lookback_options = [10, 20, 30, 50, 75, 100, 150]
        range_options = [
            (16, 33),   # Default: 1-16, 17-33, 34-49
            (14, 30),   # Adjusted
            (12, 28),   # Lower splits
            (18, 36),   # Higher splits
            (15, 32),   # Slightly different
            (20, 35),   # More high numbers
        ]
        picks_options = [
            (2, 2, 2),  # Balanced
            (2, 3, 1),  # Mid-heavy
            (1, 3, 2),  # Mid-heavy v2
            (3, 2, 1),  # Low-heavy
            (1, 2, 3),  # High-heavy
            (2, 1, 3),  # High-heavy v2
            (1, 4, 1),  # Mid-focused
        ]
        parity_options = ['balanced', 'odd', 'even', 'none']
        
        # Use yesterday's date as the cache key (most recent complete data)
        today_str = (dt.date.today() - dt.timedelta(days=1)).strftime('%Y-%m-%d')
        
        total_combinations = len(lookback_options) * len(range_options) * len(picks_options) * len(parity_options)
        print(f"{Fore.YELLOW}Testing {total_combinations} parameter combinations...")
        print(f"{Fore.YELLOW}Cache date: {today_str} | Days to test: {days_to_test}\n")
        
        results = []
        best_result = None
        best_score = 0
        
        tested = 0
        cache_hits = 0
        cache_misses = 0
        
        for lookback in lookback_options:
            for (low_max, mid_max) in range_options:
                for (picks_low, picks_mid, picks_high) in picks_options:
                    for parity in parity_options:
                        params = {
                            'lookback': lookback,
                            'low_max': low_max,
                            'mid_max': mid_max,
                            'picks_low': picks_low,
                            'picks_mid': picks_mid,
                            'picks_high': picks_high,
                            'prefer_odd_even': parity
                        }
                        
                        result = self.test_params(params, days_to_test, today_str)
                        
                        # Track cache hits
                        if result.get('from_cache'):
                            cache_hits += 1
                        else:
                            cache_misses += 1
                        
                        result['params'] = params
                        results.append(result)
                        
                        # Score based on weighted prize value (bigger prizes = exponentially higher score)
                        score = result['prize_score']
                        
                        if score > best_score:
                            best_score = score
                            best_result = result
                        
                        tested += 1
                        if tested % 50 == 0:
                            print(f"  Tested {tested}/{total_combinations}... Current best score: {best_result['prize_score']} (6h:{best_result['hits_6']} 5h:{best_result['hits_5']} 4h:{best_result['hits_4']} 3h:{best_result['hits_3']})")
        
        # Sort by prize_score (weighted by prize tier)
        results.sort(key=lambda x: x['prize_score'], reverse=True)
        
        # Print cache statistics
        print(f"\n{Fore.CYAN}Cache Statistics:")
        print(f"  From cache: {cache_hits} | Newly calculated: {cache_misses}")
        if cache_hits > 0:
            print(f"  {Fore.GREEN}Speedup: Skipped {cache_hits} calculations using cached results!")
        
        # Print top 10 results
        print(f"\n{Fore.GREEN}{'='*100}")
        print(f"{Fore.GREEN}TOP 10 PARAMETER COMBINATIONS (Optimized for Biggest Prizes)")
        print(f"{Fore.GREEN}{'='*100}\n")
        
        print(f"{'Rank':<6} | {'Score':>8} | {'6-Hit':>6} | {'5-Hit':>6} | {'4-Hit':>6} | {'3-Hit':>6} | {'Avg':>6} | Parameters")
        print("-" * 110)
        
        for i, res in enumerate(results[:10]):
            p = res['params']
            param_str = f"LB={p['lookback']}, Range=[{p['low_max']},{p['mid_max']}], Picks=({p['picks_low']},{p['picks_mid']},{p['picks_high']}), {p['prefer_odd_even']}"
            print(f"{i+1:<6} | {res['prize_score']:>8} | {res['hits_6']:>6} | {res['hits_5']:>6} | {res['hits_4']:>6} | {res['hits_3']:>6} | {res['avg_hits']:>6.2f} | {param_str}")
        
        # Best result details
        print(f"\n{Fore.MAGENTA}{'='*80}")
        print(f"{Fore.MAGENTA}BEST CONFIGURATION FOUND")
        print(f"{Fore.MAGENTA}{'='*80}\n")
        
        best = results[0]
        p = best['params']
        
        print(f"{Fore.GREEN}Parameters:")
        print(f"  Lookback Window:    {p['lookback']} draws")
        print(f"  Low Range:          1 - {p['low_max']}")
        print(f"  Mid Range:          {p['low_max']+1} - {p['mid_max']}")
        print(f"  High Range:         {p['mid_max']+1} - 49")
        print(f"  Picks (Low/Mid/Hi): {p['picks_low']} / {p['picks_mid']} / {p['picks_high']}")
        print(f"  Odd/Even Pref:      {p['prefer_odd_even']}")
        
        print(f"\n{Fore.CYAN}Performance:")
        print(f"  Prize Score:        {best['prize_score']} points")
        print(f"  6-Hit (Jackpot):    {best['hits_6']}")
        print(f"  5-Hit:              {best['hits_5']}")
        print(f"  4-Hit:              {best['hits_4']}")
        print(f"  3-Hit:              {best['hits_3']}")
        print(f"  Average Hits/Draw:  {best['avg_hits']:.3f}")
        print(f"  Max Hits:           {best['max_hits']}")
        
        # Random baseline comparison
        expected_random = 6 * 6 / 49
        improvement = ((best['avg_hits'] - expected_random) / expected_random) * 100
        print(f"\n  {Fore.GREEN if improvement > 0 else Fore.RED}vs Random: {'+' if improvement > 0 else ''}{improvement:.1f}%")
        
        # Hit distribution for best (highlighting 3+ hits as prize zone)
        if 'hits_list' in best:
            print(f"\n{Fore.YELLOW}Hit Distribution:")
            for h in range(7):
                count = best['hits_list'].count(h)
                pct = count / len(best['hits_list']) * 100
                bar = '#' * int(pct / 2)
                prize_marker = f"{Fore.GREEN}💰" if h >= 3 else ""
                print(f"  {h} hits: {count:3d} ({pct:5.1f}%) {bar} {prize_marker}{Style.RESET_ALL}")
        
        # Predict next draw with best params
        self.predict_with_best(p)
        
        return results
    
    def predict_with_best(self, params: dict):
        """Predict next draw using the best parameters."""
        tomorrow = (dt.date.today() + dt.timedelta(days=1)).strftime('%Y-%m-%d')
        predicted = self.strategy_pattern_optimized(tomorrow, params)
        
        print(f"\n{Fore.GREEN}{'='*80}")
        print(f"{Fore.GREEN}PREDICTION FOR NEXT DRAW (using best params)")
        print(f"{Fore.GREEN}{'='*80}")
        print(f"{Fore.WHITE}Recommended Numbers: {sorted(predicted)}")
        print()


def main():
    """Main entry point."""
    optimizer = PatternOptimizer(
        dbFileName="predictions.sqlite",
        datasetTable="raffleDataset",
        raffle="Bonoloto"
    )
    
    # Run optimization
    results = optimizer.optimize(days_to_test=30)


if __name__ == "__main__":
    main()
