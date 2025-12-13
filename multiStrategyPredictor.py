"""
Multi-Strategy Lottery Predictor
================================
Implements and compares multiple prediction strategies to find the best approach.
Tests each strategy against historical data and identifies the most accurate.

Strategies:
1. Hot Numbers - Most frequently appearing numbers
2. Cold Numbers - Least frequently appearing (overdue)
3. Balanced - Combination of hot + cold
4. Pattern-Based - Considers range distribution, odd/even balance
5. Weighted Recent - Exponential decay weighting
6. Gap Analysis - Numbers "due" based on average gap between appearances
7. Ensemble - Combines predictions from top-performing strategies
"""

import sqliteClass
import pandas as pd
import numpy as np
import colorama
from colorama import Fore, Style
import datetime as dt
from collections import defaultdict

colorama.init(autoreset=True)


class MultiStrategyPredictor:
    """
    Tests multiple lottery prediction strategies and compares their performance.
    """
    
    def __init__(self, dbFileName: str, datasetTable: str, raffle: str = "Bonoloto", quiet: bool = True):
        self.dbFileName = dbFileName
        self.datasetTable = datasetTable
        self.raffle = raffle
        self.quiet = quiet
        
        self.sqlite = sqliteClass.db(
            dbFileName=self.dbFileName,
            datasetTable=self.datasetTable,
            predictionsTable=""
        )
        
        # Suppress query printing if quiet mode
        if self.quiet:
            self.sqlite.quiet = True
        
        self.strategies = {
            'hot_numbers': self.strategy_hot_numbers,
            'cold_numbers': self.strategy_cold_numbers,
            'balanced': self.strategy_balanced,
            'pattern_based': self.strategy_pattern_based,
            'weighted_recent': self.strategy_weighted_recent,
            'gap_analysis': self.strategy_gap_analysis,
        }
        
        print(f"{Fore.CYAN}{'='*80}")
        print(f"{Fore.CYAN}Multi-Strategy Lottery Predictor - {self.raffle}")
        print(f"{Fore.CYAN}Testing {len(self.strategies)} different strategies")
        print(f"{Fore.CYAN}{'='*80}\n")
    
    # =========================================================================
    # STRATEGY 1: HOT NUMBERS
    # Pick the 6 numbers that appear most frequently
    # =========================================================================
    def strategy_hot_numbers(self, cutoff_date: str, lookback: int = 50) -> list:
        """Pick the most frequently appearing numbers in the last N draws."""
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
            ORDER BY FREQ DESC, NUMBER ASC
            LIMIT 6
        """
        df = self.sqlite.executeQuery(query)
        return df['NUMBER'].tolist() if not df.empty else []
    
    # =========================================================================
    # STRATEGY 2: COLD NUMBERS
    # Pick numbers that haven't appeared in the longest time
    # =========================================================================
    def strategy_cold_numbers(self, cutoff_date: str, lookback: int = 50) -> list:
        """Pick numbers that haven't appeared in the longest time."""
        query = f"""
            WITH RecentDraws AS (
                SELECT DISTINCT RESULT_DATE
                FROM {self.datasetTable}
                WHERE RAFFLE = '{self.raffle}'
                AND RESULT_DATE < '{cutoff_date}'
                ORDER BY RESULT_DATE DESC
                LIMIT {lookback}
            ),
            LastAppear AS (
                SELECT NUMBER, MAX(RESULT_DATE) as LAST_DATE
                FROM {self.datasetTable}
                WHERE RAFFLE = '{self.raffle}'
                AND NUMBER_TYPE IN ('N1','N2','N3','N4','N5','N6')
                AND RESULT_DATE IN (SELECT RESULT_DATE FROM RecentDraws)
                GROUP BY NUMBER
            ),
            AllNumbers AS (
                SELECT DISTINCT NUMBER
                FROM {self.datasetTable}
                WHERE RAFFLE = '{self.raffle}'
                AND NUMBER_TYPE IN ('N1','N2','N3','N4','N5','N6')
                AND NUMBER BETWEEN 1 AND 49
            )
            SELECT a.NUMBER, COALESCE(l.LAST_DATE, '1900-01-01') as LAST_DATE
            FROM AllNumbers a
            LEFT JOIN LastAppear l ON a.NUMBER = l.NUMBER
            ORDER BY LAST_DATE ASC, a.NUMBER ASC
            LIMIT 6
        """
        df = self.sqlite.executeQuery(query)
        return df['NUMBER'].tolist() if not df.empty else []
    
    # =========================================================================
    # STRATEGY 3: BALANCED
    # Combine top 3 hot numbers + top 3 cold numbers
    # =========================================================================
    def strategy_balanced(self, cutoff_date: str, lookback: int = 50) -> list:
        """Combine top 3 hot numbers with top 3 cold numbers."""
        # Get top hot numbers
        query_hot = f"""
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
            LIMIT 3
        """
        hot = self.sqlite.executeQuery(query_hot)['NUMBER'].tolist()
        
        # Get top cold numbers (excluding hot ones)
        hot_str = ','.join(map(str, hot)) if hot else '0'
        query_cold = f"""
            WITH RecentDraws AS (
                SELECT DISTINCT RESULT_DATE
                FROM {self.datasetTable}
                WHERE RAFFLE = '{self.raffle}'
                AND RESULT_DATE < '{cutoff_date}'
                ORDER BY RESULT_DATE DESC
                LIMIT {lookback}
            ),
            LastAppear AS (
                SELECT NUMBER, MAX(RESULT_DATE) as LAST_DATE
                FROM {self.datasetTable}
                WHERE RAFFLE = '{self.raffle}'
                AND NUMBER_TYPE IN ('N1','N2','N3','N4','N5','N6')
                AND RESULT_DATE IN (SELECT RESULT_DATE FROM RecentDraws)
                GROUP BY NUMBER
            ),
            AllNumbers AS (
                SELECT DISTINCT NUMBER
                FROM {self.datasetTable}
                WHERE RAFFLE = '{self.raffle}'
                AND NUMBER_TYPE IN ('N1','N2','N3','N4','N5','N6')
                AND NUMBER BETWEEN 1 AND 49
            )
            SELECT a.NUMBER
            FROM AllNumbers a
            LEFT JOIN LastAppear l ON a.NUMBER = l.NUMBER
            WHERE a.NUMBER NOT IN ({hot_str})
            ORDER BY COALESCE(l.LAST_DATE, '1900-01-01') ASC
            LIMIT 3
        """
        cold = self.sqlite.executeQuery(query_cold)['NUMBER'].tolist()
        
        return hot + cold
    
    # =========================================================================
    # STRATEGY 4: PATTERN-BASED
    # Ensure balanced distribution: low/mid/high + odd/even mix
    # =========================================================================
    def strategy_pattern_based(self, cutoff_date: str, lookback: int = 50) -> list:
        """Select numbers ensuring balanced range and odd/even distribution."""
        # Get frequency data
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
        df = self.sqlite.executeQuery(query)
        
        if df.empty:
            return []
        
        # Split into ranges: low (1-16), mid (17-33), high (34-49)
        df['RANGE'] = df['NUMBER'].apply(
            lambda x: 'low' if x <= 16 else ('mid' if x <= 33 else 'high')
        )
        df['PARITY'] = df['NUMBER'].apply(lambda x: 'odd' if x % 2 == 1 else 'even')
        
        selected = []
        
        # Pick 2 from each range (prioritizing by frequency)
        for range_name in ['low', 'mid', 'high']:
            range_nums = df[df['RANGE'] == range_name].head(3)
            # Try to get one odd and one even
            odd_nums = range_nums[range_nums['PARITY'] == 'odd']
            even_nums = range_nums[range_nums['PARITY'] == 'even']
            
            if not odd_nums.empty:
                selected.append(odd_nums.iloc[0]['NUMBER'])
            if not even_nums.empty:
                selected.append(even_nums.iloc[0]['NUMBER'])
            
            # If we couldn't get both, fill in
            while len([n for n in selected if df[df['NUMBER']==n]['RANGE'].values[0] == range_name]) < 2:
                remaining = range_nums[~range_nums['NUMBER'].isin(selected)]
                if remaining.empty:
                    break
                selected.append(remaining.iloc[0]['NUMBER'])
        
        return selected[:6]
    
    # =========================================================================
    # STRATEGY 5: WEIGHTED RECENT
    # Weight appearances by recency (exponential decay)
    # =========================================================================
    def strategy_weighted_recent(self, cutoff_date: str, lookback: int = 50) -> list:
        """Weight number appearances with exponential decay by recency."""
        # Get all draws with dates
        query = f"""
            WITH RecentDraws AS (
                SELECT DISTINCT RESULT_DATE,
                       ROW_NUMBER() OVER (ORDER BY RESULT_DATE DESC) as DRAW_AGE
                FROM {self.datasetTable}
                WHERE RAFFLE = '{self.raffle}'
                AND RESULT_DATE < '{cutoff_date}'
                ORDER BY RESULT_DATE DESC
                LIMIT {lookback}
            )
            SELECT d.NUMBER, r.DRAW_AGE
            FROM {self.datasetTable} d
            JOIN RecentDraws r ON d.RESULT_DATE = r.RESULT_DATE
            WHERE d.RAFFLE = '{self.raffle}'
            AND d.NUMBER_TYPE IN ('N1','N2','N3','N4','N5','N6')
        """
        df = self.sqlite.executeQuery(query)
        
        if df.empty:
            return []
        
        # Calculate weighted score: weight = e^(-age/20)
        # More recent draws have higher weight
        df['WEIGHT'] = np.exp(-df['DRAW_AGE'] / 20)
        
        # Sum weights by number
        scores = df.groupby('NUMBER')['WEIGHT'].sum().reset_index()
        scores.columns = ['NUMBER', 'SCORE']
        scores = scores.sort_values('SCORE', ascending=False)
        
        return scores.head(6)['NUMBER'].tolist()
    
    # =========================================================================
    # STRATEGY 6: GAP ANALYSIS
    # Pick numbers that are "due" based on their average gap
    # =========================================================================
    def strategy_gap_analysis(self, cutoff_date: str, lookback: int = 100) -> list:
        """Pick numbers that are overdue compared to their average gap."""
        # Get all appearances with dates
        query = f"""
            WITH RecentDraws AS (
                SELECT DISTINCT RESULT_DATE
                FROM {self.datasetTable}
                WHERE RAFFLE = '{self.raffle}'
                AND RESULT_DATE < '{cutoff_date}'
                ORDER BY RESULT_DATE DESC
                LIMIT {lookback}
            ),
            NumberAppearances AS (
                SELECT NUMBER, RESULT_DATE,
                       ROW_NUMBER() OVER (PARTITION BY NUMBER ORDER BY RESULT_DATE DESC) as APP_ORDER
                FROM {self.datasetTable}
                WHERE RAFFLE = '{self.raffle}'
                AND NUMBER_TYPE IN ('N1','N2','N3','N4','N5','N6')
                AND RESULT_DATE IN (SELECT RESULT_DATE FROM RecentDraws)
            )
            SELECT NUMBER, RESULT_DATE, APP_ORDER
            FROM NumberAppearances
            ORDER BY NUMBER, RESULT_DATE
        """
        df = self.sqlite.executeQuery(query)
        
        if df.empty:
            return []
        
        # Get the max date in our window
        max_date_query = f"""
            SELECT MAX(RESULT_DATE) as MAX_DATE
            FROM {self.datasetTable}
            WHERE RAFFLE = '{self.raffle}'
            AND RESULT_DATE < '{cutoff_date}'
        """
        max_date = self.sqlite.executeQuery(max_date_query)['MAX_DATE'].iloc[0]
        
        # Calculate average gap and current gap for each number
        scores = []
        for num in range(1, 50):
            num_data = df[df['NUMBER'] == num]['RESULT_DATE'].tolist()
            
            if len(num_data) < 2:
                # Not enough data, use default
                avg_gap = 10  # default average gap
                current_gap = 100  # assume very overdue
            else:
                # Calculate gaps between appearances
                dates = pd.to_datetime(num_data)
                # diff() on DatetimeIndex returns Timedelta objects directly
                gaps = [d.days for d in dates.diff().dropna()]
                avg_gap = np.mean(gaps) if gaps else 10
                
                # Current gap = days since last appearance
                last_date = dates.max()
                current_gap = (pd.to_datetime(max_date) - last_date).days
            
            # Score = how "overdue" the number is (current_gap / avg_gap)
            due_score = current_gap / max(avg_gap, 1)
            scores.append({'NUMBER': num, 'DUE_SCORE': due_score, 'CURRENT_GAP': current_gap})
        
        scores_df = pd.DataFrame(scores)
        scores_df = scores_df.sort_values('DUE_SCORE', ascending=False)
        
        return scores_df.head(6)['NUMBER'].tolist()
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
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
    
    def run_backtest(self, days_to_test: int = 30, lookback: int = 50):
        """
        Run all strategies against historical data and compare performance.
        """
        print(f"{Fore.YELLOW}Running Backtest ({days_to_test} draws, lookback={lookback})...\n")
        
        # Get test dates
        query = f"""
            SELECT DISTINCT RESULT_DATE
            FROM {self.datasetTable}
            WHERE RAFFLE = '{self.raffle}'
            ORDER BY RESULT_DATE DESC
            LIMIT {days_to_test}
        """
        test_dates = self.sqlite.executeQuery(query)['RESULT_DATE'].tolist()
        test_dates.reverse()
        
        # Track results for each strategy
        results = {name: [] for name in self.strategies.keys()}
        detailed_log = []
        
        for test_date in test_dates:
            actual = self.get_actual_numbers(test_date)
            if not actual:
                continue
            
            day_result = {
                'date': test_date,
                'actual': sorted(actual),
                'predictions': {}
            }
            
            for name, strategy_func in self.strategies.items():
                predicted = strategy_func(test_date, lookback)
                hits = len(set(predicted).intersection(actual))
                results[name].append(hits)
                day_result['predictions'][name] = {
                    'numbers': sorted(predicted),
                    'hits': hits
                }
            
            detailed_log.append(day_result)
        
        # Print detailed results for each day
        self._print_detailed_results(detailed_log)
        
        return results
    
    def _print_detailed_results(self, detailed_log: list):
        """Print detailed predictions for each day."""
        print(f"{Fore.CYAN}{'='*100}")
        print(f"{Fore.CYAN}DETAILED PREDICTIONS BY DAY")
        print(f"{Fore.CYAN}{'='*100}\n")
        
        for day in detailed_log:
            print(f"{Fore.WHITE}Date: {day['date']}")
            print(f"{Fore.GREEN}Actual Numbers: {day['actual']}")
            print(f"{Fore.CYAN}{'-'*60}")
            
            for strategy_name, pred_data in day['predictions'].items():
                hits = pred_data['hits']
                numbers = pred_data['numbers']
                
                # Color based on hits
                if hits >= 3:
                    color = Fore.GREEN
                    stars = '***'
                elif hits >= 2:
                    color = Fore.YELLOW
                    stars = '**'
                elif hits >= 1:
                    color = Fore.CYAN
                    stars = '*'
                else:
                    color = Fore.RED
                    stars = ''
                
                # Mark which numbers were correct
                actual_set = set(day['actual'])
                marked_nums = []
                for n in numbers:
                    if n in actual_set:
                        marked_nums.append(f"[{n}]")
                    else:
                        marked_nums.append(str(n))
                
                print(f"  {color}{strategy_name:<18}: {', '.join(marked_nums):<35} => {hits} hits {stars}")
            
            print()
        
        return detailed_log
    
    def generate_comparison(self, results: dict):
        """Generate comparison report of all strategies."""
        print(f"\n{Fore.MAGENTA}{'='*80}")
        print(f"{Fore.MAGENTA}STRATEGY COMPARISON")
        print(f"{Fore.MAGENTA}{'='*80}\n")
        
        # Random expectation: 6 picks from 49, 6 winning = 6*6/49 = 0.735
        expected_random = 6 * 6 / 49
        
        comparison = []
        for name, hits_list in results.items():
            if not hits_list:
                continue
            
            total_draws = len(hits_list)
            total_hits = sum(hits_list)
            avg_hits = total_hits / total_draws
            max_hits = max(hits_list)
            draws_with_hits = sum(1 for h in hits_list if h > 0)
            accuracy = (draws_with_hits / total_draws) * 100
            improvement = ((avg_hits - expected_random) / expected_random) * 100
            
            comparison.append({
                'name': name,
                'avg_hits': avg_hits,
                'max_hits': max_hits,
                'accuracy': accuracy,
                'improvement': improvement,
                'total_hits': total_hits
            })
        
        # Sort by average hits
        comparison.sort(key=lambda x: x['avg_hits'], reverse=True)
        
        print(f"{'Strategy':<18} | {'Avg Hits':>10} | {'Max':>5} | {'Accuracy':>10} | {'vs Random':>12}")
        print("-" * 65)
        
        for i, s in enumerate(comparison):
            rank_color = Fore.GREEN if i == 0 else (Fore.YELLOW if i == 1 else Fore.WHITE)
            imp_color = Fore.GREEN if s['improvement'] > 0 else Fore.RED
            imp_sign = '+' if s['improvement'] > 0 else ''
            
            print(f"{rank_color}{s['name']:<18} | {s['avg_hits']:>10.2f} | {s['max_hits']:>5} | "
                  f"{s['accuracy']:>9.1f}% | {imp_color}{imp_sign}{s['improvement']:>10.1f}%")
        
        print(f"\n{Fore.CYAN}Random Baseline: {expected_random:.3f} hits/draw")
        
        # Hit distribution summary
        print(f"\n{Fore.YELLOW}{'='*80}")
        print(f"{Fore.YELLOW}HIT DISTRIBUTION BY STRATEGY")
        print(f"{Fore.YELLOW}{'='*80}\n")
        
        # Header
        print(f"{'Strategy':<18}", end="")
        for h in range(7):
            print(f" | {h} hits", end="")
        print(" | Total")
        print("-" * 85)
        
        for name, hits_list in results.items():
            if not hits_list:
                continue
            hit_dist = {i: hits_list.count(i) for i in range(7)}
            total = len(hits_list)
            
            print(f"{name:<18}", end="")
            for h in range(7):
                count = hit_dist.get(h, 0)
                pct = (count / total) * 100 if total > 0 else 0
                print(f" | {count:>5}", end="")
            print(f" | {total:>5}")
        
        print()
        
        return comparison
    
    def create_ensemble_prediction(self, top_n: int = 3) -> dict:
        """
        Create ensemble prediction combining top strategies.
        Returns predicted numbers with confidence scores.
        """
        tomorrow = (dt.date.today() + dt.timedelta(days=1)).strftime('%Y-%m-%d')
        
        print(f"\n{Fore.YELLOW}{'='*80}")
        print(f"{Fore.YELLOW}ENSEMBLE PREDICTION (Next Draw)")
        print(f"{Fore.YELLOW}{'='*80}\n")
        
        # Get predictions from all strategies
        all_predictions = {}
        for name, strategy_func in self.strategies.items():
            pred = strategy_func(tomorrow, 50)
            all_predictions[name] = pred
            print(f"{Fore.CYAN}{name:<18}: {sorted(pred)}")
        
        # Count how many strategies picked each number
        number_votes = defaultdict(int)
        number_strategies = defaultdict(list)
        
        for name, numbers in all_predictions.items():
            for num in numbers:
                number_votes[num] += 1
                number_strategies[num].append(name)
        
        # Sort by votes
        sorted_numbers = sorted(number_votes.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n{Fore.GREEN}{'='*80}")
        print(f"{Fore.GREEN}CONSENSUS RANKING (by strategy agreement)")
        print(f"{Fore.GREEN}{'='*80}\n")
        
        print(f"{'Number':>8} | {'Votes':>6} | {'Picked By'}")
        print("-" * 60)
        
        for num, votes in sorted_numbers[:15]:
            strategies = ', '.join(number_strategies[num][:3])
            if len(number_strategies[num]) > 3:
                strategies += f", +{len(number_strategies[num])-3} more"
            
            vote_bar = '*' * votes
            color = Fore.GREEN if votes >= 4 else (Fore.YELLOW if votes >= 2 else Fore.WHITE)
            print(f"{color}{num:>8} | {votes:>6} | {vote_bar} {strategies}")
        
        # Final recommendation: top 6 by votes
        recommended = [num for num, votes in sorted_numbers[:6]]
        
        print(f"\n{Fore.MAGENTA}{'='*80}")
        print(f"{Fore.MAGENTA}FINAL RECOMMENDATION")
        print(f"{Fore.MAGENTA}{'='*80}")
        print(f"{Fore.GREEN}Best Consensus Numbers: {sorted(recommended)}")
        
        # Also show individual strategy recommendations
        print(f"\n{Fore.CYAN}Or pick based on best strategy:")
        
        return {
            'ensemble': recommended,
            'all_predictions': all_predictions,
            'votes': dict(sorted_numbers)
        }


def main():
    """Main entry point."""
    predictor = MultiStrategyPredictor(
        dbFileName="predictions.sqlite",
        datasetTable="raffleDataset",
        raffle="Bonoloto"
    )
    
    # Run backtest on all strategies
    results = predictor.run_backtest(days_to_test=30, lookback=50)
    
    # Generate comparison report
    comparison = predictor.generate_comparison(results)
    
    # Create ensemble prediction for next draw
    ensemble = predictor.create_ensemble_prediction()
    
    # Show the winner
    if comparison:
        best = comparison[0]
        print(f"\n{Fore.GREEN}{'*'*80}")
        print(f"{Fore.GREEN}BEST STRATEGY: {best['name'].upper()}")
        print(f"{Fore.GREEN}Average Hits: {best['avg_hits']:.2f} | Improvement vs Random: {best['improvement']:+.1f}%")
        print(f"{Fore.GREEN}{'*'*80}")


if __name__ == "__main__":
    main()
