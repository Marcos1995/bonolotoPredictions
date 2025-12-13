"""
Maximum Hits Predictor
======================
Predicts lottery numbers by combining the best-performing parameter
configurations from the optimizer cache. Focuses on maximizing 3+ hits
(prize-winning threshold) for upcoming draws.

Tests predictions against the most recent draws to validate effectiveness.
"""

import sqliteClass
import pandas as pd
import numpy as np
import colorama
from colorama import Fore, Style
import datetime as dt
from collections import defaultdict
import json

colorama.init(autoreset=True)


class MaximumHitsPredictor:
    """
    Leverages cached optimizer results to generate predictions that
    maximize the probability of achieving 3+ hits.
    """
    
    def __init__(self, dbFileName: str, datasetTable: str, raffle: str = "Bonoloto"):
        self.dbFileName = dbFileName
        self.datasetTable = datasetTable
        self.raffle = raffle
        
        self.sqlite = sqliteClass.db(
            dbFileName=self.dbFileName,
            datasetTable=self.datasetTable,
            predictionsTable=""
        )
        self.sqlite.quiet = True
        
        print(f"{Fore.CYAN}{'='*80}")
        print(f"{Fore.CYAN}Maximum Hits Predictor - {self.raffle}")
        print(f"{Fore.CYAN}Optimized for 3+ hits (prize-winning threshold)")
        print(f"{Fore.CYAN}{'='*80}\n")
    
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
    
    def get_top_cached_params(self, limit: int = 10) -> list:
        """Get the top performing parameter combinations from optimizer cache."""
        query = f"""
            SELECT PARAMS_JSON, PRIZE_SCORE, HITS_3, HITS_4, HITS_5, HITS_6, 
                   AVG_HITS, PRIZE_DRAWS, DAYS_TESTED
            FROM optimizer_cache
            WHERE RAFFLE = '{self.raffle}'
            ORDER BY PRIZE_SCORE DESC
            LIMIT {limit}
        """
        try:
            df = self.sqlite.executeQuery(query)
            if df.empty:
                print(f"{Fore.YELLOW}No cached results found. Using default parameters.")
                return self._get_default_params()
            
            params_list = []
            for _, row in df.iterrows():
                params = json.loads(row['PARAMS_JSON'])
                params['_score'] = row['PRIZE_SCORE']
                params['_prize_draws'] = row['PRIZE_DRAWS']
                params_list.append(params)
            
            return params_list
        except Exception as e:
            print(f"{Fore.YELLOW}Could not read cache: {e}. Using default parameters.")
            return self._get_default_params()
    
    def _get_default_params(self) -> list:
        """Return default high-performing parameter sets."""
        return [
            {'lookback': 50, 'low_max': 16, 'mid_max': 33, 'picks_low': 2, 'picks_mid': 2, 'picks_high': 2, 'prefer_odd_even': 'balanced'},
            {'lookback': 30, 'low_max': 16, 'mid_max': 33, 'picks_low': 2, 'picks_mid': 3, 'picks_high': 1, 'prefer_odd_even': 'balanced'},
            {'lookback': 75, 'low_max': 14, 'mid_max': 30, 'picks_low': 1, 'picks_mid': 3, 'picks_high': 2, 'prefer_odd_even': 'none'},
        ]
    
    def strategy_pattern_based(self, cutoff_date: str, params: dict) -> list:
        """
        Generate predictions using pattern-based strategy with given parameters.
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
                odd_nums = range_nums[range_nums['PARITY'] == 'odd']
                even_nums = range_nums[range_nums['PARITY'] == 'even']
                
                picks_per_parity = num_picks // 2
                remainder = num_picks % 2
                
                if not odd_nums.empty:
                    selected.extend(odd_nums.head(picks_per_parity + remainder)['NUMBER'].tolist())
                if not even_nums.empty:
                    selected.extend(even_nums.head(picks_per_parity)['NUMBER'].tolist())
                
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
    
    def generate_ensemble_prediction(self, cutoff_date: str, top_n: int = 5) -> list:
        """
        Generate prediction by combining votes from top parameter configurations.
        Returns the 6 numbers with most votes across top strategies.
        """
        top_params = self.get_top_cached_params(limit=top_n)
        
        # Count votes for each number
        number_votes = defaultdict(float)
        
        for i, params in enumerate(top_params):
            weight = 1.0 / (i + 1)  # Higher weight for better-ranked params
            predicted = self.strategy_pattern_based(cutoff_date, params)
            
            for num in predicted:
                number_votes[num] += weight
        
        # Sort by votes and return top 6
        sorted_numbers = sorted(number_votes.items(), key=lambda x: x[1], reverse=True)
        return [num for num, votes in sorted_numbers[:6]]
    
    def run_backtest(self, days_to_test: int = 30):
        """
        Run backtest against historical data, showing 3+ hit results.
        """
        print(f"{Fore.YELLOW}Running backtest on last {days_to_test} draws...\n")
        
        # Get test dates
        query = f"""
            SELECT DISTINCT RESULT_DATE
            FROM {self.datasetTable}
            WHERE RAFFLE = '{self.raffle}'
            ORDER BY RESULT_DATE DESC
            LIMIT {days_to_test}
        """
        test_dates = self.sqlite.executeQuery(query)['RESULT_DATE'].tolist()
        test_dates.reverse()  # Process chronologically
        
        hits_list = []
        detailed_results = []
        
        for test_date in test_dates:
            actual = self.get_actual_numbers(test_date)
            if not actual:
                continue
            
            predicted = self.generate_ensemble_prediction(test_date)
            hits = len(set(predicted).intersection(actual))
            hits_list.append(hits)
            
            detailed_results.append({
                'date': test_date,
                'predicted': sorted(predicted),
                'actual': sorted(actual),
                'hits': hits
            })
        
        # Print only 3+ hit results
        print(f"{Fore.GREEN}{'='*80}")
        print(f"{Fore.GREEN}PRIZE-WINNING DRAWS (3+ HITS)")
        print(f"{Fore.GREEN}{'='*80}\n")
        
        prize_draws = [r for r in detailed_results if r['hits'] >= 3]
        
        if prize_draws:
            print(f"{'Date':<12} | {'Hits':<6} | {'Predicted':<35} | {'Actual'}")
            print("-" * 90)
            for r in prize_draws:
                color = Fore.GREEN if r['hits'] >= 4 else Fore.YELLOW
                print(f"{color}{r['date']:<12} | {r['hits']:<6} | {str(r['predicted']):<35} | {r['actual']}")
        else:
            print(f"{Fore.YELLOW}No 3+ hit draws in the test period.")
        
        # Statistics
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.CYAN}BACKTEST STATISTICS")
        print(f"{Fore.CYAN}{'='*80}\n")
        
        hits_6 = sum(1 for h in hits_list if h == 6)
        hits_5 = sum(1 for h in hits_list if h == 5)
        hits_4 = sum(1 for h in hits_list if h == 4)
        hits_3 = sum(1 for h in hits_list if h == 3)
        prize_count = hits_6 + hits_5 + hits_4 + hits_3
        
        print(f"  Total draws tested:  {len(hits_list)}")
        print(f"  Average hits:        {np.mean(hits_list):.2f}")
        print(f"  Maximum hits:        {max(hits_list) if hits_list else 0}")
        print(f"")
        print(f"  {Fore.GREEN}6-Hit (Jackpot):   {hits_6}")
        print(f"  {Fore.GREEN}5-Hit:             {hits_5}")
        print(f"  {Fore.GREEN}4-Hit:             {hits_4}")
        print(f"  {Fore.YELLOW}3-Hit:             {hits_3}")
        print(f"")
        print(f"  {Fore.GREEN}Prize-winning draws: {prize_count} ({prize_count/len(hits_list)*100:.1f}%)")
        
        # Hit distribution
        print(f"\n{Fore.CYAN}Hit Distribution:")
        for h in range(7):
            count = hits_list.count(h)
            pct = count / len(hits_list) * 100 if hits_list else 0
            bar = '#' * int(pct / 2)
            prize_marker = f" {Fore.GREEN}[PRIZE]" if h >= 3 else ""
            print(f"  {h} hits: {count:3d} ({pct:5.1f}%) {bar}{prize_marker}{Style.RESET_ALL}")
        
        return detailed_results
    
    def predict_next_draw(self):
        """
        Generate predictions for the next upcoming draw.
        """
        tomorrow = (dt.date.today() + dt.timedelta(days=1)).strftime('%Y-%m-%d')
        
        print(f"\n{Fore.MAGENTA}{'='*80}")
        print(f"{Fore.MAGENTA}PREDICTION FOR NEXT DRAW")
        print(f"{Fore.MAGENTA}{'='*80}\n")
        
        # Get predictions from top parameter sets
        top_params = self.get_top_cached_params(limit=5)
        
        print(f"{Fore.CYAN}Top Strategy Predictions:")
        print("-" * 60)
        
        all_predictions = []
        for i, params in enumerate(top_params):
            predicted = self.strategy_pattern_based(tomorrow, params)
            all_predictions.append(predicted)
            lookback = params.get('lookback', '?')
            score = params.get('_score', '?')
            print(f"  Strategy {i+1} (LB={lookback}, Score={score}): {sorted(predicted)}")
        
        # Ensemble prediction
        ensemble = self.generate_ensemble_prediction(tomorrow)
        
        print(f"\n{Fore.GREEN}{'='*80}")
        print(f"{Fore.GREEN}RECOMMENDED NUMBERS (Ensemble)")
        print(f"{Fore.GREEN}{'='*80}")
        print(f"\n{Fore.WHITE}  >>> {sorted(ensemble)}")
        
        # Number vote breakdown
        number_votes = defaultdict(float)
        for i, params in enumerate(top_params):
            weight = 1.0 / (i + 1)
            predicted = self.strategy_pattern_based(tomorrow, params)
            for num in predicted:
                number_votes[num] += weight
        
        sorted_votes = sorted(number_votes.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n{Fore.CYAN}Number Confidence Ranking:")
        print(f"{'Number':<8} | {'Score':<8} | Confidence")
        print("-" * 40)
        for num, votes in sorted_votes[:10]:
            confidence = min(100, int(votes * 30))
            bar = '#' * (confidence // 10)
            selected = "*" if num in ensemble else ""
            print(f"{num:<8} | {votes:<8.2f} | {bar} {selected}")
        
        print()
        return ensemble


def main():
    """Main entry point."""
    predictor = MaximumHitsPredictor(
        dbFileName="predictions.sqlite",
        datasetTable="raffleDataset",
        raffle="Bonoloto"
    )
    
    # Run backtest to validate
    predictor.run_backtest(days_to_test=30)
    
    # Generate prediction for next draw
    predictor.predict_next_draw()


if __name__ == "__main__":
    main()
