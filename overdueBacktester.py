"""
Overdue Number Backtester
=========================
Predicts lottery numbers based on the "longest since appeared" strategy.
Tests predictions recursively against the last 30 days of actual results
using different lookback windows (20, 50, 100 raffles).
"""

import sqliteClass
import pandas as pd
import colorama
from colorama import Fore, Style
import datetime as dt

colorama.init(autoreset=True)


class OverdueBacktester:
    """
    Backtest lottery predictions using the "longest since appeared" strategy.
    Predicts the 6 numbers that have gone longest without appearing.
    """
    
    def __init__(self, dbFileName: str, datasetTable: str, raffle: str = "Bonoloto"):
        self.dbFileName = dbFileName
        self.datasetTable = datasetTable
        self.raffle = raffle
        
        # Initialize sqlite database connection
        self.sqlite = sqliteClass.db(
            dbFileName=self.dbFileName,
            datasetTable=self.datasetTable,
            predictionsTable=""  # Not needed for backtesting
        )
        
        print(f"{Fore.CYAN}{'='*80}")
        print(f"{Fore.CYAN}Overdue Number Backtester - {self.raffle}")
        print(f"{Fore.CYAN}Strategy: Predict numbers with longest gap since last appearance")
        print(f"{Fore.CYAN}{'='*80}\n")
    
    def get_overdue_numbers(self, cutoff_date: str, lookback_window: int) -> list:
        """
        Get the 6 numbers that have gone longest without appearing.
        
        Args:
            cutoff_date: Only consider raffles BEFORE this date
            lookback_window: Number of past raffles to consider (20, 50, or 100)
            
        Returns:
            List of 6 predicted numbers (most overdue)
        """
        # First, get the last N raffle dates before the cutoff
        query_dates = f"""
            SELECT DISTINCT RESULT_DATE
            FROM {self.datasetTable}
            WHERE RAFFLE = '{self.raffle}'
            AND RESULT_DATE < '{cutoff_date}'
            ORDER BY RESULT_DATE DESC
            LIMIT {lookback_window}
        """
        recent_dates = self.sqlite.executeQuery(query_dates)
        
        if recent_dates.empty or len(recent_dates) < lookback_window:
            return []  # Not enough historical data
        
        min_date = recent_dates['RESULT_DATE'].iloc[-1]
        max_date = recent_dates['RESULT_DATE'].iloc[0]
        
        # Get the last appearance of each number within the lookback window
        query_overdue = f"""
            WITH NumbersInWindow AS (
                SELECT 
                    NUMBER,
                    MAX(RESULT_DATE) as LAST_APPEARANCE
                FROM {self.datasetTable}
                WHERE RAFFLE = '{self.raffle}'
                AND NUMBER_TYPE IN ('N1', 'N2', 'N3', 'N4', 'N5', 'N6')
                AND RESULT_DATE >= '{min_date}'
                AND RESULT_DATE <= '{max_date}'
                GROUP BY NUMBER
            ),
            AllNumbers AS (
                SELECT DISTINCT NUMBER
                FROM {self.datasetTable}
                WHERE RAFFLE = '{self.raffle}'
                AND NUMBER_TYPE IN ('N1', 'N2', 'N3', 'N4', 'N5', 'N6')
                AND NUMBER BETWEEN 1 AND 49
            )
            SELECT 
                a.NUMBER,
                COALESCE(n.LAST_APPEARANCE, '1900-01-01') as LAST_APPEARANCE,
                CAST(JULIANDAY('{max_date}') - JULIANDAY(COALESCE(n.LAST_APPEARANCE, '1900-01-01')) AS INTEGER) as DAYS_SINCE
            FROM AllNumbers a
            LEFT JOIN NumbersInWindow n ON a.NUMBER = n.NUMBER
            ORDER BY LAST_APPEARANCE ASC, a.NUMBER ASC
            LIMIT 6
        """
        
        df = self.sqlite.executeQuery(query_overdue)
        
        if df.empty:
            return []
        
        return df['NUMBER'].tolist()
    
    def get_actual_numbers(self, date: str) -> set:
        """
        Get the actual numbers drawn on a specific date.
        
        Args:
            date: The draw date
            
        Returns:
            Set of 6 actual numbers
        """
        query = f"""
            SELECT NUMBER
            FROM {self.datasetTable}
            WHERE RAFFLE = '{self.raffle}'
            AND RESULT_DATE = '{date}'
            AND NUMBER_TYPE IN ('N1', 'N2', 'N3', 'N4', 'N5', 'N6')
        """
        df = self.sqlite.executeQuery(query)
        return set(df['NUMBER'].tolist()) if not df.empty else set()
    
    def run_backtest(self, days_to_test: int = 30, lookback_windows: list = [20, 50, 100]):
        """
        Run backtesting for the last N days of actual results.
        
        Args:
            days_to_test: Number of days to test (default 30)
            lookback_windows: List of lookback windows to test
            
        Returns:
            Dictionary with results per lookback window
        """
        print(f"{Fore.YELLOW}Starting Backtest...")
        print(f"{Fore.YELLOW}Testing last {days_to_test} draws with lookback windows: {lookback_windows}\n")
        
        # Get the last N draw dates
        query_dates = f"""
            SELECT DISTINCT RESULT_DATE
            FROM {self.datasetTable}
            WHERE RAFFLE = '{self.raffle}'
            ORDER BY RESULT_DATE DESC
            LIMIT {days_to_test}
        """
        test_dates = self.sqlite.executeQuery(query_dates)['RESULT_DATE'].tolist()
        test_dates.reverse()  # Process chronologically
        
        # Results storage
        all_results = {window: [] for window in lookback_windows}
        detailed_results = []
        
        print(f"{Fore.WHITE}{'Date':<12} | {'Actual Numbers':<25} | ", end="")
        for w in lookback_windows:
            print(f"{'W' + str(w) + ' Pred':<20} | {'Hits':<5} | ", end="")
        print()
        print("-" * (40 + len(lookback_windows) * 30))
        
        for test_date in test_dates:
            actual = self.get_actual_numbers(test_date)
            
            if not actual:
                continue
            
            row_result = {
                'date': test_date,
                'actual': sorted(actual)
            }
            
            print(f"{Fore.WHITE}{test_date:<12} | {str(sorted(actual)):<25} | ", end="")
            
            for window in lookback_windows:
                predicted = self.get_overdue_numbers(test_date, window)
                
                if not predicted:
                    print(f"{'N/A':<20} | {'N/A':<5} | ", end="")
                    continue
                
                hits = len(set(predicted).intersection(actual))
                all_results[window].append(hits)
                row_result[f'pred_{window}'] = predicted
                row_result[f'hits_{window}'] = hits
                
                # Color code based on hits
                if hits >= 3:
                    color = Fore.GREEN
                elif hits >= 2:
                    color = Fore.YELLOW
                elif hits >= 1:
                    color = Fore.CYAN
                else:
                    color = Fore.RED
                
                print(f"{color}{str(sorted(predicted)):<20} | {hits:<5} | ", end="")
            
            print()
            detailed_results.append(row_result)
        
        return all_results, detailed_results
    
    def generate_summary(self, all_results: dict, lookback_windows: list):
        """
        Generate and print a comprehensive summary of backtest results.
        """
        print(f"\n{Fore.MAGENTA}{'='*80}")
        print(f"{Fore.MAGENTA}BACKTEST SUMMARY")
        print(f"{Fore.MAGENTA}{'='*80}\n")
        
        summary_data = []
        
        for window in lookback_windows:
            results = all_results[window]
            if not results:
                continue
            
            total_draws = len(results)
            total_hits = sum(results)
            avg_hits = total_hits / total_draws if total_draws > 0 else 0
            max_hits = max(results) if results else 0
            
            # Hit distribution
            hit_dist = {i: results.count(i) for i in range(7)}
            
            # Accuracy: percentage of draws with at least 1 hit
            draws_with_hits = sum(1 for h in results if h > 0)
            accuracy = (draws_with_hits / total_draws) * 100 if total_draws > 0 else 0
            
            # Expected hits by random chance: 6 picks from 49 numbers, 6 winning
            # Expected = 6 * 6 / 49 ≈ 0.73
            expected_random = 6 * 6 / 49
            
            summary_data.append({
                'window': window,
                'total_draws': total_draws,
                'total_hits': total_hits,
                'avg_hits': avg_hits,
                'max_hits': max_hits,
                'accuracy': accuracy,
                'hit_dist': hit_dist,
                'expected_random': expected_random
            })
            
            print(f"{Fore.CYAN}Lookback Window: {window} Raffles")
            print(f"{Fore.CYAN}{'-'*40}")
            print(f"  Total Draws Tested:    {total_draws}")
            print(f"  Total Hits:            {total_hits}")
            print(f"  Average Hits/Draw:     {avg_hits:.2f}")
            print(f"  Max Hits in One Draw:  {max_hits}")
            print(f"  Draws with >=1 Hit:    {draws_with_hits} ({accuracy:.1f}%)")
            print(f"  Random Expectation:    ~{expected_random:.2f} hits/draw")
            
            improvement = ((avg_hits - expected_random) / expected_random) * 100 if expected_random > 0 else 0
            if improvement > 0:
                print(f"  {Fore.GREEN}Improvement vs Random:  +{improvement:.1f}%")
            else:
                print(f"  {Fore.RED}Improvement vs Random:  {improvement:.1f}%")
            
            print(f"\n  Hit Distribution:")
            for hits in range(7):
                count = hit_dist.get(hits, 0)
                pct = (count / total_draws) * 100 if total_draws > 0 else 0
                bar = '#' * int(pct / 2)
                print(f"    {hits} hits: {count:3d} ({pct:5.1f}%) {bar}")
            print()
        
        # Best performing window
        if summary_data:
            best = max(summary_data, key=lambda x: x['avg_hits'])
            print(f"{Fore.GREEN}{'='*80}")
            print(f"{Fore.GREEN}BEST PERFORMING WINDOW: {best['window']} Raffles")
            print(f"{Fore.GREEN}Average Hits: {best['avg_hits']:.2f} | Accuracy: {best['accuracy']:.1f}%")
            print(f"{Fore.GREEN}{'='*80}")
        
        return summary_data
    
    def predict_next_draw(self, lookback_windows: list = [20, 50, 100]):
        """
        Predict numbers for the next draw using the best performing strategy.
        """
        print(f"\n{Fore.YELLOW}{'='*80}")
        print(f"{Fore.YELLOW}PREDICTIONS FOR NEXT DRAW")
        print(f"{Fore.YELLOW}{'='*80}\n")
        
        # Use tomorrow as cutoff to include all data up to today
        tomorrow = (dt.date.today() + dt.timedelta(days=1)).strftime('%Y-%m-%d')
        
        for window in lookback_windows:
            predicted = self.get_overdue_numbers(tomorrow, window)
            print(f"{Fore.CYAN}Window {window} Raffles: {Fore.WHITE}{sorted(predicted)}")
        
        print()


def main():
    """Main entry point for the backtest."""
    backtester = OverdueBacktester(
        dbFileName="predictions.sqlite",
        datasetTable="raffleDataset",
        raffle="Bonoloto"
    )
    
    # Run the backtest
    all_results, detailed_results = backtester.run_backtest(
        days_to_test=30,
        lookback_windows=[20, 50, 100]
    )
    
    # Generate summary
    backtester.generate_summary(all_results, [20, 50, 100])
    
    # Predict next draw
    backtester.predict_next_draw([20, 50, 100])


if __name__ == "__main__":
    main()
