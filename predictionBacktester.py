import sqliteClass
import pandas as pd
import colorama
from colorama import Fore, Style
import datetime as dt
import numpy as np

colorama.init(autoreset=True)

class LotteryBacktester:
    """
    Backtest lottery predictions using historical data and forecast future results.
    Uses a rolling window approach to validate prediction accuracy.
    """
    
    def __init__(self, dbFileName: str, datasetTable: str, raffle: str = "Bonoloto"):
        self.dbFileName = dbFileName
        self.datasetTable = datasetTable
        self.raffle = raffle
        
        # Initialize sqlite database connection
        self.sqlite = sqliteClass.db(
            dbFileName=self.dbFileName,
            datasetTable=self.datasetTable,
            predictionsTable=""  # Not needed
        )
        
        print(f"{Fore.CYAN}{'='*80}")
        print(f"{Fore.CYAN}Lottery Prediction Backtester - {self.raffle}")
        print(f"{Fore.CYAN}{'='*80}\n")
    
    def get_prediction_scores(self, cutoff_date: str):
        """
        Generate prediction scores for all numbers (1-49) based ONLY on data available
        BEFORE the cutoff_date.
        """
        # 1. Holistic Frequency (All-time before cutoff)
        query_freq = f"""
            SELECT 
                NUMBER,
                COUNT(*) as TOTAL_FREQ
            FROM {self.datasetTable}
            WHERE RAFFLE = '{self.raffle}'
            AND NUMBER_TYPE IN ('N1','N2','N3','N4','N5','N6')
            AND RESULT_DATE < '{cutoff_date}'
            GROUP BY NUMBER
        """
        df_freq = self.sqlite.executeQuery(query_freq)
        
        # 2. Recent Frequency (Last 50 draws before cutoff)
        query_recent = f"""
            WITH RecentDraws AS (
                SELECT DISTINCT RESULT_DATE
                FROM {self.datasetTable}
                WHERE RAFFLE = '{self.raffle}'
                AND RESULT_DATE < '{cutoff_date}'
                ORDER BY RESULT_DATE DESC
                LIMIT 50
            )
            SELECT 
                NUMBER,
                COUNT(*) as RECENT_FREQ
            FROM {self.datasetTable}
            WHERE RAFFLE = '{self.raffle}'
            AND NUMBER_TYPE IN ('N1', 'N2', 'N3', 'N4', 'N5', 'N6')
            AND RESULT_DATE IN (SELECT RESULT_DATE FROM RecentDraws)
            GROUP BY NUMBER
        """
        df_recent = self.sqlite.executeQuery(query_recent)
        
        # 3. Overdue Status (Days since last appearance before cutoff)
        query_overdue = f"""
            WITH LastAppearance AS (
                SELECT 
                    NUMBER,
                    MAX(RESULT_DATE) as LAST_DRAWN
                FROM {self.datasetTable}
                WHERE RAFFLE = '{self.raffle}'
                AND NUMBER_TYPE IN ('N1', 'N2', 'N3', 'N4', 'N5', 'N6')
                AND RESULT_DATE < '{cutoff_date}'
                GROUP BY NUMBER
            ),
            CutoffDate AS (
                SELECT '{cutoff_date}' as MAX_DATE
            )
            SELECT 
                la.NUMBER,
                CAST(JULIANDAY(cd.MAX_DATE) - JULIANDAY(la.LAST_DRAWN) AS INTEGER) as DAYS_OVERDUE
            FROM LastAppearance la
            CROSS JOIN CutoffDate cd
        """
        df_overdue = self.sqlite.executeQuery(query_overdue)
        
        # Merge all data
        # Create base dataframe with all numbers 1-49
        all_nums = pd.DataFrame({'NUMBER': range(1, 50)})
        
        df = all_nums.merge(df_freq, on='NUMBER', how='left')
        df = df.merge(df_recent, on='NUMBER', how='left')
        df = df.merge(df_overdue, on='NUMBER', how='left')
        
        # Fill NaN with 0
        df = df.fillna(0)
        
        # Calculate Scores
        max_freq = df['TOTAL_FREQ'].max() if df['TOTAL_FREQ'].max() > 0 else 1
        max_recent = df['RECENT_FREQ'].max() if df['RECENT_FREQ'].max() > 0 else 1
        
        # Weights
        W_FREQ = 0.40
        W_RECENT = 0.40
        W_OVERDUE = 0.20
        
        # Normalize scores (0-100 scale)
        df['FREQ_SCORE'] = (df['TOTAL_FREQ'] / max_freq) * 100
        df['RECENT_SCORE'] = (df['RECENT_FREQ'] / max_recent) * 100
        
        # Overdue scoring: Prefer 30-150 days (sweet spot)
        def overdue_score(days):
            if days < 30: return 0
            elif days <= 150: return ((days - 30) / 120) * 100
            else: return max(0, 100 - ((days - 150) / 2))
            
        df['OVERDUE_SCORE'] = df['DAYS_OVERDUE'].apply(overdue_score)
        
        # Total Composite Score
        df['TOTAL_SCORE'] = (
            (df['FREQ_SCORE'] * W_FREQ) + 
            (df['RECENT_SCORE'] * W_RECENT) + 
            (df['OVERDUE_SCORE'] * W_OVERDUE)
        )
        
        return df.sort_values('TOTAL_SCORE', ascending=False)

    def run_backtest(self, num_draws: int = 100):
        """
        Run backtesting on the last N draws.
        """
        print(f"{Fore.YELLOW}Starting Backtest on last {num_draws} draws...")
        
        # Get list of recent draw dates
        query_dates = f"""
            SELECT DISTINCT RESULT_DATE
            FROM {self.datasetTable}
            WHERE RAFFLE = '{self.raffle}'
            ORDER BY RESULT_DATE DESC
            LIMIT {num_draws}
        """
        dates = self.sqlite.executeQuery(query_dates)['RESULT_DATE'].tolist()
        dates.reverse() # Process chronologically
        
        results = []
        
        for i, test_date in enumerate(dates):
            # Get actual numbers for this date
            query_actual = f"""
                SELECT NUMBER
                FROM {self.datasetTable}
                WHERE RAFFLE = '{self.raffle}'
                AND RESULT_DATE = '{test_date}'
                AND NUMBER_TYPE IN ('N1','N2','N3','N4','N5','N6')
            """
            actual_nums = set(self.sqlite.executeQuery(query_actual)['NUMBER'].tolist())
            
            # Predict using ONLY data before this date
            predictions = self.get_prediction_scores(test_date)
            top_10 = set(predictions.head(10)['NUMBER'].tolist())
            
            # Check hits
            hits = len(actual_nums.intersection(top_10))
            results.append(hits)
            
            if (i+1) % 10 == 0:
                print(f"Processed {i+1}/{num_draws} draws... Current Avg Hits: {np.mean(results):.2f}")
        
        # Final Statistics
        avg_hits = np.mean(results)
        hit_counts = {k: results.count(k) for k in range(7)}
        
        print(f"\n{Fore.GREEN}{'='*80}")
        print(f"{Fore.GREEN}BACKTEST RESULTS (Top 10 Predictions)")
        print(f"{Fore.GREEN}{'='*80}")
        print(f"Total Draws Tested: {num_draws}")
        print(f"Average Hits per Draw: {avg_hits:.2f}")
        print(f"Random Expectation: ~1.22 hits")
        
        print(f"\n{Fore.CYAN}Hit Distribution:")
        for hits, count in sorted(hit_counts.items()):
            pct = (count / num_draws) * 100
            bar = '#' * int(pct/2)
            print(f"  {hits} Hits: {count:3d} ({pct:5.1f}%) {bar}")
            
        return avg_hits

    def predict_next_draw(self):
        """
        Predict numbers for the next upcoming draw using ALL available data.
        """
        # Use tomorrow's date as cutoff to include everything up to today
        tomorrow = (dt.date.today() + dt.timedelta(days=1)).strftime('%Y-%m-%d')
        
        predictions = self.get_prediction_scores(tomorrow)
        
        print(f"\n{Fore.MAGENTA}{'='*80}")
        print(f"{Fore.MAGENTA}FORECAST FOR NEXT DRAW")
        print(f"{Fore.MAGENTA}{'='*80}")
        
        print(f"{Fore.YELLOW}Top 10 Most Probable Numbers:")
        print(predictions[['NUMBER', 'TOTAL_SCORE', 'TOTAL_FREQ', 'RECENT_FREQ', 'DAYS_OVERDUE']].head(10).to_string(index=False))
        
        top_10 = predictions.head(10)['NUMBER'].tolist()
        print(f"\n{Fore.GREEN}Recommended Pool: {sorted(top_10)}")
        
        return top_10

if __name__ == "__main__":
    backtester = LotteryBacktester(
        dbFileName="predictions.sqlite",
        datasetTable="raffleDataset",
        raffle="Bonoloto"
    )
    
    # Run backtest
    backtester.run_backtest(num_draws=50)
    
    # Forecast next draw
    backtester.predict_next_draw()
