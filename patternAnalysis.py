import sqliteClass
import pandas as pd
import colorama
from colorama import Fore, Style
import datetime as dt

colorama.init(autoreset=True)

class LotteryPatternAnalyzer:
    """
    Analyze lottery patterns using statistical methods.
    This class provides frequency analysis, pattern detection, and statistical insights
    without relying on machine learning predictions.
    """
    
    def __init__(self, dbFileName: str, datasetTable: str, raffle: str = "Bonoloto"):
        self.dbFileName = dbFileName
        self.datasetTable = datasetTable
        self.raffle = raffle
        
        # Initialize sqlite database connection
        self.sqlite = sqliteClass.db(
            dbFileName=self.dbFileName,
            datasetTable=self.datasetTable,
            predictionsTable=""  # Not needed for analysis
        )
        
        print(f"{Fore.CYAN}{'='*80}")
        print(f"{Fore.CYAN}Lottery Pattern Analyzer - {self.raffle}")
        print(f"{Fore.CYAN}{'='*80}\n")
    
    def analyze_number_frequency(self, number_type: str = None, top_n: int = 10):
        """
        Analyze the frequency of each number drawn.
        Shows hot numbers (most frequent) and cold numbers (least frequent).
        """
        print(f"\n{Fore.YELLOW}{'='*80}")
        print(f"{Fore.YELLOW}NUMBER FREQUENCY ANALYSIS")
        print(f"{Fore.YELLOW}{'='*80}")
        
        if number_type:
            where_clause = f"AND NUMBER_TYPE = '{number_type}'"
            print(f"{Fore.WHITE}Analyzing: {number_type}")
        else:
            where_clause = ""
            print(f"{Fore.WHITE}Analyzing: All number types")
        
        query = f"""
            SELECT 
                NUMBER_TYPE,
                NUMBER,
                COUNT(*) as FREQUENCY,
                ROUND(COUNT(*) * 100.0 / (SELECT COUNT(DISTINCT RESULT_DATE) 
                    FROM {self.datasetTable} 
                    WHERE RAFFLE = '{self.raffle}'), 2) as PERCENTAGE
            FROM {self.datasetTable}
            WHERE RAFFLE = '{self.raffle}'
            {where_clause}
            GROUP BY NUMBER_TYPE, NUMBER
            ORDER BY NUMBER_TYPE, FREQUENCY DESC
        """
        
        df = self.sqlite.executeQuery(query)
        
        # Show hot numbers (most frequent)
        print(f"\n{Fore.GREEN}[HOT] NUMBERS (Top {top_n} most frequent):")
        print(df.head(top_n).to_string(index=False))
        
        # Show cold numbers (least frequent)
        print(f"\n{Fore.BLUE}[COLD] NUMBERS (Bottom {top_n} least frequent):")
        print(df.tail(top_n).to_string(index=False))
        
        return df
    
    def analyze_overdue_numbers(self, number_type: str = None, holistic: bool = False):
        """
        Find numbers that haven't appeared in the longest time.
        If holistic=True, checks when the number last appeared in ANY position.
        """
        print(f"\n{Fore.YELLOW}{'='*80}")
        print(f"{Fore.YELLOW}OVERDUE NUMBERS ANALYSIS")
        print(f"{Fore.YELLOW}{'='*80}")
        
        if holistic:
            print(f"{Fore.WHITE}Analyzing: Holistic (Any Position)")
            group_by = "NUMBER"
            select_cols = "NUMBER,"
            where_clause = "AND NUMBER_TYPE IN ('N1', 'N2', 'N3', 'N4', 'N5', 'N6')"
        elif number_type:
            where_clause = f"AND NUMBER_TYPE = '{number_type}'"
            print(f"{Fore.WHITE}Analyzing: {number_type}")
            group_by = "NUMBER_TYPE, NUMBER"
            select_cols = "NUMBER_TYPE, NUMBER,"
        else:
            where_clause = ""
            print(f"{Fore.WHITE}Analyzing: All number types (separately)")
            group_by = "NUMBER_TYPE, NUMBER"
            select_cols = "NUMBER_TYPE, NUMBER,"
        
        query = f"""
            WITH LastAppearance AS (
                SELECT 
                    {select_cols}
                    MAX(RESULT_DATE) as LAST_DRAWN
                FROM {self.datasetTable}
                WHERE RAFFLE = '{self.raffle}'
                {where_clause}
                GROUP BY {group_by}
            ),
            MaxDate AS (
                SELECT MAX(RESULT_DATE) as MAX_DATE
                FROM {self.datasetTable}
                WHERE RAFFLE = '{self.raffle}'
            )
            SELECT 
                {select_cols}
                la.LAST_DRAWN,
                CAST(JULIANDAY(md.MAX_DATE) - JULIANDAY(la.LAST_DRAWN) AS INTEGER) as DAYS_SINCE_LAST_DRAW
            FROM LastAppearance la
            CROSS JOIN MaxDate md
            ORDER BY DAYS_SINCE_LAST_DRAW DESC
            LIMIT 20
        """
        
        df = self.sqlite.executeQuery(query)
        print(f"\n{Fore.MAGENTA}[OVERDUE] MOST OVERDUE NUMBERS:")
        print(df.to_string(index=False))
        
        return df
    
    def analyze_number_pairs(self, number_type_1: str = "N1", number_type_2: str = "N2", top_n: int = 15):
        """
        Find the most common pairs of numbers drawn together.
        """
        print(f"\n{Fore.YELLOW}{'='*80}")
        print(f"{Fore.YELLOW}NUMBER PAIRS ANALYSIS")
        print(f"{Fore.YELLOW}{'='*80}")
        print(f"{Fore.WHITE}Analyzing pairs: {number_type_1} and {number_type_2}")
        
        query = f"""
            SELECT 
                a.NUMBER as NUMBER_1,
                b.NUMBER as NUMBER_2,
                COUNT(*) as FREQUENCY
            FROM {self.datasetTable} a
            INNER JOIN {self.datasetTable} b
                ON a.RAFFLE = b.RAFFLE
                AND a.RESULT_DATE = b.RESULT_DATE
                AND a.NUMBER_TYPE = '{number_type_1}'
                AND b.NUMBER_TYPE = '{number_type_2}'
            WHERE a.RAFFLE = '{self.raffle}'
            GROUP BY a.NUMBER, b.NUMBER
            ORDER BY FREQUENCY DESC
            LIMIT {top_n}
        """
        
        df = self.sqlite.executeQuery(query)
        print(f"\n{Fore.GREEN}[PAIRS] MOST COMMON PAIRS:")
        print(df.to_string(index=False))
        
        return df
    
    def analyze_consecutive_numbers(self):
        """
        Analyze how often consecutive numbers appear in the same draw.
        """
        print(f"\n{Fore.YELLOW}{'='*80}")
        print(f"{Fore.YELLOW}CONSECUTIVE NUMBERS ANALYSIS")
        print(f"{Fore.YELLOW}{'='*80}")
        
        query = f"""
            WITH DrawNumbers AS (
                SELECT 
                    RESULT_DATE,
                    NUMBER
                FROM {self.datasetTable}
                WHERE RAFFLE = '{self.raffle}'
                AND NUMBER_TYPE IN ('N1', 'N2', 'N3', 'N4', 'N5', 'N6')
                ORDER BY RESULT_DATE, NUMBER
            )
            SELECT 
                a.RESULT_DATE,
                a.NUMBER as NUMBER_1,
                b.NUMBER as NUMBER_2
            FROM DrawNumbers a
            INNER JOIN DrawNumbers b
                ON a.RESULT_DATE = b.RESULT_DATE
                AND b.NUMBER = a.NUMBER + 1
            ORDER BY a.RESULT_DATE DESC
            LIMIT 30
        """
        
        df = self.sqlite.executeQuery(query)
        
        if len(df) > 0:
            consecutive_count = len(df)
            total_draws_query = f"SELECT COUNT(DISTINCT RESULT_DATE) as TOTAL FROM {self.datasetTable} WHERE RAFFLE = '{self.raffle}'"
            total_draws = self.sqlite.executeQuery(total_draws_query)['TOTAL'][0]
            percentage = round((consecutive_count / total_draws) * 100, 2)
            
            print(f"\n{Fore.GREEN}[STATS] CONSECUTIVE NUMBERS STATISTICS:")
            print(f"Total draws with consecutive numbers: {consecutive_count}")
            print(f"Percentage of draws: {percentage}%")
            print(f"\n{Fore.CYAN}Recent draws with consecutive numbers:")
            print(df.head(10).to_string(index=False))
        else:
            print(f"\n{Fore.YELLOW}No consecutive numbers found in recent draws.")
        
        return df
    
    def analyze_number_distribution(self, number_type: str = "N1"):
        """
        Analyze the distribution of numbers (low vs high, odd vs even).
        """
        print(f"\n{Fore.YELLOW}{'='*80}")
        print(f"{Fore.YELLOW}NUMBER DISTRIBUTION ANALYSIS - {number_type}")
        print(f"{Fore.YELLOW}{'='*80}")
        
        # Odd vs Even
        query_odd_even = f"""
            SELECT 
                CASE WHEN NUMBER % 2 = 0 THEN 'EVEN' ELSE 'ODD' END as TYPE,
                COUNT(*) as COUNT,
                ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM {self.datasetTable} 
                    WHERE RAFFLE = '{self.raffle}' AND NUMBER_TYPE = '{number_type}'), 2) as PERCENTAGE
            FROM {self.datasetTable}
            WHERE RAFFLE = '{self.raffle}'
            AND NUMBER_TYPE = '{number_type}'
            GROUP BY TYPE
        """
        
        df_odd_even = self.sqlite.executeQuery(query_odd_even)
        print(f"\n{Fore.GREEN}[ODD/EVEN]:")
        print(df_odd_even.to_string(index=False))
        
        # Low vs High (assuming numbers 1-49, split at 25)
        query_low_high = f"""
            SELECT 
                CASE WHEN NUMBER <= 25 THEN 'LOW (1-25)' ELSE 'HIGH (26-49)' END as RANGE,
                COUNT(*) as COUNT,
                ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM {self.datasetTable} 
                    WHERE RAFFLE = '{self.raffle}' AND NUMBER_TYPE = '{number_type}'), 2) as PERCENTAGE
            FROM {self.datasetTable}
            WHERE RAFFLE = '{self.raffle}'
            AND NUMBER_TYPE = '{number_type}'
            GROUP BY RANGE
        """
        
        df_low_high = self.sqlite.executeQuery(query_low_high)
        print(f"\n{Fore.GREEN}[LOW/HIGH]:")
        print(df_low_high.to_string(index=False))
        
        return df_odd_even, df_low_high
    
    def analyze_holistic_frequency(self, top_n: int = 20):
        """
        Analyze number frequency across ALL positions (N1-N6 combined).
        Since numbers are sorted, this gives true frequency regardless of position.
        """
        print(f"\n{Fore.YELLOW}{'='*80}")
        print(f"{Fore.YELLOW}HOLISTIC NUMBER FREQUENCY ANALYSIS")
        print(f"{Fore.YELLOW}{'='*80}")
        print(f"{Fore.WHITE}Analyzing all numbers (1-49) across all positions\n")
        
        query = f"""
            SELECT 
                NUMBER,
                COUNT(*) as FREQUENCY,
                ROUND(COUNT(*) * 100.0 / (SELECT COUNT(DISTINCT RESULT_DATE) * 6
                    FROM {self.datasetTable} 
                    WHERE RAFFLE = '{self.raffle}' 
                    AND NUMBER_TYPE IN ('N1','N2','N3','N4','N5','N6')), 2) as PERCENTAGE,
                ROUND(COUNT(*) * 100.0 / 6, 2) as AVG_PER_DRAW
            FROM {self.datasetTable}
            WHERE RAFFLE = '{self.raffle}'
            AND NUMBER_TYPE IN ('N1', 'N2', 'N3', 'N4', 'N5', 'N6')
            GROUP BY NUMBER
            ORDER BY FREQUENCY DESC
        """
        
        df = self.sqlite.executeQuery(query)
        
        print(f"{Fore.GREEN}[HOT] Top {top_n} Most Frequent Numbers:")
        print(df.head(top_n).to_string(index=False))
        
        print(f"\n{Fore.BLUE}[COLD] Bottom {top_n} Least Frequent Numbers:")
        print(df.tail(top_n).to_string(index=False))
        
        return df
    
    def analyze_normality_test(self):
        """
        Test if the lottery is truly random by comparing actual vs expected frequencies.
        In a truly random lottery, each number (1-49) should appear with equal probability.
        """
        print(f"\n{Fore.YELLOW}{'='*80}")
        print(f"{Fore.YELLOW}NORMALITY / RANDOMNESS TEST")
        print(f"{Fore.YELLOW}{'='*80}")
        print(f"{Fore.WHITE}Testing if lottery distribution is statistically random\n")
        
        # Get total draws
        query_total = f"""
            SELECT COUNT(DISTINCT RESULT_DATE) as TOTAL_DRAWS
            FROM {self.datasetTable}
            WHERE RAFFLE = '{self.raffle}'
        """
        total_draws = self.sqlite.executeQuery(query_total)['TOTAL_DRAWS'][0]
        
        # Expected frequency: each number should appear (total_draws * 6 / 49) times
        expected_freq = (total_draws * 6) / 49.0
        
        # Get actual frequencies for ALL numbers (1-49)
        query = f"""
            WITH AllNumbers AS (
                SELECT value as NUMBER
                FROM (
                    SELECT 1 as value UNION SELECT 2 UNION SELECT 3 UNION SELECT 4 UNION SELECT 5 UNION
                    SELECT 6 UNION SELECT 7 UNION SELECT 8 UNION SELECT 9 UNION SELECT 10 UNION
                    SELECT 11 UNION SELECT 12 UNION SELECT 13 UNION SELECT 14 UNION SELECT 15 UNION
                    SELECT 16 UNION SELECT 17 UNION SELECT 18 UNION SELECT 19 UNION SELECT 20 UNION
                    SELECT 21 UNION SELECT 22 UNION SELECT 23 UNION SELECT 24 UNION SELECT 25 UNION
                    SELECT 26 UNION SELECT 27 UNION SELECT 28 UNION SELECT 29 UNION SELECT 30 UNION
                    SELECT 31 UNION SELECT 32 UNION SELECT 33 UNION SELECT 34 UNION SELECT 35 UNION
                    SELECT 36 UNION SELECT 37 UNION SELECT 38 UNION SELECT 39 UNION SELECT 40 UNION
                    SELECT 41 UNION SELECT 42 UNION SELECT 43 UNION SELECT 44 UNION SELECT 45 UNION
                    SELECT 46 UNION SELECT 47 UNION SELECT 48 UNION SELECT 49
                )
            )
            SELECT 
                an.NUMBER,
                COALESCE(COUNT(d.NUMBER), 0) as ACTUAL_FREQ,
                ROUND({expected_freq}, 2) as EXPECTED_FREQ,
                ROUND(COALESCE(COUNT(d.NUMBER), 0) - {expected_freq}, 2) as DEVIATION,
                ROUND((COALESCE(COUNT(d.NUMBER), 0) - {expected_freq}) * 100.0 / {expected_freq}, 2) as PCT_DEVIATION
            FROM AllNumbers an
            LEFT JOIN {self.datasetTable} d
                ON an.NUMBER = d.NUMBER 
                AND d.RAFFLE = '{self.raffle}'
                AND d.NUMBER_TYPE IN ('N1', 'N2', 'N3', 'N4', 'N5', 'N6')
            GROUP BY an.NUMBER
            ORDER BY ACTUAL_FREQ ASC
        """
        
        df = self.sqlite.executeQuery(query)
        
        # Calculate chi-square statistic
        chi_square = sum((row['ACTUAL_FREQ'] - row['EXPECTED_FREQ'])**2 / row['EXPECTED_FREQ'] for _, row in df.iterrows())
        
        # Degrees of freedom = 49 - 1 = 48
        # Critical value for chi-square at 95% confidence with 48 df ≈ 65.17
        # Critical value at 99% confidence ≈ 73.68
        critical_95 = 65.17
        critical_99 = 73.68
        
        print(f"{Fore.CYAN}[TEST STATISTICS]")
        print(f"  Total Draws: {total_draws}")
        print(f"  Total Number Appearances: {total_draws * 6}")
        print(f"  Expected Frequency per Number: {expected_freq:.2f}")
        print(f"  Chi-Square Statistic: {chi_square:.2f}")
        print(f"  Critical Value (95% confidence): {critical_95}")
        print(f"  Critical Value (99% confidence): {critical_99}")
        
        if chi_square < critical_95:
            print(f"\n{Fore.GREEN}[RESULT]: Lottery appears RANDOM (chi-square < critical value)")
            print(f"{Fore.GREEN}  The distribution is statistically normal at 95% confidence.")
        elif chi_square < critical_99:
            print(f"\n{Fore.YELLOW}[RESULT]: Lottery is likely RANDOM (chi-square < 99% critical)")
            print(f"{Fore.YELLOW}  Some deviation exists but still within acceptable range.")
        else:
            print(f"\n{Fore.RED}[RESULT]: Lottery shows SIGNIFICANT DEVIATION from randomness")
            print(f"{Fore.RED}  This could indicate bias or insufficient sample size.")
        
        # Show most under-represented numbers
        print(f"\n{Fore.BLUE}[MOST UNDER-REPRESENTED] Numbers that should appear more often:")
        under_rep = df.nsmallest(10, 'ACTUAL_FREQ')
        print(under_rep[['NUMBER', 'ACTUAL_FREQ', 'EXPECTED_FREQ', 'DEVIATION', 'PCT_DEVIATION']].to_string(index=False))
        
        # Show most over-represented numbers
        print(f"\n{Fore.RED}[MOST OVER-REPRESENTED] Numbers appearing more than expected:")
        over_rep = df.nlargest(10, 'ACTUAL_FREQ')
        print(over_rep[['NUMBER', 'ACTUAL_FREQ', 'EXPECTED_FREQ', 'DEVIATION', 'PCT_DEVIATION']].to_string(index=False))
        
        # Show numbers closest to expected (most "normal")
        print(f"\n{Fore.GREEN}[MOST NORMAL] Numbers closest to expected frequency:")
        df['ABS_DEVIATION'] = abs(df['DEVIATION'])
        normal = df.nsmallest(10, 'ABS_DEVIATION')
        print(normal[['NUMBER', 'ACTUAL_FREQ', 'EXPECTED_FREQ', 'DEVIATION', 'PCT_DEVIATION']].to_string(index=False))
        
        return df, chi_square
    
    def analyze_number_gaps(self):
        """
        Analyze gaps between consecutive drawn numbers in each draw.
        """
        print(f"\n{Fore.YELLOW}{'='*80}")
        print(f"{Fore.YELLOW}NUMBER GAP ANALYSIS")
        print(f"{Fore.YELLOW}{'='*80}")
        print(f"{Fore.WHITE}Analyzing gaps between consecutive numbers in each draw\n")
        
        # Get gap statistics
        query = f"""
            WITH DrawNumbers AS (
                SELECT 
                    RESULT_DATE,
                    NUMBER,
                    ROW_NUMBER() OVER (PARTITION BY RESULT_DATE ORDER BY NUMBER) as pos
                FROM {self.datasetTable}
                WHERE RAFFLE = '{self.raffle}'
                AND NUMBER_TYPE IN ('N1', 'N2', 'N3', 'N4', 'N5', 'N6')
            ),
            Gaps AS (
                SELECT 
                    a.RESULT_DATE,
                    a.NUMBER as NUM1,
                    b.NUMBER as NUM2,
                    b.NUMBER - a.NUMBER as GAP
                FROM DrawNumbers a
                JOIN DrawNumbers b
                ON a.RESULT_DATE = b.RESULT_DATE
                AND b.pos = a.pos + 1
            )
            SELECT 
                GAP,
                COUNT(*) as FREQUENCY,
                ROUND(AVG(GAP), 2) as AVG_GAP
            FROM Gaps
            GROUP BY GAP
            ORDER BY FREQUENCY DESC
            LIMIT 15
        """
        
        df = self.sqlite.executeQuery(query)
        print(f"{Fore.GREEN}[GAPS] Most Common Gaps Between Consecutive Numbers:")
        print(df.to_string(index=False))
        
        # Average gap across all draws
        query_avg = f"""
            WITH DrawNumbers AS (
                SELECT 
                    RESULT_DATE,
                    NUMBER,
                    ROW_NUMBER() OVER (PARTITION BY RESULT_DATE ORDER BY NUMBER) as pos
                FROM {self.datasetTable}
                WHERE RAFFLE = '{self.raffle}'
                AND NUMBER_TYPE IN ('N1', 'N2', 'N3', 'N4', 'N5', 'N6')
            ),
            Gaps AS (
                SELECT 
                    b.NUMBER - a.NUMBER as GAP
                FROM DrawNumbers a
                JOIN DrawNumbers b
                ON a.RESULT_DATE = b.RESULT_DATE
                AND b.pos = a.pos + 1
            )
            SELECT 
                ROUND(AVG(GAP), 2) as OVERALL_AVG_GAP,
                MIN(GAP) as MIN_GAP,
                MAX(GAP) as MAX_GAP
            FROM Gaps
        """
        
        df_avg = self.sqlite.executeQuery(query_avg)
        print(f"\n{Fore.CYAN}[STATS] Gap Statistics:")
        print(f"  Average Gap: {df_avg['OVERALL_AVG_GAP'][0]}")
        print(f"  Min Gap: {df_avg['MIN_GAP'][0]}")
        print(f"  Max Gap: {df_avg['MAX_GAP'][0]}")
        
        return df
    
    def analyze_sum_patterns(self):
        """
        Analyze the sum of all 6 numbers in each draw.
        """
        print(f"\n{Fore.YELLOW}{'='*80}")
        print(f"{Fore.YELLOW}SUM PATTERN ANALYSIS")
        print(f"{Fore.YELLOW}{'='*80}")
        print(f"{Fore.WHITE}Analyzing sum of all 6 numbers per draw\n")
        
        query = f"""
            WITH DrawSums AS (
                SELECT 
                    RESULT_DATE,
                    SUM(NUMBER) as TOTAL_SUM
                FROM {self.datasetTable}
                WHERE RAFFLE = '{self.raffle}'
                AND NUMBER_TYPE IN ('N1', 'N2', 'N3', 'N4', 'N5', 'N6')
                GROUP BY RESULT_DATE
            )
            SELECT 
                MIN(TOTAL_SUM) as MIN_SUM,
                MAX(TOTAL_SUM) as MAX_SUM,
                ROUND(AVG(TOTAL_SUM), 2) as AVG_SUM,
                COUNT(*) as TOTAL_DRAWS
            FROM DrawSums
        """
        
        df_stats = self.sqlite.executeQuery(query)
        print(f"{Fore.GREEN}[STATS] Sum Statistics:")
        print(f"  Min Sum: {df_stats['MIN_SUM'][0]}")
        print(f"  Max Sum: {df_stats['MAX_SUM'][0]}")
        print(f"  Average Sum: {df_stats['AVG_SUM'][0]}")
        
        # Most common sum ranges
        query_ranges = f"""
            WITH DrawSums AS (
                SELECT 
                    RESULT_DATE,
                    SUM(NUMBER) as TOTAL_SUM
                FROM {self.datasetTable}
                WHERE RAFFLE = '{self.raffle}'
                AND NUMBER_TYPE IN ('N1', 'N2', 'N3', 'N4', 'N5', 'N6')
                GROUP BY RESULT_DATE
            )
            SELECT 
                CASE 
                    WHEN TOTAL_SUM < 100 THEN '< 100'
                    WHEN TOTAL_SUM BETWEEN 100 AND 119 THEN '100-119'
                    WHEN TOTAL_SUM BETWEEN 120 AND 139 THEN '120-139'
                    WHEN TOTAL_SUM BETWEEN 140 AND 159 THEN '140-159'
                    WHEN TOTAL_SUM BETWEEN 160 AND 179 THEN '160-179'
                    WHEN TOTAL_SUM BETWEEN 180 AND 199 THEN '180-199'
                    ELSE '>= 200'
                END as SUM_RANGE,
                COUNT(*) as FREQUENCY,
                ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM DrawSums), 2) as PERCENTAGE
            FROM DrawSums
            GROUP BY SUM_RANGE
            ORDER BY FREQUENCY DESC
        """
        
        df_ranges = self.sqlite.executeQuery(query_ranges)
        print(f"\n{Fore.GREEN}[RANGES] Most Common Sum Ranges:")
        print(df_ranges.to_string(index=False))
        
        return df_stats, df_ranges
    
    def analyze_number_ranges(self):
        """
        Analyze how numbers are distributed across ranges (1-10, 11-20, etc.).
        """
        print(f"\n{Fore.YELLOW}{'='*80}")
        print(f"{Fore.YELLOW}NUMBER RANGE DISTRIBUTION")
        print(f"{Fore.YELLOW}{'='*80}")
        print(f"{Fore.WHITE}Analyzing distribution across number ranges\n")
        
        query = f"""
            WITH RangedNumbers AS (
                SELECT 
                    RESULT_DATE,
                    CASE 
                        WHEN NUMBER BETWEEN 1 AND 10 THEN '01-10'
                        WHEN NUMBER BETWEEN 11 AND 20 THEN '11-20'
                        WHEN NUMBER BETWEEN 21 AND 30 THEN '21-30'
                        WHEN NUMBER BETWEEN 31 AND 40 THEN '31-40'
                        WHEN NUMBER BETWEEN 41 AND 49 THEN '41-49'
                    END as NUM_RANGE
                FROM {self.datasetTable}
                WHERE RAFFLE = '{self.raffle}'
                AND NUMBER_TYPE IN ('N1', 'N2', 'N3', 'N4', 'N5', 'N6')
            )
            SELECT 
                NUM_RANGE,
                COUNT(*) as TOTAL_APPEARANCES,
                ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM RangedNumbers), 2) as PERCENTAGE,
                ROUND(COUNT(*) * 1.0 / (SELECT COUNT(DISTINCT RESULT_DATE) FROM {self.datasetTable} WHERE RAFFLE = '{self.raffle}'), 2) as AVG_PER_DRAW
            FROM RangedNumbers
            GROUP BY NUM_RANGE
            ORDER BY NUM_RANGE
        """
        
        df = self.sqlite.executeQuery(query)
        print(f"{Fore.GREEN}[DISTRIBUTION] Numbers Per Range:")
        print(df.to_string(index=False))
        
        return df
    
    def analyze_recent_trends(self, last_n_draws: int = 50):
        """
        Analyze trends in the most recent N draws.
        """
        print(f"\n{Fore.YELLOW}{'='*80}")
        print(f"{Fore.YELLOW}RECENT TREND ANALYSIS (Last {last_n_draws} draws)")
        print(f"{Fore.YELLOW}{'='*80}")
        
        # Get recent hot numbers
        query = f"""
            WITH RecentDraws AS (
                SELECT DISTINCT RESULT_DATE
                FROM {self.datasetTable}
                WHERE RAFFLE = '{self.raffle}'
                ORDER BY RESULT_DATE DESC
                LIMIT {last_n_draws}
            )
            SELECT 
                NUMBER,
                COUNT(*) as FREQUENCY,
                ROUND(COUNT(*) * 100.0 / {last_n_draws}, 2) as APPEARANCE_RATE
            FROM {self.datasetTable}
            WHERE RAFFLE = '{self.raffle}'
            AND NUMBER_TYPE IN ('N1', 'N2', 'N3', 'N4', 'N5', 'N6')
            AND RESULT_DATE IN (SELECT RESULT_DATE FROM RecentDraws)
            GROUP BY NUMBER
            ORDER BY FREQUENCY DESC
            LIMIT 15
        """
        
        df_hot = self.sqlite.executeQuery(query)
        print(f"\n{Fore.GREEN}[HOT] Recent Hot Numbers (Last {last_n_draws} draws):")
        print(df_hot.to_string(index=False))
        
        # Get recent cold numbers
        query_cold = f"""
            WITH RecentDraws AS (
                SELECT DISTINCT RESULT_DATE
                FROM {self.datasetTable}
                WHERE RAFFLE = '{self.raffle}'
                ORDER BY RESULT_DATE DESC
                LIMIT {last_n_draws}
            ),
            AllNumbers AS (
                SELECT DISTINCT NUMBER
                FROM {self.datasetTable}
                WHERE RAFFLE = '{self.raffle}'
                AND NUMBER_TYPE IN ('N1', 'N2', 'N3', 'N4', 'N5', 'N6')
            )
            SELECT 
                an.NUMBER,
                COALESCE(COUNT(rd.NUMBER), 0) as FREQUENCY,
                ROUND(COALESCE(COUNT(rd.NUMBER), 0) * 100.0 / {last_n_draws}, 2) as APPEARANCE_RATE
            FROM AllNumbers an
            LEFT JOIN (
                SELECT NUMBER
                FROM {self.datasetTable}
                WHERE RAFFLE = '{self.raffle}'
                AND NUMBER_TYPE IN ('N1', 'N2', 'N3', 'N4', 'N5', 'N6')
                AND RESULT_DATE IN (SELECT RESULT_DATE FROM RecentDraws)
            ) rd ON an.NUMBER = rd.NUMBER
            GROUP BY an.NUMBER
            ORDER BY FREQUENCY ASC, an.NUMBER
            LIMIT 15
        """
        
        df_cold = self.sqlite.executeQuery(query_cold)
        print(f"\n{Fore.BLUE}[COLD] Recent Cold Numbers (Last {last_n_draws} draws):")
        print(df_cold.to_string(index=False))
        
        return df_hot, df_cold
    

    def run_full_analysis(self):
        """
        Run comprehensive deep analysis with holistic number analysis.
        """
        print(f"\n{Fore.CYAN}Running ENHANCED comprehensive pattern analysis...\n")
        
        # All number types in Bonoloto
        all_number_types = ["N1", "N2", "N3", "N4", "N5", "N6", "Complementario", "Reintegro"]
        
        # PART 1: DEEP HOLISTIC ANALYSES (New!)
        print(f"\n{Fore.MAGENTA}{'#'*80}")
        print(f"{Fore.MAGENTA}PART 1: DEEP HOLISTIC ANALYSES")
        print(f"{Fore.MAGENTA}{'#'*80}")
        
        self.analyze_holistic_frequency(top_n=20)
        self.analyze_normality_test()  # New normality test
        self.analyze_number_gaps()
        self.analyze_sum_patterns()
        self.analyze_number_ranges()
        self.analyze_recent_trends(last_n_draws=50)
        self.analyze_recent_trends(last_n_draws=100)
        
        # PART 2: POSITIONAL FREQUENCY (Understanding sorted nature)
        print(f"\n{Fore.MAGENTA}{'#'*80}")
        print(f"{Fore.MAGENTA}PART 2: POSITIONAL FREQUENCY (for reference only)")
        print(f"{Fore.MAGENTA}{'#'*80}")
        for num_type in ["N1", "N2", "N6"]:  # Just show first, second, and last
            self.analyze_number_frequency(number_type=num_type, top_n=5)
        
        # PART 3: OVERDUE ANALYSIS
        print(f"\n{Fore.MAGENTA}{'#'*80}")
        print(f"{Fore.MAGENTA}PART 3: OVERDUE NUMBERS")
        print(f"{Fore.MAGENTA}{'#'*80}")
        self.analyze_overdue_numbers(holistic=True)  # Holistic overdue analysis
        self.analyze_overdue_numbers()  # By type analysis (default)
        
        # PART 4: ODD/EVEN DISTRIBUTION
        print(f"\n{Fore.MAGENTA}{'#'*80}")
        print(f"{Fore.MAGENTA}PART 4: ODD/EVEN BALANCE")
        print(f"{Fore.MAGENTA}{'#'*80}")
        
        total_odd = 0
        total_even = 0
        for num_type in ["N1", "N2", "N3", "N4", "N5", "N6"]:
            query = f"""
                SELECT 
                    CASE WHEN NUMBER % 2 = 0 THEN 'EVEN' ELSE 'ODD' END as TYPE,
                    COUNT(*) as COUNT
                FROM {self.datasetTable}
                WHERE RAFFLE = '{self.raffle}' AND NUMBER_TYPE = '{num_type}'
                GROUP BY TYPE
            """
            df = self.sqlite.executeQuery(query)
            odd_count = df[df['TYPE'] == 'ODD']['COUNT'].values[0] if 'ODD' in df['TYPE'].values else 0
            even_count = df[df['TYPE'] == 'EVEN']['COUNT'].values[0] if 'EVEN' in df['TYPE'].values else 0
            total_odd += odd_count
            total_even += even_count
        
        total = total_odd + total_even
        odd_pct = round((total_odd / total) * 100, 1)
        even_pct = round((total_even / total) * 100, 1)
        print(f"{Fore.GREEN}  Overall: {odd_pct}% ODD / {even_pct}% EVEN")
        
        # PART 5: COMPREHENSIVE INSIGHTS & PREDICTIONS
        self.generate_insights_summary()
        
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.GREEN}[COMPLETE] Enhanced analysis complete!")
        print(f"{Fore.CYAN}{'='*80}\n")
    
    def generate_insights_summary(self):
        """
        Generate a comprehensive summary with actionable insights AND predicted numbers.
        """
        print(f"\n{Fore.MAGENTA}{'#'*80}")
        print(f"{Fore.MAGENTA}PART 5: COMPREHENSIVE INSIGHTS & PREDICTIONS")
        print(f"{Fore.MAGENTA}{'#'*80}\n")
        
        print(f"{Fore.CYAN}{'='*80}")
        print(f"{Fore.CYAN}ACTIONABLE INSIGHTS & PREDICTIONS")
        print(f"{Fore.CYAN}{'='*80}\n")
        
        # Get holistic frequency data
        query_freq = f"""
            SELECT 
                NUMBER,
                COUNT(*) as TOTAL_FREQ,
                ROUND(COUNT(*) * 100.0 / (SELECT COUNT(DISTINCT RESULT_DATE) * 6
                    FROM {self.datasetTable} 
                    WHERE RAFFLE = '{self.raffle}' 
                    AND NUMBER_TYPE IN ('N1','N2','N3','N4','N5','N6')), 2) as FREQ_PCT
            FROM {self.datasetTable}
            WHERE RAFFLE = '{self.raffle}'
            AND NUMBER_TYPE IN ('N1', 'N2', 'N3', 'N4', 'N5', 'N6')
            GROUP BY NUMBER
            ORDER BY TOTAL_FREQ DESC
        """
        df_freq = self.sqlite.executeQuery(query_freq)
        
        # Get recent hot numbers (last 50 draws)
        query_recent = f"""
            WITH RecentDraws AS (
                SELECT DISTINCT RESULT_DATE
                FROM {self.datasetTable}
                WHERE RAFFLE = '{self.raffle}'
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
            ORDER BY RECENT_FREQ DESC
        """
        df_recent = self.sqlite.executeQuery(query_recent)
        
        # Get overdue numbers
        query_overdue = f"""
            WITH LastAppearance AS (
                SELECT 
                    NUMBER,
                    MAX(RESULT_DATE) as LAST_DRAWN
                FROM {self.datasetTable}
                WHERE RAFFLE = '{self.raffle}'
                AND NUMBER_TYPE IN ('N1', 'N2', 'N3', 'N4', 'N5', 'N6')
                GROUP BY NUMBER
            ),
            MaxDate AS (
                SELECT MAX(RESULT_DATE) as MAX_DATE
                FROM {self.datasetTable}
                WHERE RAFFLE = '{self.raffle}'
            )
            SELECT 
                la.NUMBER,
                CAST(JULIANDAY(md.MAX_DATE) - JULIANDAY(la.LAST_DRAWN) AS INTEGER) as DAYS_OVERDUE
            FROM LastAppearance la
            CROSS JOIN MaxDate md
            ORDER BY DAYS_OVERDUE DESC
        """
        df_overdue = self.sqlite.executeQuery(query_overdue)
        
        # Create a scoring system for prediction
        # Merge all dataframes
        prediction_df = df_freq.copy()
        prediction_df = prediction_df.merge(df_recent, on='NUMBER', how='left')
        prediction_df = prediction_df.merge(df_overdue, on='NUMBER', how='left')
        prediction_df['RECENT_FREQ'] = prediction_df['RECENT_FREQ'].fillna(0)
        prediction_df['DAYS_OVERDUE'] = prediction_df['DAYS_OVERDUE'].fillna(0)
        
        # Calculate composite score
        # - 40% weight on overall frequency
        # - 40% weight on recent frequency  
        # - 20% weight on overdue factor (moderate overdue is good, too much is bad)
        max_freq = prediction_df['TOTAL_FREQ'].max()
        max_recent = prediction_df['RECENT_FREQ'].max()
        max_overdue = prediction_df['DAYS_OVERDUE'].max()
        
        prediction_df['FREQ_SCORE'] = (prediction_df['TOTAL_FREQ'] / max_freq) * 40
        prediction_df['RECENT_SCORE'] = (prediction_df['RECENT_FREQ'] / max_recent) * 40
        
        # Overdue scoring: prefer moderately overdue (30-150 days), penalize too recent or too old
        def overdue_score(days):
            if days < 30:
                return 0  # Too recent
            elif days <= 150:
                return ((days - 30) / 120) * 20  # Linear increase
            else:
                # Decrease score as overdue increases beyond 150 days
                return max(0, 20 - ((days - 150) / 50))
        
        prediction_df['OVERDUE_SCORE'] = prediction_df['DAYS_OVERDUE'].apply(overdue_score)
        prediction_df['TOTAL_SCORE'] = prediction_df['FREQ_SCORE'] + prediction_df['RECENT_SCORE'] + prediction_df['OVERDUE_SCORE']
        
        # Sort by total score
        prediction_df = prediction_df.sort_values('TOTAL_SCORE', ascending=False)
        
        # Get sum statistics
        query_sum_stats = f"""
            WITH DrawSums AS (
                SELECT SUM(NUMBER) as TOTAL_SUM
                FROM {self.datasetTable}
                WHERE RAFFLE = '{self.raffle}'
                AND NUMBER_TYPE IN ('N1', 'N2', 'N3', 'N4', 'N5', 'N6')
                GROUP BY RESULT_DATE
            )
            SELECT 
                ROUND(AVG(TOTAL_SUM), 0) as AVG_SUM,
                MIN(TOTAL_SUM) as MIN_SUM,
                MAX(TOTAL_SUM) as MAX_SUM
            FROM DrawSums
        """
        df_sum_stats = self.sqlite.executeQuery(query_sum_stats)
        target_sum = int(df_sum_stats['AVG_SUM'][0])
        
        print(f"{Fore.YELLOW}[1] TOP PREDICTED NUMBERS (Scored Algorithm)")
        print(f"{Fore.YELLOW}{'-'*80}")
        print(f"{Fore.WHITE}Top 15 numbers based on frequency, recent trends, and overdue analysis:\n")
        print(prediction_df[['NUMBER', 'TOTAL_SCORE', 'TOTAL_FREQ', 'RECENT_FREQ', 'DAYS_OVERDUE']].head(15).to_string(index=False))
        
        # Generate 3 different prediction sets
        print(f"\n{Fore.YELLOW}[2] RECOMMENDED NUMBER SETS FOR NEXT DRAW")
        print(f"{Fore.YELLOW}{'-'*80}\n")
        
        # SET 1: Top-scored balanced set
        top_numbers = prediction_df['NUMBER'].head(12).tolist()
        
        # Try to create balanced set (mix of odd/even, different ranges)
        def generate_balanced_set(candidates, target_sum_range=(target_sum-20, target_sum+20)):
            import random
            best_set = None
            best_score = -1
            
            # Try multiple combinations
            for attempt in range(1000):
                random.shuffle(candidates)
                test_set = sorted(candidates[:6])
                
                # Check balance
                odd_count = sum(1 for n in test_set if n % 2 == 1)
                total = sum(test_set)
                
                # Score this set
                score = 0
                if 2 <= odd_count <= 4:  # Prefer 2-4 odd numbers
                    score += 20
                if target_sum_range[0] <= total <= target_sum_range[1]:
                    score += 30
                
                # Check range distribution
                ranges = [0] * 5
                for n in test_set:
                    if n <= 10: ranges[0] += 1
                    elif n <= 20: ranges[1] += 1
                    elif n <= 30: ranges[2] += 1
                    elif n <= 40: ranges[3] += 1
                    else: ranges[4] += 1
                
                # Prefer sets with numbers from different ranges
                if all(r <= 2 for r in ranges):
                    score += 20
                
                if score > best_score:
                    best_score = score
                    best_set = test_set
            
            return best_set
        
        # Generate sets
        set1 = generate_balanced_set(top_numbers[:12])
        set2 = generate_balanced_set(top_numbers[3:15])
        set3 = generate_balanced_set(top_numbers[6:18])
        
        def print_set_analysis(numbers, set_name):
            odd = sum(1 for n in numbers if n % 2 == 1)
            even = 6 - odd
            total = sum(numbers)
            print(f"{Fore.GREEN}{set_name}: {', '.join(map(str, numbers))}")
            print(f"{Fore.WHITE}  Odd/Even: {odd}/{even} | Sum: {total} | Range: {numbers[0]}-{numbers[-1]}\n")
        
        print_set_analysis(set1, "SET 1 (Highest Confidence)")
        print_set_analysis(set2, "SET 2 (Alternative)")
        print_set_analysis(set3, "SET 3 (Balanced Mix)")
        
        print(f"{Fore.YELLOW}[3] KEY STATISTICAL INSIGHTS")
        print(f"{Fore.YELLOW}{'-'*80}")
        
        # Hot numbers
        hot_5 = ', '.join(map(str, df_freq.head(5)['NUMBER'].tolist()))
        print(f"{Fore.GREEN}  Hottest Numbers (All-Time): {hot_5}")
        
        # Recent hot
        recent_hot_5 = ', '.join(map(str, df_recent.head(5)['NUMBER'].tolist()))
        print(f"{Fore.GREEN}  Hottest Numbers (Last 50 draws): {recent_hot_5}")
        
        # Most overdue
        overdue_5 = ', '.join(map(str, df_overdue.head(5)['NUMBER'].tolist()))
        print(f"{Fore.BLUE}  Most Overdue Numbers: {overdue_5}")
        
        # Sum statistics
        print(f"\n{Fore.CYAN}  Expected Sum Range: {target_sum-20} to {target_sum+20}")
        print(f"{Fore.CYAN}  Historical Avg: {target_sum}")
        
        print(f"\n{Fore.YELLOW}[4] PREDICTION STRATEGY RECOMMENDATIONS")
        print(f"{Fore.YELLOW}{'-'*80}")
        
        print(f"{Fore.GREEN}  [PRIMARY STRATEGY]")
        print(f"{Fore.WHITE}    1. Use SET 1 as your primary choice (highest confidence)")
        print(f"{Fore.WHITE}    2. Consider SET 2 or SET 3 for additional combinations")
        
        print(f"\n{Fore.GREEN}  [CUSTOMIZATION TIPS]")
        print(f"{Fore.WHITE}    3. Mix numbers from 'Hottest' and 'Most Overdue' lists")
        print(f"{Fore.WHITE}    4. Maintain 3 odd / 3 even balance (or 2/4 or 4/2)")
        print(f"{Fore.WHITE}    5. Keep sum between {target_sum-20} and {target_sum+20}")
        print(f"{Fore.WHITE}    6. Include numbers from different decade ranges")
        
        print(f"\n{Fore.GREEN}  [IMPORTANT DISCLAIMER]")
        print(f"{Fore.YELLOW}    Remember: Lottery draws are RANDOM. These predictions")
        print(f"{Fore.YELLOW}    are based on historical patterns, but past performance")
        print(f"{Fore.YELLOW}    does NOT guarantee future results. Play responsibly!")
        
        print(f"\n{Fore.CYAN}{'='*80}\n")


# Example usage
if __name__ == "__main__":
    analyzer = LotteryPatternAnalyzer(
        dbFileName="predictions.sqlite",
        datasetTable="raffleDataset",
        raffle="Bonoloto"
    )
    
    # Run full analysis
    analyzer.run_full_analysis()
    
    # Or run individual analyses:
    # analyzer.analyze_number_frequency(number_type="N1", top_n=10)
    # analyzer.analyze_overdue_numbers(number_type="N1")
    # analyzer.analyze_number_pairs("N1", "N2", top_n=15)
