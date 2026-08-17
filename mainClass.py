import sqliteClass
import commonFunctions as cf
import webScraper
import raffles
# --------------------------------
import pandas as pd
import datetime as dt
import colorama


class predictData:

    def __init__(self, dbFileName: str, datasetTable: str, predictionsTable: str):

        self.dbFileName = dbFileName
        self.datasetTable = datasetTable
        self.tempDatasetTable = "TMP_" + self.datasetTable
        self.predictionsTable = predictionsTable

        self.sqlite = sqliteClass.db(
            dbFileName=self.dbFileName,
            datasetTable=self.datasetTable,
            predictionsTable=self.predictionsTable
        )

        self.raffle = ""
        self.raffleDesc = "RAFFLE"
        self.dateDesc = "RESULT_DATE"
        self.unpivotedTableTitleDesc = "NUMBER_TYPE"
        self.unpivotedTableValueDesc = "NUMBER"
        self.unpivotColumnsDesc = ["N1", "N2", "N3", "N4", "N5", "N6", "Complementario", "Reintegro"]
        self.sortUnpivotedDf = [self.dateDesc, self.unpivotedTableTitleDesc, self.unpivotedTableValueDesc]
        self.getDataset()

    def _max_date(self, raffle):
        query = f"""
            SELECT COALESCE(MAX({self.dateDesc}), '1970-01-01') AS {self.dateDesc}
            FROM {self.datasetTable}
            WHERE {self.raffleDesc} = '{raffle}'
        """
        return dt.datetime.strptime(
            self.sqlite.executeQuery(query)[self.dateDesc][0], "%Y-%m-%d"
        ).date()

    def getDataset(self):
        for raffle, spec in raffles.GAMES.items():
            self.raffle = raffle
            long_df = raffles.to_long(spec, raffle)
            max_date = self._max_date(raffle)
            if long_df.empty:
                cf.printInfo(f"{raffle}: CSV empty", colorama.Fore.YELLOW)
                continue
            long_df[self.dateDesc] = pd.to_datetime(long_df[self.dateDesc]).dt.date
            new_df = long_df[long_df[self.dateDesc] > max_date]
            if new_df.empty:
                cf.printInfo(f"{raffle}: no new CSV rows (max={max_date})", colorama.Fore.YELLOW)
            else:
                cf.printInfo(f"{raffle}: inserting {new_df[self.dateDesc].nunique()} new draws", colorama.Fore.GREEN)
                self.insertData(sourceDf=new_df.sort_values(by=self.sortUnpivotedDf))
                max_date = self._max_date(raffle)
            if raffle == "Bonoloto":
                self.scrapeLatestResults(max_date)
        # ponytail: multi-game Lotoideas CSV ingest + edgeHunt; no sel+confirm edge on any game.

    def scrapeLatestResults(self, maxDate):
        try:
            scraped_results = webScraper.scrape_bonoloto(headless=True, max_results=10)

            if not scraped_results:
                cf.printInfo("No results scraped from website.", colorama.Fore.YELLOW)
                return

            rows = []
            for result in scraped_results:
                result_date_str = result.get("date")
                if not result_date_str:
                    continue

                result_date = dt.datetime.strptime(result_date_str, "%Y-%m-%d").date()
                if result_date <= maxDate:
                    continue

                numbers = result.get("numbers", [])
                if len(numbers) != 6:
                    continue
                numbers = sorted(int(n) for n in numbers)

                for i, num in enumerate(numbers):
                    rows.append({
                        self.raffleDesc: self.raffle,
                        self.dateDesc: result_date,
                        self.unpivotedTableTitleDesc: self.unpivotColumnsDesc[i],
                        self.unpivotedTableValueDesc: num
                    })

                if result.get("complementario") is not None:
                    rows.append({
                        self.raffleDesc: self.raffle,
                        self.dateDesc: result_date,
                        self.unpivotedTableTitleDesc: "Complementario",
                        self.unpivotedTableValueDesc: int(result["complementario"])
                    })

                if result.get("reintegro") is not None:
                    rows.append({
                        self.raffleDesc: self.raffle,
                        self.dateDesc: result_date,
                        self.unpivotedTableTitleDesc: "Reintegro",
                        self.unpivotedTableValueDesc: int(result["reintegro"])
                    })

            if rows:
                scrapedDf = pd.DataFrame(rows)
                unique_dates = scrapedDf[self.dateDesc].unique()
                cf.printInfo(
                    f"Scraped {len(unique_dates)} new draw(s) from official website: {sorted([str(d) for d in unique_dates])}",
                    colorama.Fore.GREEN
                )
                self.insertData(sourceDf=scrapedDf)
            else:
                cf.printInfo("Database is up to date with latest scraped results.", colorama.Fore.GREEN)

        except Exception as e:
            cf.printInfo(f"Error during web scraping: {e}. Proceeding with existing data...", colorama.Fore.RED)

    def insertData(self, sourceDf: pd.DataFrame):
        # ponytail: pandas to_sql; string-concat INSERT dies on ~100k historic rows
        import sqlite3
        df = sourceDf.copy()
        df[self.dateDesc] = df[self.dateDesc].astype(str)
        con = sqlite3.connect(self.dbFileName)
        try:
            df.to_sql(self.tempDatasetTable, con, if_exists="replace", index=False)
            con.execute(
                f"""
                INSERT INTO {self.datasetTable} ({self.raffleDesc}, {self.dateDesc}, {self.unpivotedTableTitleDesc}, {self.unpivotedTableValueDesc})
                SELECT '{self.raffle}', tmp.{self.dateDesc}, tmp.{self.unpivotedTableTitleDesc}, tmp.{self.unpivotedTableValueDesc}
                FROM {self.tempDatasetTable} tmp
                LEFT JOIN {self.datasetTable} t
                  ON t.{self.raffleDesc} = tmp.{self.raffleDesc}
                 AND t.{self.dateDesc} = tmp.{self.dateDesc}
                 AND t.{self.unpivotedTableTitleDesc} = tmp.{self.unpivotedTableTitleDesc}
                WHERE t.{self.dateDesc} IS NULL
                """
            )
            con.execute(f"DELETE FROM {self.tempDatasetTable}")
            con.commit()
        finally:
            con.close()
