import sqliteClass
import commonFunctions as cf
import webScraper
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

        self.raffleProperties = {
            "Bonoloto": [
                "https://docs.google.com/spreadsheets/u/0/d/175SqVQ3E7PFZ0ebwr2o98Kb6YEAwSUykGFh6ascEfI0/pubhtml/sheet?headers=false&gid=1",  # 1988 - 2012
                "https://docs.google.com/spreadsheets/u/0/d/175SqVQ3E7PFZ0ebwr2o98Kb6YEAwSUykGFh6ascEfI0/pubhtml/sheet?headers=false&gid=0"  # 2013 - Present
            ]
        }

        self.raffle = self.url = ""

        self.raffleDesc = "RAFFLE"
        self.dateDesc = "RESULT_DATE"
        self.unpivotedTableTitleDesc = "NUMBER_TYPE"
        self.unpivotedTableValueDesc = "NUMBER"

        self.unpivotColumnsDesc = ["N1", "N2", "N3", "N4", "N5", "N6", "Complementario", "Reintegro"]
        self.allColumnsDesc = [self.dateDesc] + self.unpivotColumnsDesc
        self.sortUnpivotedDf = [self.dateDesc, self.unpivotedTableTitleDesc, self.unpivotedTableValueDesc]
        self.getDataset()

    def getDataset(self):

        for raffle, url in self.raffleProperties.items():
            self.raffle = raffle

            if isinstance(url, str):
                url = [url]

            totalDf = pd.DataFrame()

            for urlLink in url:
                self.url = urlLink
                tables = pd.read_html(self.url, header=1)
                df = tables[0]
                df.drop(df.columns[0], axis=1, inplace=True)
                df.columns = self.allColumnsDesc
                df = df.dropna(subset=[self.dateDesc])
                df[self.dateDesc] = pd.to_datetime(df[self.dateDesc], dayfirst=True, errors="coerce")
                df = df.dropna(subset=[self.dateDesc])
                df[self.dateDesc] = df[self.dateDesc].dt.date
                for col in self.unpivotColumnsDesc:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna(subset=self.unpivotColumnsDesc)
                df[self.unpivotColumnsDesc] = df[self.unpivotColumnsDesc].round().astype(int)
                totalDf = pd.concat([totalDf, df], ignore_index=True)

            query = f"""
                SELECT
                    COALESCE(MAX({self.dateDesc}), '1970-01-01') AS {self.dateDesc}
                FROM {self.datasetTable}
                WHERE {self.raffleDesc} = '{self.raffle}'
            """

            cf.printInfo(desc=query, color=colorama.Fore.GREEN)
            maxDate = dt.datetime.strptime(
                self.sqlite.executeQuery(query)[self.dateDesc][0], "%Y-%m-%d"
            ).date()
            totalDf = totalDf[totalDf[self.dateDesc] > maxDate]

            if totalDf.empty:
                cf.printInfo("No new data from Google Sheets. Scraping official website for latest results...", colorama.Fore.YELLOW)
            else:
                totalDf = pd.melt(
                    totalDf,
                    id_vars=self.dateDesc, value_vars=self.unpivotColumnsDesc,
                    var_name=self.unpivotedTableTitleDesc, value_name=self.unpivotedTableValueDesc
                )
                totalDf = totalDf.sort_values(by=self.sortUnpivotedDf)
                totalDf.insert(loc=0, column=self.raffleDesc, value=self.raffle)
                self.insertData(sourceDf=totalDf)
                maxDate = dt.datetime.strptime(
                    self.sqlite.executeQuery(query)[self.dateDesc][0], "%Y-%m-%d"
                ).date()
                cf.printInfo("Checking official website for any more recent results...", colorama.Fore.YELLOW)

            self.scrapeLatestResults(maxDate)
            # ponytail: 21 strategies, 30-draw walk-forward, none beat chance (best 0.90 vs 0.74; 95% bar ~1.01). No predictor.

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

        query = f"DELETE FROM {self.tempDatasetTable}"
        self.sqlite.executeQuery(query)
        self.sqlite.insertIntoFromPandasDf(sourceDf=sourceDf, targetTable=self.tempDatasetTable)

        query = f"""
            INSERT INTO {self.datasetTable} ({self.raffleDesc}, {self.dateDesc}, {self.unpivotedTableTitleDesc}, {self.unpivotedTableValueDesc})
            SELECT '{self.raffle}' as {self.raffleDesc}, tmp.{self.dateDesc}, tmp.{self.unpivotedTableTitleDesc}, tmp.{self.unpivotedTableValueDesc}
            FROM {self.tempDatasetTable} tmp
            LEFT JOIN {self.datasetTable} t
            ON t.{self.raffleDesc} = tmp.{self.raffleDesc}
            AND t.{self.dateDesc} = tmp.{self.dateDesc}
            AND t.{self.unpivotedTableTitleDesc} = tmp.{self.unpivotedTableTitleDesc}
            WHERE t.{self.dateDesc} IS NULL
        """
        self.sqlite.executeQuery(query)
        self.sqlite.executeQuery(f"DELETE FROM {self.tempDatasetTable}")
