import sqliteClass
import pandas as pd
import datetime as dt

class predictData:

    def __init__(self):
        # Historic dataset for the Bonoloto
        self.url = "https://docs.google.com/spreadsheets/u/0/d/175SqVQ3E7PFZ0ebwr2o98Kb6YEAwSUykGFh6ascEfI0/pubhtml/sheet?headers=false&gid=0"
        
        # DB date column name
        self.dateDesc = "FECHA"
        
        # Rename all columns
        self.unpivotColumnsDesc = ["N1", "N2", "N3", "N4", "N5", "N6", "COMPLEMENTARIO", "REINTEGRO"]

        # Append date column in first position
        self.allColumnsDesc = [self.dateDesc] + self.unpivotColumnsDesc

        self.unpivotedTableTitleDesc = "TIPO"

        self.unpivotedTableValueDesc = "VALOR"

        self.sortUnpivotedDf = [self.dateDesc, self.unpivotedTableTitleDesc, self.unpivotedTableValueDesc]

        self.getDataset()

    def getDataset(self):

        # Returns list of all tables on page
        tables = pd.read_html(self.url, header=1)

        # Select table
        df = tables[0]

        # Drop first column as it's an added index
        df.drop(df.columns[0], axis=1, inplace=True)

        # Rename all columns
        df.columns = self.allColumnsDesc

        # Drop rows with NaN values 
        df = df.dropna()

        # Drop all rows with a length over 15. There is a large text in the dataset talking about the pandemic break
        df = df.loc[df[self.dateDesc].str.len() <= 15]

        # Set FECHA as date
        df[self.dateDesc] = pd.to_datetime(df[self.dateDesc]).dt.date

        print(df)

        # Unpivot df columns
        df = pd.melt(df,
            id_vars=self.dateDesc, value_vars=self.unpivotColumnsDesc,
            var_name=self.unpivotedTableTitleDesc, value_name=self.unpivotedTableValueDesc
        )

        df = df.sort_values(by=self.sortUnpivotedDf)

        print(df)

        self.insertData(sourceDf=df)

    def insertData(self, sourceDf:pd.DataFrame):

        sqlite = sqliteClass.db()

        sqlite.insertIntoFromPandasDf(sourceDf=sourceDf, targetTable="bonolotoResults")