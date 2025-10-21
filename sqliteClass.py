import commonFunctions
import sqlite3
import colorama
import pandas as pd

class db:

    def __init__(self, dbFileName, datasetTable, predictionsTable):

        self.dbFileName = dbFileName
        self.datasetTable = datasetTable
        self.tempDatasetTable = "TMP_" + self.datasetTable
        self.predictionsTable = predictionsTable

        # Create database and tables if they doesn't already exists
        self.createDatabaseStructureIfNotExists()


    # This function creates the database and the tables if they doesn't exist
    def createDatabaseStructureIfNotExists(self):

        # Creating table as per requirement
        query = f"""
            CREATE TABLE IF NOT EXISTS '{self.datasetTable}' (
                'ID' INTEGER,
                'RAFFLE' VARCHAR(30) NOT NULL,
                'RESULT_DATE' DATE NOT NULL,
                'NUMBER_TYPE' VARCHAR(30) NOT NULL,
                'NUMBER' INTEGER NOT NULL,
                'ENTRY_DATE' DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY('ID' AUTOINCREMENT)
            );
        """

        self.executeQuery(query)

        # Creating TMP table as per requirement
        query = f"""
            CREATE TABLE IF NOT EXISTS '{self.tempDatasetTable}' (
                'RAFFLE' VARCHAR(30) NOT NULL,
                'RESULT_DATE' DATE NOT NULL,
                'NUMBER_TYPE' VARCHAR(30) NOT NULL,
                'NUMBER' INTEGER NOT NULL,
                'ENTRY_DATE' DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """

        self.executeQuery(query)

        # Creating predictions table as per requirement
        query = f"""
            CREATE TABLE IF NOT EXISTS '{self.predictionsTable}' (
                'ID' INTEGER,
                'RAFFLE' VARCHAR(30) NOT NULL,
                'START_DATE' DATE NOT NULL,
                'END_DATE' DATE NOT NULL,
                'PREDICTION_DATE' DATE NOT NULL,
                'NUMBER_TYPE' VARCHAR(30) NOT NULL,
                'PREDICTION_NUMBER' FLOAT NOT NULL,
                'FLOOR_NUMBER' INTEGER NOT NULL,
                'CEIL_NUMBER' INTEGER NOT NULL,
                'BATCH_SIZE' INTEGER NOT NULL,
                'EPOCH' INTEGER NOT NULL,
                'ENTRY_DATE' DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY('ID' AUTOINCREMENT)
            );
        """

        self.executeQuery(query)


    # Prepare the pandas DataFrame data to be inserted in a table
    def insertIntoFromPandasDf(self, sourceDf=None, targetTable: str=None):
        
        # Validations
        if sourceDf is None or targetTable is None or sourceDf.empty:
            return

        # Log the dataframe preview
        commonFunctions.printInfo(sourceDf, colorama.Fore.BLUE)

        # Prepare column names and parameter placeholders safely
        column_names = list(sourceDf.columns)
        column_names_sql = ", ".join([f'"{c}"' for c in column_names])
        placeholders_sql = ", ".join(["?"] * len(column_names))
        insert_sql = f"INSERT INTO {targetTable} ({column_names_sql}) VALUES ({placeholders_sql})"

        # Convert dataframe rows to tuples; keep None for NaN so sqlite stores NULL
        def _normalize_value(v):
            if pd.isna(v):
                return None
            return v

        rows = [tuple(_normalize_value(v) for v in row) for row in sourceDf.itertuples(index=False, name=None)]

        # Execute parameterized bulk insert for security and performance
        conn = None
        try:
            conn = sqlite3.connect(self.dbFileName)
            cursor = conn.cursor()
            cursor.executemany(insert_sql, rows)
            conn.commit()
        finally:
            if conn is not None:
                conn.close()


    # Execute any query
    def executeQuery(self, query: str=None):
        
        # Validation
        if query is None:
            return

        commonFunctions.printInfo(query, colorama.Fore.CYAN)

        # Verify and assign which type of query is it, commit or not commit one
        keyWords = ["INSERT", "CREATE", "ALTER", "DELETE", "UPDATE"]
        isToCommitTransaction = any(command in query for command in keyWords)

        try:

            # Open connection
            conn = sqlite3.connect(self.dbFileName)

            # INSERT, UPDATE or DELETE statements
            if isToCommitTransaction:

                # Create cursor
                cursor = conn.cursor()

                # Execute query
                cursor.execute(query)

                # Commit transaction
                conn.commit()
            
            else: # SELECT statement, returns pandas DataFrame

                # Declare pandas df
                df = pd.DataFrame()

                # Execute query
                cursor = conn.execute(query)

                # Fetch all data
                selectedData = cursor.fetchall()

                # Get column names in order
                columnNames = list(map(lambda x: x[0], cursor.description))

                # Assign pandas df
                df = pd.DataFrame(selectedData, columns=columnNames)

            # Close conn
            conn.close()

            # If we executed a SELECT statement, return a formatted pandas dataFrame
            if not isToCommitTransaction:
                return df

        except Exception as e:
            commonFunctions.printInfo(f"Error en executeQuery() {e}", colorama.Fore.RED)
            #commonFunctions.printInfo(query, colorama.Fore.RED)
            exit()
