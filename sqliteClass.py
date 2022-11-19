import commonFunctions
import sqlite3
import colorama
import pandas as pd

class db:

    def __init__(self, dbFileName, datasetTable, tempDatasetTable, predictionsTable):

        self.dbFileName = dbFileName
        self.datasetTable = datasetTable
        self.predictionsTable = predictionsTable

        self.tempDatasetTable = "TMP_" + self.datasetTable

        # Create database and tables if they doesn't already exists
        self.createDatabaseStructureIfNotExists()


    # This function creates the database and the tables if they doesn't exist
    def createDatabaseStructureIfNotExists(self):

        # Creating table as per requirement
        query = """
            CREATE TABLE IF NOT EXISTS 'raffleDataset' (
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
        query = """
            CREATE TABLE IF NOT EXISTS 'TMP_raffleDataset' (
                'RAFFLE' VARCHAR(30) NOT NULL,
                'RESULT_DATE' DATE NOT NULL,
                'NUMBER_TYPE' VARCHAR(30) NOT NULL,
                'NUMBER' INTEGER NOT NULL,
                'ENTRY_DATE' DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """

        self.executeQuery(query)

        # Creating predictions table as per requirement
        query = """
            CREATE TABLE IF NOT EXISTS 'rafflePredictions' (
                'ID' INTEGER,
                'RAFFLE' VARCHAR(30) NOT NULL,
                'START_DATE' DATE NOT NULL,
                'END_DATE' DATE NOT NULL,
                'PREDICTION_DATE' DATE NOT NULL,
                'NUMBER_TYPE' VARCHAR(30) NOT NULL,
                'PREDICTION_NUMBER' FLOAT NOT NULL,
                'ENTRY_DATE' DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY('ID' AUTOINCREMENT)
            );
        """

        self.executeQuery(query)


    # Prepare the pandas DataFrame data to be inserted in a table
    def insertIntoFromPandasDf(self, sourceDf=None, targetTable: str=None):

        # Validations
        if sourceDf is None or targetTable is None:
            return

        commonFunctions.printInfo(sourceDf, colorama.Fore.BLUE)

        values = ""

        # Prepare column names to be inserted
        columnNames = "'" + "', '".join(list(sourceDf)) + "'"

        # For each row, concatenate the values to prepare the INSERT INTO statement
        for i in range(len(sourceDf)):

            # Split with ", " the values to be inserted (needed if we want to insert more than 1 row in the same insert condition)
            if i > 0:
                values += ", "

            # Concatenate values to be inserted from each row
            rowValues = sourceDf.iloc[i,:].apply(str).values
            values += "('" + "', '".join(rowValues) + "')"

        # Create query statement
        query = f"""
                INSERT INTO {targetTable} ({columnNames})
                VALUES {values};
            """

        # Execute the query
        self.executeQuery(query)


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
