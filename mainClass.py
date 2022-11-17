import sqliteClass
import commonFunctions
import pandas as pd
import datetime as dt
import colorama
#import sklearn

class predictData:

    def __init__(self):

        self.sqlite = sqliteClass.db()

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

        self.tableDesc = "bonolotoHistory"
        self.tempTableDesc = "TMP_" + self.tableDesc

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
        df[self.dateDesc] = pd.to_datetime(df[self.dateDesc], format='%d/%m/%Y').dt.date

        query = f"""
            SELECT
                COALESCE(MAX({self.dateDesc}), '1970-01-01') AS {self.dateDesc}
            FROM {self.tableDesc}
        """

        # Set max date from our dataset
        maxDate = self.sqlite.executeQuery(query)[self.dateDesc][0]

        # Format date as type datetime.date
        maxDate = dt.datetime.strptime(maxDate, '%Y-%m-%d').date()

        # Filter dataset to select the rows to insert only
        df = df[df[self.dateDesc] > maxDate]

        # If there are no rows to insert
        if df.empty:
            self.predictions()

        else: # There are rows to insert

            # Unpivot df columns
            df = pd.melt(df,
                id_vars=self.dateDesc, value_vars=self.unpivotColumnsDesc,
                var_name=self.unpivotedTableTitleDesc, value_name=self.unpivotedTableValueDesc
            )

            # Order columns
            df = df.sort_values(by=self.sortUnpivotedDf)

            # Insert dataframe to the DB
            self.insertData(sourceDf=df)

    # Insert dataframe in the DB
    def insertData(self, sourceDf:pd.DataFrame()):

        query = f"DELETE FROM {self.tempTableDesc}"

        self.sqlite.executeQuery(query)

        self.sqlite.insertIntoFromPandasDf(sourceDf=sourceDf, targetTable=self.tempTableDesc)

        query = f"""
            INSERT INTO {self.tableDesc} ({self.dateDesc}, {self.unpivotedTableTitleDesc}, {self.unpivotedTableValueDesc})
            SELECT tmp.{self.dateDesc}, tmp.{self.unpivotedTableTitleDesc}, tmp.{self.unpivotedTableValueDesc}
            FROM {self.tempTableDesc} tmp
            LEFT JOIN {self.tableDesc} t
            ON t.{self.dateDesc} = tmp.{self.dateDesc}
            AND t.{self.unpivotedTableTitleDesc} = tmp.{self.unpivotedTableTitleDesc}
            WHERE t.{self.dateDesc} IS NULL
        """

        self.sqlite.executeQuery(query)

        query = f"DELETE FROM {self.tempTableDesc}"

        self.sqlite.executeQuery(query)

    # Predict the results for any day and any number type
    def predictions(self):
        1
        """
        # Import train_test_split from sklearn.model_selection using the import keyword.
        from sklearn.model_selection import train_test_split
        # Import os module using the import keyword
        import os
        # Import dataset using read_csv() function by pasing the dataset name as
        # an argument to it.
        # Store it in a variable.
        bike_dataset = pd.read_csv("bikeDataset.csv")
        # Make a copy of the original given dataset and store it in another variable.
        bike = bike_dataset.copy()
        # Give the columns to be updated list as static input and store it in a variable
        categorical_column_updated = ['season', 'yr', 'mnth', 'weathersit', 'holiday']
        bike = pd.get_dummies(bike, columns=categorical_column_updated)
        # separate the dependent and independent variables into two data frames.
        X = bike.drop(['cnt'], axis=1)
        Y = bike['cnt']
        # Divide the dataset into 80 percent training and 20 percent testing.
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=.20, random_state=0
        )

        #On our dataset, we're going to build a Decision Tree Model.
        from sklearn.tree import DecisionTreeRegressor
        #We pass max_depth as argument to decision Tree Regressor
        DT_model = DecisionTreeRegressor(max_depth=5).fit(X_train,Y_train)
        #Predictions based on data testing
        DT_prediction = DT_model.predict(X_test) 
        #Print the value of prediction
        print(DT_prediction)

        #On our dataset, we're going to build a KNN model.
        from sklearn.neighbors import KNeighborsRegressor
        #We pass n_neighborss as argument to KNeighborsRegressor
        KNN_model = KNeighborsRegressor(n_neighbors=3).fit(X_train,Y_train)
        #Predictions based on data testing
        KNN_predict = KNN_model.predict(X_test)
        #Print the value of prediction
        print(KNN_predict)
        """