import sqliteClass
import pandas as pd
import datetime as dt
import sklearn

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

        # Unpivot df columns
        df = pd.melt(df,
            id_vars=self.dateDesc, value_vars=self.unpivotColumnsDesc,
            var_name=self.unpivotedTableTitleDesc, value_name=self.unpivotedTableValueDesc
        )

        df = df.sort_values(by=self.sortUnpivotedDf)

        self.insertData(sourceDf=df)

    def insertData(self, sourceDf:pd.DataFrame):

        sqlite = sqliteClass.db()

        sqlite.insertIntoFromPandasDf(sourceDf=sourceDf, targetTable=self.tempTableDesc)

        query = f"""
            DELETE FROM {self.tempTableDesc};

            INSERT INTO {self.tableDesc} ({self.dateDesc}, {self.unpivotedTableTitleDesc}, {self.unpivotedTableValueDesc})
            SELECT tmp.{self.dateDesc}, tmp.{self.unpivotedTableTitleDesc}, tmp.{self.unpivotedTableValueDesc}
            FROM {self.tempTableDesc} tmp
            LEFT JOIN {self.tableDesc} t
            ON t.{self.dateDesc} = tmp.{self.dateDesc}
            AND t.{self.unpivotedTableTitleDesc} = tmp.{self.unpivotedTableTitleDesc}
            WHERE t.{self.dateDesc} IS NULL;

            DELETE FROM {self.tempTableDesc};
        """

        sqlite.executeQuery(query)

    def predictions(self):

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