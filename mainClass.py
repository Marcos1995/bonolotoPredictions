import sqliteClass
import commonFunctions as cf
# --------------------------------
import pandas as pd
import datetime as dt
import colorama
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential 
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class predictData:

    def __init__(self, dbFileName: str, datasetTable: str, predictionsTable: str, batch_size: int, epoch: int):

        self.dbFileName = dbFileName
        self.datasetTable = datasetTable
        self.tempDatasetTable = "TMP_" + self.datasetTable
        self.predictionsTable = predictionsTable

        self.batch_size = batch_size
        self.epoch = epoch

        self.startDate = self.endDate = ""

        # Initialize sqlite database connection
        self.sqlite = sqliteClass.db(
            dbFileName=self.dbFileName,
            datasetTable=self.datasetTable,
            predictionsTable=self.predictionsTable
        )

        self.raffleProperties = {
            "Bonoloto": "https://docs.google.com/spreadsheets/u/0/d/175SqVQ3E7PFZ0ebwr2o98Kb6YEAwSUykGFh6ascEfI0/pubhtml/sheet?headers=false&gid=0"
        }

        self.raffle = self.url = ""

        self.raffleDesc = "RAFFLE"
        self.dateDesc = "RESULT_DATE"
        self.unpivotedTableTitleDesc = "NUMBER_TYPE"
        self.unpivotedTableValueDesc = "NUMBER"
        
        # Rename all columns
        self.unpivotColumnsDesc = ["N1", "N2", "N3", "N4", "N5", "N6", "Complementario", "Reintegro"]

        # Append date column in first position
        self.allColumnsDesc = [self.dateDesc] + self.unpivotColumnsDesc

        self.sortUnpivotedDf = [self.dateDesc, self.unpivotedTableTitleDesc, self.unpivotedTableValueDesc]

        self.validationDays = 0

        self.startDateDesc = "START_DATE"
        self.endDateDesc = "END_DATE"
        self.predictionDateDesc = "PREDICTION_DATE"
        self.predictionNumberDesc = "PREDICTION_NUMBER"
        self.floorNumberDesc = "FLOOR_NUMBER"
        self.ceilNumberDesc = "CEIL_NUMBER"
        self.batchSizeDesc = "BATCH_SIZE"
        self.epochDesc = "EPOCH"

        self.getDataset()


    # Get the dataset for the predefined raffles
    def getDataset(self):

        for raffle, url in self.raffleProperties.items():
            self.raffle = raffle
            self.url = url

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
                FROM {self.datasetTable}
                WHERE {self.raffleDesc} = '{self.raffle}'
            """

            # Set max date from our dataset
            maxDate = self.sqlite.executeQuery(query)[self.dateDesc][0]

            # Format date as type datetime.date
            maxDate = dt.datetime.strptime(maxDate, '%Y-%m-%d').date()

            # Filter dataset to select the rows to insert only
            df = df[df[self.dateDesc] > maxDate]

            # If there are no rows to insert
            if df.empty:
                self.getDataToPredict()

            else: # There are rows to insert

                # Unpivot df columns
                df = pd.melt(df,
                    id_vars=self.dateDesc, value_vars=self.unpivotColumnsDesc,
                    var_name=self.unpivotedTableTitleDesc, value_name=self.unpivotedTableValueDesc
                )

                # Order columns
                df = df.sort_values(by=self.sortUnpivotedDf)

                # Inserting the column at the beginning in the DataFrame
                df.insert(loc=0, column=self.raffleDesc, value=self.raffle)

                # Insert dataframe to the DB
                self.insertData(sourceDf=df)


    # Insert dataframe in the DB
    def insertData(self, sourceDf: pd.DataFrame()):

        # Truncate TMP table
        query = f"DELETE FROM {self.tempDatasetTable}"

        self.sqlite.executeQuery(query)

        # Insert data into TMP table
        self.sqlite.insertIntoFromPandasDf(sourceDf=sourceDf, targetTable=self.tempDatasetTable)

        # Insert new data only into the final table from the TMP table
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

        # Truncate TMP table
        query = f"DELETE FROM {self.tempDatasetTable}"

        self.sqlite.executeQuery(query)

        # Let's predict!
        self.getDataToPredict()

    # Predict the results for any day and any number type
    def getDataToPredict(self):

        # Get historic data
        query = f"""
            SELECT
                {self.dateDesc}, {self.unpivotedTableTitleDesc}, {self.unpivotedTableValueDesc}
            FROM {self.datasetTable}
            WHERE {self.raffleDesc} = '{self.raffle}'
            AND {self.dateDesc} >= (SELECT date(MAX({self.dateDesc}),'-1 year') FROM {self.datasetTable})
            ORDER BY {self.dateDesc}, {self.unpivotedTableTitleDesc}
        """

        # Get all the dataset
        df = self.sqlite.executeQuery(query)

        self.startDate = df[self.dateDesc][0]
        self.endDate = df[self.dateDesc].iloc[-1]

        # Pivot data
        df = df.pivot(index=self.dateDesc, columns=self.unpivotedTableTitleDesc, values=self.unpivotedTableValueDesc)

        cf.printInfo(df, colorama.Fore.BLUE)

        """
        plt.style.use('fivethirtyeight')

        # Visualize the closing price history
        # We create a plot with name 'Close Price History'
        plt.figure(figsize=(16,8))
        plt.title(self.raffle)

        # We give the plot the data (the closing price of our stock)
        plt.plot(df[self.unpivotColumnsDesc])

        # We label the axis
        plt.xlabel(self.dateDesc, fontsize=18)
        plt.ylabel(self.unpivotedTableValueDesc, fontsize=18)

        # Function add a legend  
        plt.legend(self.unpivotColumnsDesc, loc ="lower right")

        # Avoid overlapping
        plt.xticks(np.arange(0, len(df)+1, 20))
        plt.gcf().autofmt_xdate()

        # We show the plot
        #plt.show()
        """

        self.createModelByColumn(df)


    def createModelByColumn(self, df: pd.DataFrame()):

        # Get historic data
        query = f"""
            SELECT *
            FROM {self.predictionsTable}
            LIMIT 0
        """

        # Get all the dataset
        predictionsToInsert = self.sqlite.executeQuery(query)

        predictionsToInsert.drop(['ID', 'ENTRY_DATE'], axis=1, inplace=True) 

        for typeValue in self.unpivotColumnsDesc:

            # Create a new dataframe with only the typeValue column
            data = df.filter([typeValue])

            # Convert the dataframe to a numpy array
            dataset = data.values

            # Get the number of rows to train the model on
            training_data_len = math.ceil(len(dataset) * 0.8)
            self.validationDays = 60 #len(dataset) - training_data_len

            # Scale the data
            scaler = MinMaxScaler(feature_range=(0,1))
            scaled_data = scaler.fit_transform(dataset)

            # Create the training data set 
            # Create the scaled training data set
            train_data = scaled_data[0:training_data_len, :]

            # Split the data into x_train and y_train data sets
            x_train = []
            y_train = []

            # We create a loop
            for i in range(self.validationDays, len(train_data)):
                x_train.append(train_data[i-self.validationDays:i, 0]) #Will conaint self.validationDays values (0-59)
                y_train.append(train_data[i, 0]) #Will contain the 61th value (self.validationDays)
                if i <= self.validationDays:
                    print(x_train)
                    print(y_train)
                    print()

            # Convert the x_train and y_train to numpy arrays
            x_train, y_train = np.array(x_train), np.array(y_train)

            # Reshape the data
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            x_train.shape

            # Build the LSTM model
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model.add(LSTM(50, return_sequences=False))
            model.add(Dense(25))
            model.add(Dense(1))

            # Compile the model
            model.compile(optimizer='adam', loss='mean_squared_error')

            # Train the model
            model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epoch)

            # Create the testing data set
            # Create a new array containing scaled values from index 1738 to 2247
            test_data = scaled_data[training_data_len - self.validationDays:]

            # Create the data set x_test and y_test
            x_test = []
            y_test = dataset[training_data_len:, :]
            for i in range(self.validationDays, len(test_data)):
                x_test.append(test_data[i-self.validationDays:i, 0])

            # Convert the data to a numpy array
            x_test = np.array(x_test)

            # Reshape the data
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

            # Get the model's predicted price values for the x_test data set
            predictions = model.predict(x_test)
            predictions = scaler.inverse_transform(predictions)

            # Evaluate model (get the root mean squared error (RMSE))
            rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
            rmse

            # Plot the data
            train = data[:training_data_len]
            valid = data[training_data_len:]
            valid['Predictions'] = predictions

            """
            # Visualize the data
            plt.style.use('fivethirtyeight')
            plt.figure(figsize=(16,8))
            plt.title('Model')
            plt.xlabel(self.dateDesc, fontsize=18)
            plt.ylabel(self.unpivotedTableValueDesc, fontsize=18)
            plt.plot(train[typeValue])
            plt.plot(valid[[typeValue, 'Predictions']])
            plt.legend(['Train', 'Validation', 'Predictions'], loc='lower right')

            # Avoid overlapping
            plt.xticks(np.arange(0, len(df)+1, 20))
            plt.gcf().autofmt_xdate()

            #plt.show()
            """

            X_FUTURE = 1
            predictions = np.array([])
            last = x_test[-1]

            for i in range(X_FUTURE):
                curr_prediction = model.predict(np.array([last]))
                print(curr_prediction)
                last = np.concatenate([last[1:], curr_prediction])
                predictions = np.concatenate([predictions, curr_prediction[0]])

            # Inverse-transform with correct shape regardless of horizon length
            predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).ravel()
            cf.printInfo(predictions, colorama.Fore.BLUE)

            dicts = []

            # Get last date from index robustly as datetime.date
            last_index_value = data.index[-1]
            if isinstance(last_index_value, dt.datetime):
                curr_date = last_index_value.date()
            elif isinstance(last_index_value, dt.date):
                curr_date = last_index_value
            else:
                curr_date = pd.to_datetime(last_index_value).date()
            
            # Insert last historic data value to connect the presiction line with the next day value
            dicts.append({'Predictions': data[typeValue].iloc[-1], self.dateDesc: str(curr_date)})

            for i in range(X_FUTURE):
                curr_date = curr_date + dt.timedelta(days=1)
                dicts.append({'Predictions': predictions[i], self.dateDesc: str(curr_date)})

            new_data = pd.DataFrame(dicts).set_index(self.dateDesc)
            cf.printInfo(new_data, colorama.Fore.GREEN)

            # Plot the data
            train = data

            """
            # Visualize the data
            plt.style.use('fivethirtyeight')
            plt.figure(figsize=(16,8))
            plt.title(self.raffle)
            #plt.xlabel(self.dateDesc, fontsize=18)
            #plt.ylabel('Close Price USD ($)', fontsize=18)
            plt.plot(train[typeValue])
            plt.plot(new_data['Predictions'])
            plt.legend(['Train', 'Predictions'], loc='upper left')

            # Avoid overlapping
            plt.xticks(np.arange(0, len(df)+1, 20))
            plt.gcf().autofmt_xdate()
            """

            # Visualize the data
            plt.style.use('fivethirtyeight')
            plt.figure(figsize=(16,8))
            plt.title(f"{self.raffle} {curr_date} {typeValue} {new_data['Predictions'].iloc[-1]}")
            #plt.xlabel(self.dateDesc, fontsize=18)
            #plt.ylabel(self.unpivotedTableValueDesc, fontsize=18)
            plt.plot(train[typeValue])
            plt.plot(valid[[typeValue, 'Predictions']])
            plt.plot(new_data['Predictions'])
            plt.legend(['Train', 'Validation', 'Predictions', 'Schedule'], loc='upper left')

            # Avoid overlapping
            plt.xticks(np.arange(0, len(df)+1, int(len(df) / 10)))
            plt.gcf().autofmt_xdate()

            dicts = [{
                self.raffleDesc: self.raffle,
                self.startDateDesc: self.startDate,
                self.endDateDesc: self.endDate,
                self.predictionDateDesc: curr_date,
                self.unpivotedTableTitleDesc: typeValue,
                self.predictionNumberDesc: new_data['Predictions'].iloc[-1],
                self.floorNumberDesc: math.floor(new_data['Predictions'].iloc[-1]),
                self.ceilNumberDesc: math.ceil(new_data['Predictions'].iloc[-1]),
                self.batchSizeDesc: self.batch_size,
                self.epochDesc: self.epoch
            }]

            # Append new prediction row(s) in a pandas 2.x compatible way
            predictionsToInsert = pd.concat([predictionsToInsert, pd.DataFrame(dicts)], ignore_index=True)

        self.sqlite.insertIntoFromPandasDf(sourceDf=predictionsToInsert, targetTable=self.predictionsTable)

        plt.show()



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