import sqliteClass
import commonFunctions
# --------------------------------
import pandas as pd
import datetime as dt
import colorama
import sklearn
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential 
from keras.layers import Dense, LSTM
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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

        query = f"""
            SELECT
                {self.dateDesc}, {self.unpivotedTableTitleDesc}, {self.unpivotedTableValueDesc}
            FROM {self.tableDesc}
            ORDER BY {self.dateDesc}, {self.unpivotedTableTitleDesc}
        """

        # Get all the dataset
        df = self.sqlite.executeQuery(query)

        commonFunctions.printInfo(df, colorama.Fore.BLUE)

        df = df.pivot(index=self.dateDesc, columns=self.unpivotedTableTitleDesc, values=self.unpivotedTableValueDesc)

        commonFunctions.printInfo(df, colorama.Fore.BLUE)

        plt.style.use('fivethirtyeight')

        #Visualize the closing price history
        #We create a plot with name 'Close Price History'
        plt.figure(figsize=(16,8))
        plt.title('Bonoloto')

        #We give the plot the data (the closing price of our stock)
        plt.plot(df[self.unpivotColumnsDesc])

        #We label the axis
        plt.xlabel(self.dateDesc, fontsize=18)
        plt.ylabel(self.unpivotedTableValueDesc, fontsize=18)

        # Function add a legend  
        plt.legend(self.unpivotColumnsDesc, loc ="lower right")

        # Avoid overlapping
        plt.xticks(np.arange(0, len(df)+1, 365))
        plt.gcf().autofmt_xdate()

        #We show the plot
        plt.show()

        for typeValue in self.unpivotColumnsDesc:

            #Create a new dataframe with only the typeValue column
            data = df.filter([typeValue])

            #Convert the dataframe to a numpy array
            dataset = data.values
            #Get the number of rows to train the model on
            training_data_len = math.ceil(len(dataset) * 0.8)

            #Scale the data
            scaler = MinMaxScaler(feature_range=(0,1))
            scaled_data = scaler.fit_transform(dataset)

            #Create the training data set 
            #Create the scaled training data set
            train_data = scaled_data[0:training_data_len, :]
            #Split the data into x_train and y_train data sets
            x_train = []
            y_train = []
            #We create a loop
            for i in range(60, len(train_data)):
                x_train.append(train_data[i-60:i, 0]) #Will conaint 60 values (0-59)
                y_train.append(train_data[i, 0]) #Will contain the 61th value (60)
                if i <= 60:
                    print(x_train)
                    print(y_train)
                    print()


            #Convert the x_train and y_train to numpy arrays
            x_train, y_train = np.array(x_train), np.array(y_train)

            #Reshape the data
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            x_train.shape

            #Build the LSTM model
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model.add(LSTM(50, return_sequences=False))
            model.add(Dense(25))
            model.add(Dense(1))

            #Compile the model
            model.compile(optimizer='adam', loss='mean_squared_error')

            #Train the model
            model.fit(x_train, y_train, batch_size=1, epochs=1)

            #Create the testing data set
            #Create a new array containing scaled values from index 1738 to 2247
            test_data = scaled_data[training_data_len - 60:]
            #Create the data set x_test and y_test
            x_test = []
            y_test = dataset[training_data_len:, :]
            for i in range(60, len(test_data)):
                x_test.append(test_data[i-60:i, 0])

            #Convert the data to a numpy array
            x_test = np.array(x_test)

            #Reshape the data
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

            #Get the model's predicted price values for the x_test data set
            predictions = model.predict(x_test)
            predictions = scaler.inverse_transform(predictions)
            predictions

            #Evaluate model (get the root mean quared error (RMSE))
            rmse = np.sqrt( np.mean( predictions - y_test )**2 )
            rmse

            #Plot the data
            train = data[:training_data_len]
            valid = data[training_data_len:]
            valid['Predictions'] = predictions
            #Visualize the data
            plt.figure(figsize=(16,8))
            plt.title('Model')
            plt.xlabel('Date', fontsize=18)
            plt.ylabel('Close Price USD ($)', fontsize=18)
            plt.plot(train[typeValue])
            plt.plot(valid[[typeValue, 'Predictions']])
            plt.legend(['Train', 'Validation', 'Predictions'], loc='lower right')
            plt.show()

            X_FUTURE = 1
            predictions = np.array([])
            last = x_test[-1]

            for i in range(X_FUTURE):
                curr_prediction = model.predict(np.array([last]))
                print(curr_prediction)
                last = np.concatenate([last[1:], curr_prediction])
                predictions = np.concatenate([predictions, curr_prediction[0]])

            predictions = scaler.inverse_transform([predictions])[0]
            print(predictions)

            dicts = []
            curr_date = data.index[-1]
            for i in range(X_FUTURE):
                curr_date = curr_date + dt.timedelta(days=1)
                dicts.append({'Predictions':predictions[i], "Date": curr_date})

            new_data = pd.DataFrame(dicts).set_index("Date")

            #Plot the data
            train = data
            #Visualize the data
            plt.figure(figsize=(16,8))
            plt.title('Model')
            plt.xlabel('Date', fontsize=18)
            plt.ylabel('Close Price USD ($)', fontsize=18)
            plt.plot(train[typeValue])
            plt.plot(new_data['Predictions'])
            plt.legend(['Train', 'Predictions'], loc='lower right')
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