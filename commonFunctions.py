import datetime as dt
import colorama

# ------------------------------------------------------------------------------------

# Format percentages (example: 2.3456789 --> 134.5679 %)
def formatPercentages(val):
    return round((val - 1) * 100, 4)


# Convert boolean values to int (True -> 1 and False -> 0). Else (just in case) -> -1
def boolToInt(val):
    if val == "True" or val == True:
        res = 1
    elif val == "False" or val == False:
        res = 0
    else:
        res = -1

    return res


# Print any data with the datetime to debug properly (really useful)
def printInfo(desc, color=""):
    print(f"{dt.datetime.now()} // {color}{desc}{colorama.Fore.RESET}")