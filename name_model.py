import numpy
import pandas
import statsmodels.api as sm

def custom_heuristic(file_path):

    predictions = {}
    df = pandas.read_csv(file_path)
    #get title from Name
    f = lambda x: x['Name'].split(",")[1].split(".")[0].rstrip().lstrip()
    df["Title"] = df.apply(f, axis=1)
    #get surname
    f = lambda x: x['Name'].split(",")[0].rstrip().lstrip()
    df["Surname"] = df.apply(f, axis=1)
    #refactor title
    df["Title"][df["Title"].str.contains("Capt|Don|Major|Col|Rev|Dr|Sir|Mr|Jonkheer")==True] = 'man'
    df["Title"][df["Title"].str.contains("Mrs|the Countess|Dona|Mme|Mlle|Ms|Miss|Lady")==True] = 'woman'
    
    #create new feature: woman-child group
    df['Surname'][df["Title"]=='man'] = 'noGroup'
    df['SurnameFreq'] = df.groupby('Surname')['Surname'].transform('count')
    df['Surname'][df["SurnameFreq"]<=1] = 'noGroup'
    #calculate survival rate
    df['SurviveRate'] = df['Survived'].groupby(df['Surname']).transform('mean')
    print df.groupby('SurviveRate').count()
    

    for _, passenger in df.iterrows():
        passenger_id = passenger['PassengerId']
        if passenger['Sex'] == 'female':
            predictions[passenger_id] = 1
        elif passenger['Sex'] == 'male':
            predictions[passenger_id] = 0
        elif (passenger['Sex'] == 'female' and passenger['Pclass'] == 3):
            predictions[passenger_id] = 0
        elif (passenger['Sex'] == 'male' and passenger['Name'].find('Master') != -1) :
            predictions[passenger_id] = 1
        elif (passenger['Sex'] == 'male' and passenger['Age'] < 8 ) :
            predictions[passenger_id] = 1    
            
    return predictions


file_path = "/Users/pengchengliu/go/src/github.com/user/Titanic/data/train.csv"
predictions = custom_heuristic(file_path)
print predictions