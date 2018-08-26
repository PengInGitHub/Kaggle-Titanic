#place to store re-usable methods
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re


def load_data():
    file_path = "/Users/pengchengliu/go/src/github.com/user/Titanic/data/"
    train = pd.read_csv(file_path+"train.csv")
    test = pd.read_csv(file_path+"test.csv")
    combine = [train, test]
    return train, test, combine

def missing_value(data):
    #missing data visualization
    #create table
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending=False)
    table = pd.concat([total, percent],axis=1,keys=['Total','Percent'])
    table = table[table['Percent']>0]
    #create chart
    plt.subplots(figsize=(8,6))
    plt.xticks(rotation='90')
    sns.barplot(table.index, table["Percent"],color="green",alpha=0.5)
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data by feature' , fontsize=15)
    return table

def get_title(name):
    titel_search = re.search(' ([A-Za-z]+)\.', name)
    if titel_search:
        return titel_search.group(1)
    return ""
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    