import seaborn as sns
import pandas as pd
import datetime
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib

import re

# check the correlation between columns and plot heatmap of the correlations 
def check_corr(df):
    c= df.corr().abs()

    s = c.unstack()
    so = s.sort_values(ascending=False)
    so.columns=['name1','name2','correlation']
    pd.set_option('display.height', 500)

    so=so.drop_duplicates()
    correlatedfeatures=so[(so>0.85) & (so<1.0) ].index.get_level_values(1).unique().tolist()
    print(so)
    
    f, ax = plt.subplots(figsize=(5, 4))
    corr = df.corr()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax)
    
# check if there is duplicates in a dataframe
def check_duplicates(df, key):
    return df[df.duplicated([key],keep=False)].sort_values(key)

# fillna by the previous value in the group 

def shiftfillna(df,column, groupcolumn):
    misscount1=df[column].isnull().sum()
    df[column]=df.groupby(groupcolumn).apply(lambda x: x[column].fillna(x[column].shift(1))).reset_index(level=0,drop=True)
    misscount2=df[column].isnull().sum()
    print(misscount1, misscount2)
    return misscount1, misscount2


# clean a column
def cleanStrColumn (df, column):
    df.is_copy=False
    tobeconverted=[]
    for i in df[column].unique():
        try:float(i)
        except:tobeconverted.append(i)
    print('to be converted', tobeconverted)
    for i in tobeconverted:
        print('try convert',i)
        try:
            digit=float(re.findall('\d+', i)[0])
            print(digit, i)
            df[column].replace(i, digit ,inplace=True)
        except: pass        
    [df[column].replace(i,np.nan,inplace=True) for i in ['99999','None','0','9999','hosting','E14 9NN','NA','Na','999','NONE']]
    df.loc[df[column].str.contains('nlimit')==True,column]=np.nan
    print('Missing records', df[column].isnull().sum())
    print(df[column].astype(float).max())
    df[column]=df[column].astype('float',errors='coerce')
#    df[column].fillna(df[column].astype(float).max(), inplace=True)
    print(df[column].unique())

# rolling mean of a column 
def rollingmean(df,column, window):
    dfroll=df.groupby('AcctId').apply(lambda x: x[column].rolling(window).mean()).reset_index()
    newcolumn=column+'rollmean'+str(window)
    dfroll.columns=['AcctId','secondindex',newcolumn]
    return pd.concat([df,dfroll[newcolumn]],axis=1)

# make boxplot
def boxplot(df,target_column,group_column):
    #sns.set(style="white")
    plt.figure(figsize=(3,2))
    lim1=df[target_column].quantile(0.01)
    lim2=df[target_column].quantile(0.975)
    df.boxplot(column=target_column,by=group_column)
    plt.ylim(lim1,lim2)
    print('skew',stats.skew(df[target_column]))
    if stats.skew(df[target_column])>2.:
        plt.yscale('log')
        plt.ylim(lim1,lim2)
    plt.suptitle("")

    

# make kde plot
def kdeplot(df, target_column, group_column):
    plt.figure()
    print(target_column, group_column)
    df.loc[df[group_column]==True,target_column].plot(kind='kde', color='#8cd9b3',label=group_column)
    df.loc[df[group_column]==False,target_column].plot(kind='kde',color='#e08585',label='Non'+ group_column)
    plt.legend(loc='best')
    plt.xlim(0,df[target_column].quantile(0.95))

# make comparison histgram stack on top of each other
def histplot(df, target_column, group_column):
    f, axes = plt.subplots(2, 1, figsize=(4, 5), sharex=True)
    df.loc[df[group_column]==True,target_column].hist(bins=10, color='#8cd9b3',label=group_column,ax=axes[0])
    df.loc[df[group_column]==False,target_column].hist(bins=10, color='#e08585',label='Non'+group_column,ax=axes[1])
    plt.legend(loc='best')
    plt.xlim(0,df[target_column].quantile(0.975))


import scipy.stats as stats

# check p values and select related  variables
def checkPvalue (df,group_column):    
    selected=[]
    features=[]
    pvalues=[]
    for column in df.select_dtypes(include=['float64','int64']).columns:
        pvalue=stats.ttest_ind(df.loc[df[group_column]==True,column],
                               df.loc[df[group_column]==False,column],equal_var=False).pvalue
        print(column, pvalue)
        features.append(column)
        pvalues.append(pvalue)
        if pvalue<0.1 : 
            selected.append(column)
    dfpvalue=pd.DataFrame([features, pvalues]).T
    dfpvalue.columns=['features','pvalue']
    dfpvalue.sort_values('pvalue',inplace=True)
    return selected, dfpvalue 

# convert country to continent
def continent(df,countrycolumn):
    df['Continent']=np.nan
#    print (df[countrycolumn].unique())
    df.loc[df[countrycolumn].isin(['USA', 'United States','New York','Stamford','California','US','Boston','Kansas City','Austin','CA',\
                                   'Los Angeles','San Francisco','McLean','Canton','Chicago','Park City','Salem','Baltimore','Braintree',\
                                  'Greenwich','Montclair','Brooklyn','Seattle','Atlanta','Frontier Renewables, LLC','South Carolina','Texas',\
                                  'Bedminster NJ','Dallas','Philadelphia','Florida','Colorado','Miramar','United Sates','St Louis',\
                                   'Pleasanton','Fort Collins','Scottsdale','Carlsbad','Newark','U.S.A','United States of America','United Stats',\
                                  'Irvine','Harvey','Hillsboro','Radnor','Bakersfield','APRO','Virgin Islands, British','Virgin Islands','America','USA']),\
           ['Continent']]='USA'
    df.loc[df[countrycolumn].isin(['Canada','Vancouver','Burlington','Woodstock','Calgary','Toronto','Quebec','Corner Brook','CHI-ROGERS',\
                                  'Ontario','Ottawa','Montreal','Kanata','Regina','Mississauga','canada','CANADA','N2C 1L3','Pittsburgh',\
                                  'Magog','416-687-8716','FMI','Canada']),\
           ['Continent']]='Canada'
    
    df.loc[df[countrycolumn].isin(['Germany','Albania','Andorra','Austria','Belarus','Belgium','Bosnia and Herzegovina','Bulgaria',\
                                 'Croatia','Cyprus','Czech Republic','Denmark','Estonia','Finland','France','Georgia','Germany',\
                                 'Greece','Hungary','Iceland','Ireland','Italy','Latvia','Liechtenstein','Lithuania','Luxembourg',\
                                 'Macedonia','Malta','Moldova','Monaco','Montenegro','Netherlands','Norway','Poland','Portugal',\
                                 'Romania','Russia','San Marino','Serbia','Slovakia','Slovenia','Spain','Sweden','Switzerland',\
                                 'Turkey','Ukraine','United Kingdom','Vatican','Guernsey','Isle of Man','Jersey','Russian Federation',\
                                 'PORTUGAL','Europe','Paris','Milan','Warsaw, Poland','SPAIN','RUSSIAN FEDERATION','The Netherlands',\
                                 'Sweesen','Nederland','DEUTSCHLAND','Netherlands Antilles','Slovenija',\
                                   'CommProve','Wales','Deutschland','Gibraltar','Bosnia and Hercegovina']),['Continent']]='Europe'
    df.loc[df[countrycolumn].isin(['UK','England','London', 'United Kingdom','ENGLAND']),['Continent']]='UK'

    df.loc[df[countrycolumn].isin(['Pakistan','China','India','Israel','Russia','Japan','South Korea','Taiwan','Singapore','Philippines',\
                                 'Malaysia','Oman','Sri Lanka','Thailand','Syria','Auckland','Sydney',\
                                 'Australia','Fiji','Kiribati','Marshall Islands','Micronesia','Nauru','New Zealand','Palau',\
                                 'Papua New Guinea','Samoa','Solomon Islands','Tonga','Tuvalu','Vanuatu','Indonesia','Hong Kong',\
                                 'United Arab Emirates','Egypt','Saudi Arabia','Qatar','Kuwait','Korea, Republic of','Mongolia','Lebanon',\
                                  'Kazakstan','U.A.E.','Korea','Vietnam','Bahrain','Kingdom of Bahrain','Herzliya','Asia/Pacific Region',\
                                  'Republic of Kazakhstan','Tel Aviv','Kazakhstan']),\
           ['Continent']]='Asia'

    df.loc[df[countrycolumn].isin(['Belize','Costa Rica','El Salvador','Guatemala','Honduras','Mexico','Nicaragua','Panama','Costa Rica']),
           ['Continent']]='CentralAmerica'
    df.loc[df[countrycolumn].isin(['Anguilla','Antigua and Barbuda','Aruba','Bahamas','Barbados','Bonaire, Saint Eustatius and Saba',\
                                   'British Virgin Islands','Cayman Islands','Cuba','Curaçao','Dominica','Dominican Republic','Grenada',\
                                   'Guadeloupe','Haiti','Jamaica','Martinique','Monserrat','Puerto Rico','Saint-Barthélemy',\
                                   'St. Kitts and Nevis','Saint Lucia','Saint Martin','Saint Vincent and the Grenadines','Sint Maarten',\
                                   'Trinidad and Tobago','Turks and Caicos Islands','Virgin Islands (US)','Bermuda']), \
           ['Continent']]='Caribbean'
                                
    df.loc[df[countrycolumn].isin(['Argentina','Bolivia','Brazil','Chile','Colombia','Ecuador','Falkland Islands (Malvinas)',\
                                   'French Guiana','Guyana','Paraguay','Peru','Suriname','Uruguay','Venezuela','Columbia',\
                                    'Trinidad & Tobago','Brasil','Pamplona']),['Continent']]='SouthAmerica'

    df.loc[df[countrycolumn].isin(['Togo','Nigeria','South Africa','Burundi','Ghana','Rwanda','Mozambique','Mauritius','Morocco','Algeria','Ethiopia',\
                                   'Kenya','Zambia','Senegal','Sudan','Uganda','Mauritus','Jordan']), ['Continent']]='Africa' 

    print(df[df.Continent.isnull()==True][countrycolumn].value_counts())
    df.Continent.fillna('Other',inplace=True)
    df.Continent=df.Continent.astype('category')    
    print(df.Continent.value_counts())

