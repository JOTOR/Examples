def eda_plot_cat(DATAFRAME, TARGET, MAX_ELEMENTS, TIME_RULE='W', W=8, H=4):
    """
    DATAFRAME: Pandas DataFrame to be included in the analysis
    TARGET: Name of the Column with the Metric or Target (***Categorical***)
    MAX_ELEMENTS: Max number of elements to be displayed in the charts for features with a high cardinality
    TIME_RULE: Resampling method for the time related features. Please refer to the pandas.DataFrame.resample documentation
    W: Width of the resulting plots
    H: Height of the resulting plots
    Example: jupyter_eda_plot_cat(DATAFRAME=df, TARGET='SLA_Compliance',MAX_ELEMENTS=15, TIME_RULE='W')
    Result: Descriptive Analytics about the DataFrame and a Series of Plots of the features available in the DataFrame
            against the TARGET
    Note: To be used in a Jupyter Notebook and a Python IDE such as Spyder, PyCharm, Visual Studio Code, etc.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('darkgrid')
    print("GENERATING AUTOMATIC EXPLORATORY DATAFRAME ANALYSIS")
    print("----------------------------------------------------")
    print("----------------------------------------------------")
    print("Features available in the DataSet:")
    print(DATAFRAME.columns)
    print("----------------------------------------------------")
    print("----------------------------------------------------")
    print("DATAFRAME Type of Features:")
    print(DATAFRAME.info())
    print("----------------------------------------------------")
    print("----------------------------------------------------")
    print("Basic Statistics of Features:")
    print(DATAFRAME.describe(include='all'))
    print("----------------------------------------------------")
    print("----------------------------------------------------")
    print("GENERATING AUTOMATIC EXPLORATORY DATAFRAME ANALYSIS PLOTS")
    print("----------------------------------------------------")
    print("----------------------------------------------------")
    for column in DATAFRAME.columns:
    
        if (DATAFRAME[column].dtype=='O' or DATAFRAME[column].dtype=='int64' or DATAFRAME[column].dtype=='int32' or DATAFRAME[column].dtype=='float32' or DATAFRAME[column].dtype=='float64') and (DATAFRAME[column].nunique() == len(DATAFRAME)):
            print("High Cardinality Feature: ","[",column, "]"," Possible unique identifier!")
    
        elif (DATAFRAME[column].dtype=='O') and (DATAFRAME[column].nunique() <= MAX_ELEMENTS):
            DATAFRAME[column].fillna(value='-', inplace=True)
            plt.figure(figsize=(W,H))
            plt.xticks(rotation=90)
            sns.countplot(x=DATAFRAME[column],hue=TARGET,data=DATAFRAME,order = DATAFRAME[column].value_counts().index)
            plt.title(column+" Distribution by "+TARGET)
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
            plt.show()
    
        elif (DATAFRAME[column].dtype=='O') and (DATAFRAME[column].nunique() > MAX_ELEMENTS or DATAFRAME[column].nunique() <= len(DATAFRAME)-1):
            DATAFRAME[column].fillna(value='-', inplace=True)
            cnts = DATAFRAME[column].value_counts()[0:MAX_ELEMENTS].index.values
            cnts = pd.DataFrame(data=cnts, columns=[column])
            cnts['CAT']="TOP_ELEMENT"
            sub_df = pd.merge(DATAFRAME[[column,TARGET]],cnts,on=column,how='left')
            sub_df = sub_df.dropna().drop(columns=['CAT'])
            plt.figure(figsize=(W,H))
            plt.xticks(rotation=90)
            sns.countplot(x=sub_df[column],hue=TARGET,data=sub_df, order = sub_df[column].value_counts().index)
            plt.title("Top "+ str(MAX_ELEMENTS)+" Elements of "+column+" by "+TARGET)
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
            plt.show()

        elif (DATAFRAME[column].dtype=='<M8[ns]'): 
            sub_df = DATAFRAME.loc[:,(column,TARGET)]
            sub_df['Vol'] = 1
            sub_df = sub_df.pivot_table(index=column,columns=TARGET,values='Vol',aggfunc=sum)
            sub_df.resample(TIME_RULE).sum().plot.bar(figsize=(W,H))
            plt.xticks(rotation=90)
            plt.title(column +" "+TIME_RULE+" distribution by " + TARGET)
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
            plt.show()
    
        elif (DATAFRAME[column].dtype=='int64' or DATAFRAME[column].dtype=='float64' or DATAFRAME[column].dtype=='float32' or DATAFRAME[column].dtype=='int32') and (DATAFRAME[column].nunique()<=MAX_ELEMENTS): 
            plt.figure(figsize=(W,H))
            plt.xticks(rotation=90)
            sns.countplot(x=DATAFRAME[column],hue=TARGET,data=DATAFRAME,order = DATAFRAME[column].value_counts().index)
            plt.title(column+" Distribution by "+TARGET)
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
            plt.show()
    
        elif (DATAFRAME[column].dtype=='int64' or DATAFRAME[column].dtype=='float64' or DATAFRAME[column].dtype=='int32' or DATAFRAME[column].dtype=='float32') and (DATAFRAME[column].nunique()>MAX_ELEMENTS): 
            plt.figure(figsize=(W,H))
            plt.xticks(rotation=90)
            sns.boxplot(x=TARGET, y=DATAFRAME[column],data=DATAFRAME)
            plt.title(column+" Distribution by "+TARGET)
            plt.show()
        
        else: print("FEATURE "+"["+column+"] "+"NOT INCLUDED IN ANALYSIS!!!!")

def eda_plot_cont(DATAFRAME, TARGET, MAX_ELEMENTS, TIME_RULE='W', W=8, H=4):
    """
    DATAFRAME: Pandas DataFrame to be included in the analysis
    TARGET: Name of the Column with the Metric or Target (***Continuous***)
    MAX_ELEMENTS: Max number of elements to be displayed in the charts for features with a high cardinality
    TIME_RULE: Resampling method for the time related features. Please refer to the pandas.DataFrame.resample documentation
    W: Width of the resulting plots
    H: Height of the resulting plots
    Example: jupyter_eda_plot_cat(DATAFRAME=df, TARGET='SLA_Compliance',MAX_ELEMENTS=15, TIME_RULE='W')
    Result: Descriptive Analytics about the DATAFRAME and a Series of Plots of the features available in the DATAFRAME
            against the TARGET
    Note: To be used in Jupyter Notebooks and a Python IDE such as Spyder, PyCharm, Visual Studio Code, etc.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('darkgrid')
    print("GENERATING AUTOMATIC EXPLORATORY DATAFRAME ANALYSIS")
    print("----------------------------------------------------")
    print("----------------------------------------------------")
    print("Features available in the DataSet:")
    print(DATAFRAME.columns)
    print("----------------------------------------------------")
    print("----------------------------------------------------")
    print("DATAFRAME Type of Features:")
    print(DATAFRAME.info())
    print("----------------------------------------------------")
    print("----------------------------------------------------")
    print("Basic Statistics of Features:")
    print(DATAFRAME.describe(include='all'))
    print("----------------------------------------------------")
    print("----------------------------------------------------")
    print("GENERATING AUTOMATIC EXPLORATORY DATAFRAME ANALYSIS PLOTS")
    print("----------------------------------------------------")
    print("----------------------------------------------------")
    for column in DATAFRAME.columns:
    
        if (DATAFRAME[column].dtype=='O' or DATAFRAME[column].dtype=='int64' or DATAFRAME[column].dtype=='int32' or DATAFRAME[column].dtype=='float64' or DATAFRAME[column].dtype=='float32') and (DATAFRAME[column].nunique() == len(DATAFRAME)):
            print("High Cardinality Feature: ","[",column, "]"," Possible unique identifier!")
    
        elif (DATAFRAME[column].dtype=='O') and (DATAFRAME[column].nunique() <= MAX_ELEMENTS):
            DATAFRAME[column].fillna(value='-', inplace=True)
            plt.figure(figsize=(W,H))
            plt.xticks(rotation=90)
            sns.boxplot(x=DATAFRAME[column],y=TARGET, data=DATAFRAME)
            plt.title(column+" Distribution by "+TARGET)
            plt.show()
    
        elif (DATAFRAME[column].dtype=='O') and (DATAFRAME[column].nunique() > MAX_ELEMENTS or DATAFRAME[column].nunique() <= len(DATAFRAME)-1):
            DATAFRAME[column].fillna(value='-', inplace=True)
            cnts = DATAFRAME[column].value_counts()[0:MAX_ELEMENTS].index.values
            cnts = pd.DataFrame(data=cnts, columns=[column])
            cnts['CAT']="TOP_ELEMENT"
            sub_df = pd.merge(DATAFRAME[[column,TARGET]],cnts,on=column,how='left')
            sub_df = sub_df.dropna().drop(columns=['CAT'])
            plt.figure(figsize=(W,H))
            plt.xticks(rotation=90)
            sns.boxplot(x=sub_df[column],y=TARGET,data=sub_df)
            plt.title("Top "+ str(MAX_ELEMENTS)+" Elements of "+column+" by "+TARGET)
            plt.show()

        elif (DATAFRAME[column].dtype=='<M8[ns]'): 
            DATAFRAME.loc[:,(column,TARGET)].set_index(column).resample(TIME_RULE).median().plot(figsize=(W,H))
            plt.title(column +" "+TIME_RULE+" distribution by " + TARGET)
            plt.show()
    
        elif (DATAFRAME[column].dtype=='int64' or DATAFRAME[column].dtype=='float64' or DATAFRAME[column].dtype=='int32' or DATAFRAME[column].dtype=='float32'): 
            plt.figure(figsize=(W,H))
            plt.xticks(rotation=90)
            sns.scatterplot(x=DATAFRAME[column],y=TARGET,data=DATAFRAME)
            plt.title(column+" Distribution by "+TARGET)
            plt.show()
    
        else: print("FEATURE "+"["+column+"] "+"NOT INCLUDED IN ANALYSIS!!!!")

def clean_feature_names(DATAFRAME):
    """
    Function used to clean the name of the features, blank spaces, dots and another special characters (?,*)
    are replaced by an underscore character '_' and a lowercase function is applied to the column names
  
    Input: dataframe name
    Output: dataframe columns normalized
  
    Example: 
    df.columns
    ['Color_Reference', 'Number Rooms', 'Area Total', 'Zone']
    clean_feature_names(df) 
    df.columns
    ['color_reference', 'number_rooms', 'area_total', 'zone']
    """
    normalized_names = DATAFRAME.columns.str.replace(".","_",regex=False)
    normalized_names = normalized_names.str.replace(" ","_",regex=False)
    normalized_names = normalized_names.str.replace("(","",regex=False)
    normalized_names = normalized_names.str.replace(")","",regex=False)
    normalized_names = normalized_names.str.replace("/","_",regex=False)
    normalized_names = normalized_names.str.replace("\\","_",regex=False)
    normalized_names = normalized_names.str.replace("?","",regex=False)
    normalized_names = normalized_names.str.replace("¿","",regex=False)
    normalized_names = normalized_names.str.replace("*","",regex=False)
    normalized_names = normalized_names.str.replace("º","",regex=False)
    normalized_names = normalized_names.str.replace("ª","",regex=False)
    normalized_names = normalized_names.str.replace("|","",regex=False)
    normalized_names = normalized_names.str.replace("-","_",regex=False)
    normalized_names = normalized_names.str.replace("·","",regex=False)
    normalized_names = normalized_names.str.replace("@","",regex=False)
    normalized_names = normalized_names.str.replace("$","",regex=False)
    normalized_names = normalized_names.str.replace("+","",regex=False)
    normalized_names = normalized_names.str.lower()
    DATAFRAME.columns = normalized_names

def get_null_data_photo(DATAFRAME, W=12, H=6):
    """
    Function used to get a "photo" of the data, specially the distribution of the null values
    DATAFRAME: Name of the Pandas dataframe to be included in the "photo"
    W: Widht of the "photo"
    H: Height of the "photo"
    Example: get_data_photo(data, 12,6)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('darkgrid')
    print("GENERATING NULL VALUES PLOTS")
    print("-----------------------------")
    print(" ** Caution: Time to complete does depend of the number of rows in the DataFrame **")
    if len(DATAFRAME)<30000:
        plt.figure(figsize=(W,H))
        plt.title("Null Values distribution in the Dataset")
        sns.heatmap(DATAFRAME.isna())
        plt.figure(figsize=(W,H))
        plt.title("Null Values distribution by Feature Name")
        DATAFRAME.isna().sum().sort_values(ascending=False).plot.bar()
    else:
        plt.figure(figsize=(W,H))
        plt.title("Null Values distribution in the Dataset - Using 10% Sample!!")
        sns.heatmap(DATAFRAME.sample(frac=0.1).isna())
        plt.figure(figsize=(W,H))
        plt.title("Null Values distribution by Feature Name")
        DATAFRAME.isna().sum().sort_values(ascending=False).plot.bar()

def show_correlation(DATAFRAME, TARGET=None):
    """
    DATAFRAME: Pandas DataFrame to be included in the analysis/plot
    TARGET: Use "None" If the target is (**continuous**) otherwise use the 
    column name (**categorical**) to be included in the resulting plots as the "Target"
    Note: Resulting plots are considering/including numeric features only 
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('darkgrid')  
    
    FEATURES = []
    for column in DATAFRAME.columns:
        if DATAFRAME[column].dtype != 'O':
            FEATURES.append(column)
            
    if TARGET == None:
        sns.heatmap(DATAFRAME[FEATURES].corr(method='pearson'),annot=True,cmap='bwr')
        plt.title('Pearson Correlation Coefficient of Numeric Features')
        plt.show()
        sns.heatmap(DATAFRAME[FEATURES].corr(method='kendall'),annot=True,cmap='bwr')
        plt.title('Kendall Correlation Coefficient of Numeric Features')
        plt.show()
        sns.heatmap(DATAFRAME[FEATURES].corr(method='spearman'),annot=True,cmap='bwr')
        plt.title('Spearman Correlation Coefficient of Numeric Features')
        plt.show()
        print('Caution: The Matrix Scatterplot of Numeric Features is using a 20% sample of the DATAFRAME')
        sns.pairplot(DATAFRAME[FEATURES].sample(frac=0.2))
    else:
        FEATURES.append(TARGET)
        sns.heatmap(DATAFRAME[FEATURES].corr(method='pearson'),annot=True,cmap='bwr')
        plt.title('Pearson Correlation Coefficient of Numeric Features')
        plt.show()
        sns.heatmap(DATAFRAME[FEATURES].corr(method='kendall'),annot=True,cmap='bwr')
        plt.title('Kendall Correlation Coefficient of Numeric Features')
        plt.show()
        sns.heatmap(DATAFRAME[FEATURES].corr(method='spearman'),annot=True,cmap='bwr')
        plt.title('Spearman Correlation Coefficient of Numeric Features')
        plt.show()
        print('Caution: The Matrix Scatterplot of Numeric Features is using a 20% sample of the DATAFRAME')
        sns.pairplot(DATAFRAME[FEATURES].sample(frac=0.2),hue=TARGET)

def unsupervised_categorization(DATAFRAME, TEXT, NUM_CATS, STOPWORDS, LANGUAGE='english', MAX_WORDS =1000):
    """
    DATAFRAME: Pandas DataFrame to be included the categorization process
    TEXT: String with the column name that contains the text to be categorized
    NUM_CATS: Number of categories to create
    STOPWORDS: String (in lowercase) with additional stopwords to be removed from the TEXT column, 
    example of this input: "lotus notes fcf user have na"
    LANGUAGE: Language of the TEXT column to be categorized
    MAX_WORDS: Most frequent words to be included in the categorization process. Default is 1000
    Notes and References:
     * "nltk" is a dependency, make sure to install and update this library prior to execute this function
     * A LatentDirichletAllocation (Topic Modeling) estimator is fit with the TEXT column
     * Sebastian Raschka, Vahid Mirjalili. Python Machine Learning. 2nd Edition. 
       Birmingham, UK: Packt Publishing, 2017. ISBN: 978-3958457331. Pages 276-277
    Example:
    stop = "lotus notes fcf user have na a"
    unsupervised_categorization(DATAFRAME=DATAFRAME, TEXT='Summary', NUM_CATS=5, STOPWORDS=stop, LANGUAGE='english')
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.metrics import silhouette_score
    import numpy as np
    import pandas as pd
    import re
    from nltk.corpus import stopwords 
    stop_words = set(stopwords.words(LANGUAGE)) 
    
    DATAFRAME[TEXT] = DATAFRAME[TEXT].str.lower()
    DATAFRAME[TEXT] = DATAFRAME[TEXT].apply(lambda x: re.sub('[^\w\s]',' ',str(x)))
    DATAFRAME[TEXT] = DATAFRAME[TEXT].apply(lambda x: re.sub('[0-9]','',str(x)))
    DATAFRAME[TEXT] = DATAFRAME[TEXT].apply(lambda x: " ".join(x for x in str(x).split() if x not in STOPWORDS))
    
    vectorizer = CountVectorizer(max_df=0.1, max_features=MAX_WORDS, stop_words=stop_words)
    categorizer = LatentDirichletAllocation(n_components=NUM_CATS, random_state=1234)
    
    X = vectorizer.fit_transform(DATAFRAME[TEXT].values)
    topics = categorizer.fit_transform(X) 
    DATAFRAME["Category_Number"]= np.argmax(topics, axis=1)
    print("[Category_Number] has been added to DATAFRAME")
    print("----------------------------------------------------------------------------")
    
    for topic in DATAFRAME["Category_Number"].unique():
        print("Category Number: ", topic)
        print(pd.Series(" ".join(DATAFRAME[DATAFRAME["Category_Number"]==topic][TEXT]).split()).value_counts()[0:10])
    print("----------------------------------------------------------------------------")

    print("Volume Distribution by Category Number:")
    print(DATAFRAME['Category_Number'].value_counts())
    print("----------------------------------------------------------------------------")
    #print("Silhouette_Score : {:g}".format(silhouette_score(X,DATAFRAME["Category_Number"])))
    #print("-1 is the worst, +1 is the best")
    print("Categorization process has been completed!!")
        
def reduce_cardinality(DATAFRAME, MAX_ELEMENTS):
    """
    DATAFRAME: Pandas DataFrame to be used in the process
    MAX_ELEMENTS: Number of Top Categories to be defined, categories with a small volume will be replaced by "Other"
    Caution: The category "Other" is replaced in DATAFRAME directly!!
    """
    import pandas as pd
    pd.set_option('mode.chained_assignment',None)
    for column in DATAFRAME.columns:        
        if (DATAFRAME[column].dtype=='O' or DATAFRAME[column].dtype=='int64' or DATAFRAME[column].dtype=='int32' or DATAFRAME[column].dtype=='float32' or DATAFRAME[column].dtype=='float64') and (DATAFRAME[column].nunique() == len(DATAFRAME)):
            print("High Cardinality Feature: ","[",column, "]"," Possible unique identifier, excluded from this process!")
        
        elif (DATAFRAME[column].dtype=='O') and (DATAFRAME[column].nunique() > MAX_ELEMENTS):
            counts = DATAFRAME[column].value_counts()
            mask = DATAFRAME[column].isin(counts[MAX_ELEMENTS:].index)
            DATAFRAME[column].loc[mask]="Other"
            print("Cardinality in feature","[",column,"]","has been reduced")
    
    print("Process has been completed!!")

def reduce_cardinality_single_column(DATAFRAME, MAX_ELEMENTS, COLUMN):
    """
    DATAFRAME: Pandas DataFrame to be used in the process
    MAX_ELEMENTS: Number of Top Categories to be defined, categories with a small volume will be replaced by "Other"
    Caution: The category "Other" is replaced in DATAFRAME directly!!
    """
    import pandas as pd
    pd.set_option('mode.chained_assignment',None)

    if (DATAFRAME[COLUMN].dtype=='O' or DATAFRAME[COLUMN].dtype=='int64' or DATAFRAME[COLUMN].dtype=='int32' or DATAFRAME[COLUMN].dtype=='float32' or DATAFRAME[COLUMN].dtype=='float64') and (DATAFRAME[COLUMN].nunique() == len(DATAFRAME)):
        print("High Cardinality Feature: ","[",COLUMN, "]"," Possible unique identifier, excluded from this process!")
        
    elif (DATAFRAME[COLUMN].dtype=='O') and (DATAFRAME[COLUMN].nunique() > MAX_ELEMENTS):
        counts = DATAFRAME[COLUMN].value_counts()
        mask = DATAFRAME[COLUMN].isin(counts[MAX_ELEMENTS:].index)
        DATAFRAME[COLUMN].loc[mask]="Other"
        print("Cardinality in feature","[",COLUMN,"]","has been reduced")
    
    print("Process has been completed!!")

def supervised_categorization(DFA, TEXT, CATS, DFB, STOPWORDS, LANGUAGE='english'):
    """
    DFA: Pandas DataFrame that contains the text and the categories that were assigned through a 'human' categorization
    TEXT: Column name that contains the Text to be categorized, this Column Name should be the same in both DataFrames
    CATS: Column name that contains the Categories that were assigned through a 'human' process
    DFB: Pandas DataFrame to be categorized based in the Text and Categories of the DataFrame DFA
    STOPWORDS: String (in lowercase) with additional stopwords to be removed from the TEXT column, 
    example of this input: "lotus notes fcf user have na"
    LANGUAGE: Language of the TEXT, 'english' by default
    Notes: 
    * "nltk" is a dependency, make sure to install and update this library prior to execute this function
    * Only to be used if the categories defined in DFA should be extended to DFB and if DFA and DFB have the same domain
    Caution: This categorization is performed by a basic ML Pipeline, a CV accuracy score 
    is provided but no Train/Test split approach was applied
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import MaxAbsScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score, KFold
    import numpy as np
    import re
    from nltk.corpus import stopwords 
    stop_words = set(stopwords.words(LANGUAGE)) 
    import pandas as pd
    pd.set_option('mode.chained_assignment',None)
    
    DFA[TEXT] = DFA[TEXT].apply(lambda x: str(x).lower())
    DFA[TEXT] = DFA[TEXT].apply(lambda x: re.sub('[^\w\s]',' ',str(x)))
    DFA[TEXT] = DFA[TEXT].apply(lambda x: re.sub('[0-9]','',str(x)))
    DFA[TEXT] = DFA[TEXT].apply(lambda x: " ".join(x for x in str(x).split() if x not in STOPWORDS))
    
    DFB[TEXT] = DFB[TEXT].apply(lambda x: str(x).lower())
    DFB[TEXT] = DFB[TEXT].apply(lambda x: re.sub('[^\w\s]',' ',str(x)))
    DFB[TEXT] = DFB[TEXT].apply(lambda x: re.sub('[0-9]','',str(x)))
    DFB[TEXT] = DFB[TEXT].apply(lambda x: " ".join(x for x in str(x).split() if x not in STOPWORDS))

    vectorizer = TfidfVectorizer(strip_accents='ascii', stop_words=stop_words, ngram_range=(1,3), max_features=1000)
    scaler = MaxAbsScaler()
    clf = LogisticRegression(max_iter=500, random_state=1234)
    
    pl = make_pipeline(vectorizer, scaler, clf)
    
    pl.fit(DFA[TEXT],DFA[CATS])
    print("Working in Categorization Process...")
    DFB['Predicted_Category'] = pl.predict(DFB[TEXT])
    print("Predicted categories have been added to the DataFrame...")
    folds = KFold(n_splits=3)
    cvs = cross_val_score(pl, DFA[TEXT],DFA[CATS], cv=folds, scoring='accuracy')
    print("Approx Accuracy %: {:g}".format(np.median(cvs)*100))
    print("Procces has been Completed!!")

def eda_plot_uni(DATAFRAME):
    """
    DATAFRAME: Pandas DataFrame to be used in the analysis
    Notes: 
    - To be used in Jupyter Notebooks Only
    - This function generates a complete univariate analysis of the DATAFRAME features
    - This is a wrapper of the pandas_profiling library, which is a dependency, please
    make sure to install this library prior to execute this function
    (pip install pandas_profiling)
    - For additional information please refer to https://github.com/pandas-profiling/pandas-profiling
    """
    import pandas as pd
    from pandas_profiling import ProfileReport
  
    if len(DATAFRAME) < 80000:
        profile = ProfileReport(DATAFRAME, title="Pandas Profile Report of DATAFRAME")
        profile.to_notebook_iframe()
    else:
        profile = ProfileReport(DATAFRAME, title="Pandas Profile Report of DATAFRAME [Minimal]", minimal=True)
        profile.to_notebook_iframe()

def time_series_anomaly(DATAFRAME, DATE, NUM, W=12, H=4, K=21):
    """
    DATAFRAME: Pandas DataFrame to be used in the analysis
    DATE: Column Name with the Date Feature of the time series
    NUM: Column Name with the numeric feature of the time series
    W: Width of the resulting plot
    H: Height of the resulting plot
    K: Number of neighboors / sensibility of the anomaly detection
    Example: time_series_anomaly(ts,'Day','Volume')
    """
    from sklearn.neighbors import LocalOutlierFactor
    import numpy as np
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-darkgrid')
    
    DATAFRAME.set_index(DATE,inplace=True)
    
    lof = LocalOutlierFactor(n_neighbors=K)
    outs = lof.fit_predict(np.array(DATAFRAME[NUM]).reshape(-1,1))
    DATAFRAME['Anomaly'] = outs
    print("Anomalies have been included in DATAFRAME, Flagged as [-1]")
    print("-----------------------------------------------------------")
    plt.figure(figsize=(W,H))
    plt.xticks(rotation=90)
    plt.plot(DATAFRAME[NUM])
    plt.scatter(x=DATAFRAME.index, y=DATAFRAME[NUM], c=outs, cmap='viridis')
    plt.title('Anomalies in the Time Series')
    
    print(DATAFRAME[DATAFRAME['Anomaly']==-1])
    print("-----------------------------------------------------------")

def anomaly_binary_target(DATAFRAME, FEATURES, TARGET, MAX_ELEMENTS, PL, NL, W=8, H=4):
    """
    DATAFRAME: Pandas DataFrame to be used in the analysis
    FEATURES: List of *Categorical* Features to be included in the analysis. These features 
    should be available in the DATAFRAME
    TARGET: Binary Target to be included in the analysis
    MAX_ELEMENTS: Max number of elements to be included in the resulting plots
    W: Width of the resulting plots
    H: Height of the resulting plots
    PL: Name of the Positive Label, for example, "Met" or 1. (This is one of the 2 values of the target)
    NL: Name of the Negative Label, for example, "Missed" or 0. (This is one of the 2 values of the target)
    Note: The Difference in the labels is equal to (Total_Volume(PL) - Total_Volume(NL))
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('darkgrid')  
    
    if DATAFRAME[TARGET].nunique()>2:
        print("Error: Target is not a Binary feature!! Only Binary features (yes/no, met/missed, 1/0) are allowed")
    else:
        sub_df = DATAFRAME[FEATURES]
        sub_df = pd.get_dummies(data=sub_df)
        sub_df = pd.concat([DATAFRAME[TARGET],sub_df],axis="columns")
        sub_df = sub_df.groupby(TARGET).sum()
        sub_df = np.transpose(sub_df)
        sub_df['Var'] = sub_df.loc[:,PL].astype('int') - sub_df.loc[:,NL].astype('int')
        sub_df.sort_values(by='Var',inplace=True)
        sub_df.head(MAX_ELEMENTS).plot.bar(figsize=(W,H))
        plt.title('Data Elements with Highest Negative Difference')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
        sub_df.tail(MAX_ELEMENTS).plot.bar(figsize=(W,H))
        plt.title('Data Elements with Highest Positive Difference')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
        print('Caution: Completion time does depend of the cardinality of the dataframe features')
        print('----------------------------------------------')
        print('Data Elements with Highest Negative Difference:')
        print(sub_df.head(MAX_ELEMENTS))
        print('----------------------------------------------')
        print('Data Elements with Highest Positive Difference:')
        print(sub_df.tail(MAX_ELEMENTS))
        print('----------------------------------------------')
        
def analyze_data(DATAFRAME, TARGET):
    """
    DATAFRAME: Pandas DataFrame. It is expected to be pre-processed (without unique identifiers or unique values)
    TARGET: Target to be analyzed. If Categorical make sure it is in a Object/String format, 
    otherwise please change or transform the target prior to use this function. If target is Continuous, keep it in 
    the same format
    Note: This function relies on H2OAutoML therefore it is a dependency, make sure to 
    install this library(pip install h2o) prior to use this function
    Expected Output: Top 15 Features with a negative/positive influence over TARGET
    """
    import h2o
    from h2o.automl import H2OAutoML
    h2o.init()
    import pandas as pd
    
    hdf = h2o.H2OFrame(pd.concat([DATAFRAME[TARGET],pd.get_dummies(DATAFRAME.drop(columns=TARGET))],axis='columns'))
    
    y = TARGET
    x = hdf.columns
    x.remove(y)
    
    aml = H2OAutoML(max_models = 3, seed = 1234, max_runtime_secs_per_model=120,include_algos = ["GBM"])
    print("Caution: This may take a while, completion time does depend of the number of rows and cardinality of DATAFRAME")
    aml.train(x = x, y = y, training_frame = hdf)
    lb = aml.leaderboard
    
    model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:,0])
    m = h2o.get_model([mid for mid in model_ids if "GBM" in mid][0])
    print("Top 15 Features with a negative/positive influence over TARGET:")
    print(m.varimp(use_pandas=True)[:15])
    print("----------------------------------------------------------------")
    
def get_transactions_df(DATAFRAME, TRANSACTION_ID, ITEM):
    """
    DATAFRAME: Pandas DataFrame to be used in the function
    TRANSACTION_ID: Name of the column with the Transaction ID
    ITEM: Name of the column with the Item/Product/Category
    Output: A small dataframe with the Unique Transaction ID and a list of Items that
    can be used in a Basket Optimization Analysis or Frequent Patterns Mining
    For additional information please refer to:
    http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/
    """
    tdf = DATAFRAME.groupby(TRANSACTION_ID)[ITEM].agg(lambda x: ','.join(tuple(x))).reset_index()
    tdf['Transaction_List'] = tdf[ITEM].apply(lambda x: str(x).split(','))
    return tdf[[TRANSACTION_ID,'Transaction_List']]

def stratified_sampler(DATAFRAME, GROUP, NUMBER=20):
    """
    DATAFRAME: Pandas dataframe to sample from
    GROUP: Name of the column to be used in the stratified sample
    NUMBER: Sample size
    Notes:
    * Returns a Pandas dataframe with the stratified sample, if there is an item with a value smaller than the required sample size,
    the entire "sub-population" for this item is included in the resulting dataframe
    Example: sdf = stratified_sampler(DATAFRAME=df, GROUP="reported_source", NUMBER=20)
    """
    import pandas as pd
    vdf = DATAFRAME[GROUP].value_counts().reset_index()
    sampled_df = []
    
    for row in vdf.index:
        if vdf[GROUP].iloc[row] >= NUMBER:
            sampled_df.append(DATAFRAME[DATAFRAME[GROUP]==vdf["index"].iloc[row]].sample(NUMBER))
        else:
            sampled_df.append(DATAFRAME[DATAFRAME[GROUP]==vdf["index"].iloc[row]].sample(vdf[GROUP].iloc[row]))
            print("Smaller value of observations than required group for feature: ","[",vdf["index"].iloc[row],"]","; Including all the tickets in the population!!")
    
    stratified_df = pd.concat(sampled_df)
    return(stratified_df)

def combine_tabular_files(PATH, EXT):
    """
    PATH: Complete path where the files are located, for example: 'C:/MyFolder/Reports/'
    EXT: Extension of the files, xlsx, xls and csv are supported
    Result: A csv file with all the rows of the files appended
    Requirements: All files must have the same schema, structure, tab name, column names, file extension and the 
    info to be appended must be located in the first tab
    Example: combine_tabular_files(PATH ='C:/MyFolder/Reports/', EXT='xlsx')
    Note: If you get a ValueError, check the extension of the files located on PATH and 
    make sure is the same used in EXT
    """
    import pandas as pd
    import glob
    
    if(EXT=='csv'):
        data = []
        for file in glob.glob(PATH+'*'+EXT):
            data.append(pd.read_csv(file, encoding='latin-1'))
            print("Reading File: "+str(len(data)))
        df = pd.concat(data)
        df.drop_duplicates(inplace=True)
        df.to_csv(PATH+'Combined_Info.csv', index=False, encoding='utf-8-sig')
        print("Process has been completed, A file called 'Combined_Info.csv' has been included into PATH")
    
    elif(EXT=='xls' or EXT=='xlsx'):
        data = []
        for file in glob.glob(PATH+'*'+EXT):
            data.append(pd.read_excel(file, encoding='latin-1'))
            print("Reading File: "+str(len(data)))
        df = pd.concat(data)
        df.drop_duplicates(inplace=True)
        df.to_csv(PATH+'Combined_Info.csv', index=False, encoding='utf-8-sig')
        print("Process has been completed, A file called 'Combined_Info.csv' has been included into PATH")
    else:
        print("File Extension "+EXT+" is not supported!!")

def auto_eda_binary_negative_plotly(DATAFRAME, TARGET, LABEL_MAP={"Made":1, "Missed":0}):
    """
    This function fits a Logistic Regression and provides the coefficients of the features with the
    highest negative influence over the target and the charts involving those features
    DATAFRAME: Pandas dataframe to be used on the analysis
    TARGET: Name of the dataframe column to be used as the target, it should be binary!!
    LABEL_MAP: Dictionary with the numerical encoding of the target labels, ***it is recommended 
    to asssign a '0' to the negative label (Missed/No/DSAT)***
    Notes: 
    * plotly.express as px is a dependency so make sure to install/upgrade this library prior to execute this function
    * Please remove the unique identifier (if any) of the dataframe prior to execute this function
    * This function does accept continuous features
    * Please do not encode the categories with numbers, use "Yes", "No", "A", "B", "C" instead
    """
    import pandas as pd
    pd.set_option('mode.chained_assignment',None)
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.pipeline import make_pipeline
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    
    types_df = pd.DataFrame(data=DATAFRAME.dtypes, columns=['Type'])
    #Time Stamps or date related features are not supported
    FEATURES = types_df[types_df['Type']!='datetime64[ns]'].index.values
    DATAFRAME = DATAFRAME[FEATURES]
    #Encoding the labels
    DATAFRAME[TARGET] = DATAFRAME[TARGET].map(LABEL_MAP)
    DATAFRAME['VOL'] = 1
    #Removing null records, no impute estimator was used
    DATAFRAME.dropna(inplace=True)
    
    X = pd.get_dummies(DATAFRAME.drop(columns=[TARGET,'VOL']), prefix_sep='*')
    y = DATAFRAME[TARGET]
    
    lr = make_pipeline(MinMaxScaler(feature_range=(0,1)),
                       LogisticRegression(C=1, class_weight='balanced', random_state=1234, max_iter=2500))
    print("Fitting Logistic Regression")
    lr.fit(X, y)
    score = lr.score(X, y)
    score = score*100
    print("Process replicated with an accuracy of {:.2f}%".format(score))
    coef_df = pd.DataFrame(data=lr.steps[1][1].coef_[0], index=X.columns, columns=['Inf']).sort_values(by="Inf")
    
    neg_inf_df = coef_df[coef_df['Inf']<0]
    neg_inf_df = coef_df[0:50]
    
    print("Top 15 Items with a Negative Influence over "+TARGET)
    print(neg_inf_df[0:15])
    
    neg_inf_df.reset_index(inplace=True)
    neg_inf_df.columns = ["Feature_Item", "Imp"]
    #Including an '*'' in order to avoid an index out of range error
    neg_inf_df['Feature_Item']= neg_inf_df['Feature_Item'].apply(lambda x: x+"*")
    neg_inf_df['Feature'] = neg_inf_df['Feature_Item'].apply(lambda x: x.split("*")[0])
    neg_inf_df['Item'] = neg_inf_df['Feature_Item'].apply(lambda x: x.split("*")[1])
    
    DATAFRAME[TARGET] = DATAFRAME[TARGET].astype('category')
    
    for feature in neg_inf_df.Feature.unique():
        if feature in DATAFRAME.columns:
            if types_df[types_df.index==feature]["Type"][0]=="object":
                if DATAFRAME[feature].nunique() < 10:
                    aux = DATAFRAME.groupby([feature, TARGET]).count()['VOL'].reset_index()
                    fig = px.bar(aux, x=feature, y='VOL',color=TARGET, barmode='group', title=feature+" influence over "+TARGET, text='VOL')
                    fig.show()
                    figg = px.histogram(aux, x=feature, y='VOL',color=TARGET, barmode="stack", barnorm='percent', title=feature+" influence over "+TARGET)
                    figg.show()
                else:
                    ITEMS = neg_inf_df[neg_inf_df["Feature"]==feature]['Item'].unique()
                    sub_df = DATAFRAME[DATAFRAME[feature].isin(ITEMS)]
                    aux = sub_df.groupby([feature, TARGET]).count()['VOL'].reset_index()
                    fig = px.bar(aux, x=feature, y='VOL', color=TARGET, barmode='group', title=feature+" influence over "+TARGET, text='VOL')
                    fig.show()
                    figg = px.histogram(aux, x=feature, y='VOL', color=TARGET, barmode="stack", barnorm='percent', title=feature+" influence over "+TARGET)
                    figg.show()
            elif types_df[types_df.index==feature]["Type"][0]=="float64" or types_df[types_df.index==feature]["Type"][0]=="float32" or types_df[types_df.index==feature]["Type"][0]=="int32" or types_df[types_df.index==feature]["Type"][0]=="int64":
                fig = px.box(data_frame=DATAFRAME, x=feature, y=TARGET, orientation='h', title=feature+" influence over "+TARGET)
                fig.show()
                print(DATAFRAME.groupby(TARGET)[feature].describe())

def auto_eda_binary_positive_plotly(DATAFRAME, TARGET, LABEL_MAP={"Made":1, "Missed":0}):
    """
    This function fits a Logistic Regression and provides the coefficients of the features with the
    highest positive influence over the target and the charts involving those features
    DATAFRAME: Pandas dataframe to be used on the analysis
    TARGET: Name of the dataframe column to be used as the target, it should be binary!!
    LABEL_MAP: Dictionary with the numerical encoding of the target labels, ***it is recommended 
    to asssign a '0' to the negative label (Missed/No/DSAT)***
    Notes: 
    * plotly.express as px is a dependency so make sure to install/upgrade this library prior to execute this function
    * Please remove the unique identifier (if any) of the dataframe prior to execute this function
    * This function does accept continuous features
    * Please do not encode the categories with numbers, use "Yes", "No", "A", "B", "C" instead
    """
    import pandas as pd
    pd.set_option('mode.chained_assignment',None)
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.pipeline import make_pipeline
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    
    types_df = pd.DataFrame(data=DATAFRAME.dtypes, columns=['Type'])
    #Time Stamps or date related features are not supported
    FEATURES = types_df[types_df['Type']!='datetime64[ns]'].index.values
    DATAFRAME = DATAFRAME[FEATURES]
    #Encoding the labels
    DATAFRAME[TARGET] = DATAFRAME[TARGET].map(LABEL_MAP)
    DATAFRAME['VOL'] = 1
    #Removing null records, no impute estimator was used
    DATAFRAME.dropna(inplace=True)
    
    X = pd.get_dummies(DATAFRAME.drop(columns=[TARGET,'VOL']), prefix_sep='*')
    y = DATAFRAME[TARGET]
    
    lr = make_pipeline(MinMaxScaler(feature_range=(0,1)),
                       LogisticRegression(C=1, class_weight='balanced', random_state=1234, max_iter=2500))
    print("Fitting Logistic Regression")
    lr.fit(X, y)
    score = lr.score(X, y)
    score = score*100
    print("Process replicated with an accuracy of {:.2f}%".format(score))
    coef_df = pd.DataFrame(data=lr.steps[1][1].coef_[0], index=X.columns, columns=['Inf']).sort_values(by="Inf", ascending=False)
    
    neg_inf_df = coef_df[coef_df['Inf']>0]
    neg_inf_df = coef_df[0:50]
    
    print("Top 15 Items with a Positive Influence over "+TARGET)
    print(neg_inf_df[0:15])
    
    neg_inf_df.reset_index(inplace=True)
    neg_inf_df.columns = ["Feature_Item", "Imp"]
    #Including an '*'' in order to avoid an index out of range error
    neg_inf_df['Feature_Item']= neg_inf_df['Feature_Item'].apply(lambda x: x+"*")
    neg_inf_df['Feature'] = neg_inf_df['Feature_Item'].apply(lambda x: x.split("*")[0])
    neg_inf_df['Item'] = neg_inf_df['Feature_Item'].apply(lambda x: x.split("*")[1])
    
    DATAFRAME[TARGET] = DATAFRAME[TARGET].astype('category')
    
    for feature in neg_inf_df.Feature.unique():
        if feature in DATAFRAME.columns:
            if types_df[types_df.index==feature]["Type"][0]=="object":
                if DATAFRAME[feature].nunique() < 10:
                    aux = DATAFRAME.groupby([feature, TARGET]).count()['VOL'].reset_index()
                    fig = px.bar(aux, x=feature, y='VOL',color=TARGET, barmode='group', title=feature+" influence over "+TARGET, text='VOL')
                    fig.show()
                    figg = px.histogram(aux, x=feature, y='VOL',color=TARGET, barmode="stack", barnorm='percent', title=feature+" influence over "+TARGET)
                    figg.show()
                else:
                    ITEMS = neg_inf_df[neg_inf_df["Feature"]==feature]['Item'].unique()
                    sub_df = DATAFRAME[DATAFRAME[feature].isin(ITEMS)]
                    aux = sub_df.groupby([feature, TARGET]).count()['VOL'].reset_index()
                    fig = px.bar(aux, x=feature, y='VOL', color=TARGET, barmode='group', title=feature+" influence over "+TARGET, text='VOL')
                    fig.show()
                    figg = px.histogram(aux, x=feature, y='VOL', color=TARGET, barmode="stack", barnorm='percent', title=feature+" influence over "+TARGET)
                    figg.show()
            elif types_df[types_df.index==feature]["Type"][0]=="float64" or types_df[types_df.index==feature]["Type"][0]=="float32" or types_df[types_df.index==feature]["Type"][0]=="int32" or types_df[types_df.index==feature]["Type"][0]=="int64":
                fig = px.box(data_frame=DATAFRAME, x=feature, y=TARGET, orientation='h', title=feature+" influence over "+TARGET)
                fig.show()
                print(DATAFRAME.groupby(TARGET)[feature].describe())

def auto_eda_reg_negative_plotly(DATAFRAME, TARGET):
    """
    This function fits a Ridge Regression and provides the coefficients of the features with the
    highest negative influence over the target and the charts involving those features
    DATAFRAME: Pandas dataframe to be used on the analysis
    TARGET: Name of the dataframe column to be used as the target, it should be numeric (discrete or continuous)!!
    Notes: 
    * plotly.express as px is a dependency so make sure to install/upgrade this library prior to execute this function
    * Please remove the unique identifier (if any) of the dataframe prior to execute this function
    * This function does accept continuous features
    * Please do not encode the categories with numbers, use "Yes", "No", "A", "B", "C" instead
    """
    import pandas as pd
    pd.set_option('mode.chained_assignment',None)
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.pipeline import make_pipeline
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    
    types_df = pd.DataFrame(data=DATAFRAME.dtypes, columns=['Type'])
    #Time Stamps or date related features are not supported
    FEATURES = types_df[types_df['Type']!='datetime64[ns]'].index.values
    DATAFRAME = DATAFRAME[FEATURES]
    #Encoding the labels
    DATAFRAME['VOL'] = 1
    #Removing null records, no impute estimator was used
    DATAFRAME.dropna(inplace=True)
    
    X = pd.get_dummies(DATAFRAME.drop(columns=[TARGET,'VOL']), prefix_sep='*')
    y = DATAFRAME[TARGET]
    
    lr = make_pipeline(MinMaxScaler(feature_range=(0, 1)),
                       Ridge(random_state=1234, max_iter=2500, fit_intercept=True, normalize=True))
    print("Fitting Ridge Regression")
    lr.fit(X, y)
    score = lr.score(X, y)
    score = score*100
    print("Process replicated with an accuracy of {:.2f}%".format(score))
    coef_df = pd.DataFrame(data=lr.steps[1][1].coef_.reshape(-1,1), index=X.columns, columns=['Inf']).sort_values(by="Inf")
    
    neg_inf_df = coef_df[coef_df['Inf']<0]
    neg_inf_df = coef_df[0:50]
    
    print("Top 15 Items with a Negative Influence over "+TARGET)
    print(neg_inf_df[0:15])
    
    neg_inf_df.reset_index(inplace=True)
    neg_inf_df.columns = ["Feature_Item", "Imp"]
    #Including an '*'' in order to avoid an index out of range error
    neg_inf_df['Feature_Item']= neg_inf_df['Feature_Item'].apply(lambda x: x+"*")
    neg_inf_df['Feature'] = neg_inf_df['Feature_Item'].apply(lambda x: x.split("*")[0])
    neg_inf_df['Item'] = neg_inf_df['Feature_Item'].apply(lambda x: x.split("*")[1])
        
    for feature in neg_inf_df.Feature.unique():
        if feature in DATAFRAME.columns:
            if types_df[types_df.index==feature]["Type"][0]=="object":
                ITEMS = neg_inf_df[neg_inf_df["Feature"]==feature]['Item'].unique()
                sub_df = DATAFRAME[DATAFRAME[feature].isin(ITEMS)]
                fig = px.box(data_frame =sub_df, x=feature, y=TARGET, title=feature+" influence over "+TARGET)
                fig.show()
            elif types_df[types_df.index==feature]["Type"][0]=="float64" or types_df[types_df.index==feature]["Type"][0]=="float32" or types_df[types_df.index==feature]["Type"][0]=="int32" or types_df[types_df.index==feature]["Type"][0]=="int64":
                fig = px.scatter(data_frame=DATAFRAME, x=feature, y=TARGET, title=feature+" influence over "+TARGET)
                fig.show()
                figg = px.density_heatmap(data_frame=DATAFRAME, x=feature, y=TARGET, title=feature+" influence over "+TARGET)
                figg.show()
                
def auto_eda_reg_positive_plotly(DATAFRAME, TARGET):
    """
    This function fits a Ridge Regression and provides the coefficients of the features with the
    highest positive influence over the target and the charts involving those features
    DATAFRAME: Pandas dataframe to be used on the analysis
    TARGET: Name of the dataframe column to be used as the target, it should be numeric (discrete or continuous)!!
    Notes: 
    * plotly.express as px is a dependency so make sure to install/upgrade this library prior to execute this function
    * Please remove the unique identifier (if any) of the dataframe prior to execute this function
    * This function does accept continuous features
    * Please do not encode the categories with numbers, use "Yes", "No", "A", "B", "C" instead
    """
    import pandas as pd
    pd.set_option('mode.chained_assignment',None)
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import MinMaxScaler, StandardScaler 
    from sklearn.pipeline import make_pipeline
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    
    types_df = pd.DataFrame(data=DATAFRAME.dtypes, columns=['Type'])
    #Time Stamps or date related features are not supported
    FEATURES = types_df[types_df['Type']!='datetime64[ns]'].index.values
    DATAFRAME = DATAFRAME[FEATURES]
    #Encoding the labels
    DATAFRAME['VOL'] = 1
    #Removing null records, no impute estimator was used
    DATAFRAME.dropna(inplace=True)
    
    X = pd.get_dummies(DATAFRAME.drop(columns=[TARGET,'VOL']), prefix_sep='*')
    y = DATAFRAME[TARGET]
    
    lr = make_pipeline(MinMaxScaler(feature_range=(0, 1)),
                       Ridge(random_state=1234, max_iter=2500, fit_intercept=True, normalize=True))
    print("Fitting Ridge Regression")
    lr.fit(X, y)
    score = lr.score(X, y)
    score = score*100
    print("Process replicated with an accuracy of {:.2f}%".format(score))
    coef_df = pd.DataFrame(data=lr.steps[1][1].coef_.reshape(-1,1), index=X.columns, columns=['Inf']).sort_values(by="Inf", ascending=False)
    
    neg_inf_df = coef_df[coef_df['Inf']>0]
    neg_inf_df = coef_df[0:50]
    
    print("Top 15 Items with a Positive Influence over "+TARGET)
    print(neg_inf_df[0:15])
    
    neg_inf_df.reset_index(inplace=True)
    neg_inf_df.columns = ["Feature_Item", "Imp"]
    #Including an '*'' in order to avoid an index out of range error
    neg_inf_df['Feature_Item']= neg_inf_df['Feature_Item'].apply(lambda x: x+"*")
    neg_inf_df['Feature'] = neg_inf_df['Feature_Item'].apply(lambda x: x.split("*")[0])
    neg_inf_df['Item'] = neg_inf_df['Feature_Item'].apply(lambda x: x.split("*")[1])
        
    for feature in neg_inf_df.Feature.unique():
        if feature in DATAFRAME.columns:
            if types_df[types_df.index==feature]["Type"][0]=="object":
                ITEMS = neg_inf_df[neg_inf_df["Feature"]==feature]['Item'].unique()
                sub_df = DATAFRAME[DATAFRAME[feature].isin(ITEMS)]
                fig = px.box(data_frame =sub_df, x=feature, y=TARGET, title=feature+" influence over "+TARGET)
                fig.show()
            elif types_df[types_df.index==feature]["Type"][0]=="float64" or types_df[types_df.index==feature]["Type"][0]=="float32" or types_df[types_df.index==feature]["Type"][0]=="int32" or types_df[types_df.index==feature]["Type"][0]=="int64":
                fig = px.scatter(data_frame=DATAFRAME, x=feature, y=TARGET, title=feature+" influence over "+TARGET)
                fig.show()
                figg = px.density_heatmap(data_frame=DATAFRAME, x=feature, y=TARGET, title=feature+" influence over "+TARGET)
                figg.show()

def translate_text(FROM_LANG="en", TO_LANG="es", TEXT="Text"):

    """
    Parameters
    ----------
    FROM_LANG : ISO 639-1 Code of the source langauge. The default is "en".
    TO_LANG : ISO 639-1 Code of the target language. The default is "es".
    TEXT : String to be translated from the source to target language

    Returns
    -------
    translation : translated text
        
    Supported Languages
    -----------------------------------
    {'code': 'en', 'name': 'English'},
    {'code': 'ar', 'name': 'Arabic'},
    {'code': 'zh', 'name': 'Chinese'},
    {'code': 'fr', 'name': 'French'},
    {'code': 'de', 'name': 'German'},
    {'code': 'it', 'name': 'Italian'},
    {'code': 'pt', 'name': 'Portuguese'},
    {'code': 'ru', 'name': 'Russian'},
    {'code': 'es', 'name': 'Spanish'}
    
    Notes
    -------
    - Reference: https://github.com/uav4geo/LibreTranslate
    - Please do not include any sensitive information in the TEXT sent to the API!!!
    - This is a Python wrapper of the example shown on the official documentation for JavaScript
    """
    import requests
    import json

    URL = "https://libretranslate.com/translate"
    HEADERS = {"Content-Type": "application/json"}
    BODY =  {'q': TEXT,
		     'source': FROM_LANG,
		     'target': TO_LANG}
    
    r = requests.post(URL, data=json.dumps(BODY), headers=HEADERS)
    translation = r.json()['translatedText']
    return translation

def detect_language(TEXT="Text"):
    """
    Parameters
    ----------
    TEXT : String to get the language from

    Returns
    -------
    language: ISO 631-9 Code of the detected language
    confidence : confidence of the detected language between (0 and 1)
        
    Supported Languages
    ----------------------------------
    {'code': 'en', 'name': 'English'},
    {'code': 'ar', 'name': 'Arabic'},
    {'code': 'zh', 'name': 'Chinese'},
    {'code': 'fr', 'name': 'French'},
    {'code': 'de', 'name': 'German'},
    {'code': 'it', 'name': 'Italian'},
    {'code': 'pt', 'name': 'Portuguese'},
    {'code': 'ru', 'name': 'Russian'},
    {'code': 'es', 'name': 'Spanish'}
    
    Notes
    -------
    - Reference: https://github.com/uav4geo/LibreTranslate
    - Please do not include any sensitive information in the TEXT sent to the API!!!
    - This is a Python wrapper of the example shown on the official documentation for JavaScript
    """
    import requests
    import json

    URL = "https://libretranslate.com/detect"
    HEADERS = {"Content-Type": "application/json"}
    BODY =  {'q': TEXT}
    
    r = requests.post(URL, data=json.dumps(BODY), headers=HEADERS)
    confidence = r.json()[0]['confidence']
    language = r.json()[0]['language']
    return language, confidence

def get_text_between_delimiters(START="<", END=">", TEXT="TEXT"):
    """
    START: First delimiter character
    END: Second delimiter character
    TEXT: Text that contains the string between the two delimiters
    Reference: https://stackoverflow.com/questions/3368969/find-string-between-two-substrings
    """
    res = TEXT[TEXT.find(START)+len(START):TEXT.rfind(END)]
    return res

def circular_variable_dataframe(DATAFRAME, COL, UNIT):
    '''
    To be used in the transformation of circular variables or time related variables/features
    
    Parameters
    ----------
    DATAFRAME : Pandas dataframe
    COL : Name of the column with the time unit (Day, Month, Day of the Month, Hour, Minute)
    UNIT : Constant unit to be used in the transformation of the circular variable, for example, 12 for Months, 24 for Hours, 
    60 for Minutes, 365 for days of the year
    
    Returns
    -------
    Two additional columns with a suffix "_s" or "_c" are added to DATAFRAME
    
    Notes:
    If the resulting variables are included in a ML Pipeline is suggested to use a MaxAbsScaler in the ColumnTransformer
    '''
    import numpy as np
    CONS = (2 * np.pi) / UNIT
    DATAFRAME[COL+'_s'] = np.sin(DATAFRAME[COL]*CONS)
    DATAFRAME[COL+'_c'] = np.cos(DATAFRAME[COL]*CONS)

def circular_variable_single_value(X, UNIT):
    '''
    To be used in the transformation of circular variables or time related variables/features
    
    Parameters
    ----------
    X : single value to be transformed
    UNIT : UNIT : Constant unit to be used in the transformation of the circular variable, for example, 12 for Months, 24 for Hours, 
    60 for Minutes, 365 for days of the year
    
    Returns
    -------
    col_sin : SIN transformation of the unit and single value X
    col_cos : COS transformation of the unit and single value X
    
    Notes:
    If the resulting variables are included in a ML Pipeline is suggested to use a MaxAbsScaler in the ColumnTransformer
    '''
    import numpy as np
    CONS = (2 * np.pi) / UNIT
    col_sin = np.sin(X*CONS)
    col_cos = np.cos(X*CONS)
    return col_sin, col_cos

def sql_like(x, pattern):
    """
    x: Text in where the pattern is located
    pattern: to be used like in a SQL LIKE statement
    Example: df_notnull.loc[sql_like(df_notnull['description'], '.* math .*'), ['video_id', 'title']]
    """
    
    import numpy as np
    import re

    r = re.compile(pattern)
    vlike = np.vectorize(lambda val: bool(re.fullmatch(val)))
    return vlike(x)

def plot_prophet_prediction(ori_df, pred_df, cutoff_date, title):
    """
    Function to be used with forecasting models fitted in Prophet
    
    Parameters:
    ----------
    ori_df = original pandas dataframe used to fit the Prophet model
    pred_df = pandas dataframe with the predictions of the Prophet model, in the documentation this 
    dataframe is usually denoted as "forecast"
    cutoff_date = String in format YYYY-MM-DD that is used to show the predictions and its confidence interval
    title: Title of the plot
    
    Returns
    -------
    plot with the forecast values generated by the Prophet model
    
    Example
    -------
    plot_prophet_prediction(ori_df=wdf, pred_df=forecast, cutoff_date='2021-05-09', title='I am a Title')
    """
    pal = ["#1E74AA","#2394DB","#BA0C2F","#000000","#ED7423","#FFC627","#00955D","#FF5951"]
    FEATS = ['ds','yhat','yhat_upper','yhat_lower']
    pd.concat([ori_df, pred_df[FEATS]], axis="columns")
    a = ori_df.reset_index().drop(columns='index')
    b = pred_df[FEATS]
    pdf = pd.concat([a,b], axis='columns', ignore_index=False)
    pdf = pdf.iloc[:,1:]
    plt.figure(figsize=(12,6))
    plt.plot(pdf['ds'], pdf['y'], color=pal[3], marker='.', linestyle='solid', label='Historic', alpha=0.35)
    plt.plot(pdf[pdf['ds']>=cutoff_date]['ds'], pdf[pdf['ds']>=cutoff_date]['yhat'], color=pal[2], linestyle='dotted', marker='.', label='Forecast')
    plt.fill_between(pdf[pdf['ds']>=cutoff_date]['ds'], pdf[pdf['ds']>=cutoff_date]['yhat_upper'], pdf[pdf['ds']>=cutoff_date]['yhat_lower'],alpha=0.1)
    plt.legend(bbox_to_anchor=(1.15, 0.5), loc=0, borderaxespad=0)
    plt.title(title)