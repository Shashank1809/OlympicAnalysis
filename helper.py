import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def fetch_medal_tally(df, year, country):
    medal_df = df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'])

    flag = 0
    if year == 'Overall' and country == 'Overall':
        temp_df = medal_df
    if year == 'Overall' and country != 'Overall':
        flag = 1
        temp_df = medal_df[medal_df['region'] == country]
    if year != 'Overall' and country == 'Overall':
        temp_df = medal_df[medal_df['Year'] == int(year)]
    if year != 'Overall' and country != 'Overall':
        temp_df = medal_df[(medal_df['Year'] == year) & (medal_df['region'] == country)]

    if flag == 1:
        x = temp_df.groupby('Year').sum()[['Gold', 'Silver', 'Bronze']].sort_values('Year').reset_index()
    else:
        x = temp_df.groupby('region').sum()[['Gold', 'Silver', 'Bronze']].sort_values('Gold', ascending=False).reset_index()

    x['total'] = x['Gold'] + x['Silver'] + x['Bronze']

    return x

def medal_tally(df):
    medal_tally =  df.drop_duplicates(subset=['Team','NOC','Games','Year','City','Sport','Event','Medal'])
    medal_tally = medal_tally.groupby('region').sum()[['Gold','Silver','Bronze']].sort_values(by='Gold', ascending=False).reset_index()
    medal_tally['total'] = medal_tally['Gold'] + medal_tally['Silver'] + medal_tally['Bronze']

    return medal_tally

def country_year_list(df):
    years = df['Year'].unique().tolist()
    years.sort()
    years.insert(0, 'Overall')

    country = np.unique(df['region'].dropna().values)
    country = country.tolist()
    country.sort()
    country.insert(0, 'Overall')

    return years, country

def data_over_time(df,col):
    nations_over_time = df.drop_duplicates(['Year',col])['Year'].value_counts().reset_index().sort_values('Year')
    nations_over_time.rename(columns={'Year':'Year','count':col}, inplace=True)
    return nations_over_time

def most_successful(df, sport,top_n=15):
    temp_df = df.dropna(subset = ['Medal'])

    if sport != 'Overall':
        temp_df = temp_df[temp_df['Sport'] == sport]

    x = temp_df['Name'].value_counts().reset_index().head(top_n).merge(df,left_on='Name',right_on='Name',how='left')[['Name','count','Sport','region']].drop_duplicates('Name')
    x.rename(columns={'count':'Medals','region':'Country'},inplace = True)
    return x

def yearwise_medal_tally(df,country,sport='Overall'):
    temp_df = df.dropna(subset = ['Medal'])
    temp_df.drop_duplicates(subset=['Team','NOC','Games','Year','City','Sport','Event','Medal'],inplace=True)

    new_df = temp_df[temp_df['region'] == country]
    # --- ADD THIS LOGIC ---
    if sport != 'Overall':
        new_df = new_df[new_df['Sport'] == sport]
    # --- END OF NEW LOGIC ---
    final_df = new_df.groupby('Year').count()['Medal'].reset_index()

    return final_df

def country_event_heatmap(df,country):
    temp_df = df.dropna(subset = ['Medal'])
    temp_df.drop_duplicates(subset=['Team','NOC','Games','Year','City','Sport','Event','Medal'],inplace=True)

    new_df = temp_df[temp_df['region'] == country]

    pt = new_df.pivot_table(index='Sport', columns='Year', values='Medal', aggfunc='count').fillna(0)
    return pt

def most_successful_countrywise(df, country,sport='Overall',top_n=10):
    temp_df = df.dropna(subset = ['Medal'])

    temp_df = temp_df[temp_df['region'] == country]

    # --- ADD THIS LOGIC ---
    if sport != 'Overall':
        temp_df = temp_df[temp_df['Sport'] == sport]
    # --- END OF NEW LOGIC ---

    x = temp_df['Name'].value_counts().reset_index().head(top_n).merge(df,left_on='Name',right_on='Name',how='left')[['Name','count','Sport']].drop_duplicates('Name')
    x.rename(columns={'count':'Medals'},inplace = True)
    return x

def country_sport_event_heatmap(df, country, sport):
    temp_df = df.dropna(subset=['Medal'])
    temp_df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'], inplace=True)

    # Filter for country and the selected sport
    new_df = temp_df[(temp_df['region'] == country) & (temp_df['Sport'] == sport)]

    # Pivot on Event (rows) and Year (columns)
    pt = new_df.pivot_table(index='Event', columns='Year', values='Medal', aggfunc='count').fillna(0).astype(int)
    return pt

def weight_v_height(df,sport):
    athlete_df = df.drop_duplicates(subset = ['Name','region'])
    athlete_df['Medal'].fillna('No Medal',inplace=True)
    if sport != 'Overall':
        temp_df = athlete_df[athlete_df['Sport'] == sport]
        return temp_df
    else:
        return athlete_df

def men_vs_women(df, sport='Overall'):
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])

    # Filter by sport if it's not 'Overall'
    if sport != 'Overall':
        athlete_df = athlete_df[athlete_df['Sport'] == sport]
    # --- END OF NEW LOGIC ---

    # Check if df is empty after filtering
    if athlete_df.empty:
        return pd.DataFrame(columns=['Year', 'Male', 'Female'])

    men = athlete_df[athlete_df['Sex'] == 'M'].groupby('Year').count()['Name'].reset_index()
    women = athlete_df[athlete_df['Sex'] == 'F'].groupby('Year').count()['Name'].reset_index()

    # Use an outer merge to handle years where only one gender participated
    final = men.merge(women,on='Year',how='outer')
    final.rename(columns={'Name_x':'Male','Name_y':'Female'},inplace=True)


    final.fillna(0, inplace=True)
    final['Male'] = final['Male'].astype(int)
    final['Female'] = final['Female'].astype(int)

    final = final.sort_values(by='Year')

    return final


def country_comparison_data(df, country1, country2):
    temp_df = df[df['region'].isin([country1, country2])]

    comparison_df = temp_df.groupby(['Year', 'region'])['Medal'].count().reset_index()

    comparison_df = comparison_df.pivot(index='Year', columns='region', values='Medal').fillna(0).astype(int)

    return comparison_df


def country_medal_breakdown(df, country1, country2):

    #Prepares data for the G/S/B medal breakdown grouped bar chart.
    # We use the main df which has one-hot encoded medals
    temp_df = df[df['region'].isin([country1, country2])]

    # Group by region and sum the medal columns
    breakdown = temp_df.groupby('region')[['Gold', 'Silver', 'Bronze']].sum().reset_index()

    # Melt the DataFrame to make it compatible with Plotly express
    melted_df = breakdown.melt(id_vars='region', value_vars=['Gold', 'Silver', 'Bronze'],
                               var_name='Medal', value_name='Count')
    return melted_df


def country_gender_medals(df, country1, country2):

    #Prepares data for the gender-based medal comparison bar chart.
    temp_df = df[df['region'].isin([country1, country2])]

    # Filter for rows that actually have a medal
    temp_df = temp_df.dropna(subset=['Medal'])

    # Group by region and sex, then count the number of medals
    gender_df = temp_df.groupby(['region', 'Sex'])['Medal'].count().reset_index()
    return gender_df


def country_top_sports(df, country1, country2):

    #Prepares data for the top sports comparison horizontal bar chart.
    temp_df = df[df['region'].isin([country1, country2])]
    temp_df = temp_df.dropna(subset=['Medal'])

    # Group by region and Sport, counting medals
    sports_df = temp_df.groupby(['region', 'Sport'])['Medal'].count().reset_index()

    # Find the top 10 sports for Country 1
    top_10_c1 = sports_df[sports_df['region'] == country1].nlargest(10, 'Medal')['Sport'].tolist()
    # Find the top 10 sports for Country 2
    top_10_c2 = sports_df[sports_df['region'] == country2].nlargest(10, 'Medal')['Sport'].tolist()

    # Create a union of both lists to get all relevant sports
    top_sports_union = list(set(top_10_c1) | set(top_10_c2))

    # Filter the grouped data to only include these top sports
    final_df = sports_df[sports_df['Sport'].isin(top_sports_union)]

    return final_df


def train_prediction_model(df):
    """
    Trains a model to predict medal success.
    Returns the trained pipeline and lists of unique sports and regions.
    """
    # 1. Feature Engineering: Create the target variable
    model_df = df.copy()
    # Create a binary target: 1 if athlete won a medal, 0 if not
    model_df['Medal_Won'] = model_df['Medal'].apply(lambda x: 0 if pd.isna(x) else 1)

    # 2. Data Cleaning: Drop rows where key features are missing
    model_df.dropna(subset=['Age', 'Height', 'Weight', 'region', 'Sport', 'Sex'], inplace=True)

    # Get unique lists for the UI dropdowns
    sports_list = model_df['Sport'].unique().tolist()
    sports_list.sort()
    region_list = model_df['region'].unique().tolist()
    region_list.sort()

    # 3. Define Features (X) and Target (y)
    # We select the features that are good predictors
    features = ['Age', 'Height', 'Weight', 'Sex', 'Sport', 'region']
    X = model_df[features]
    y = model_df['Medal_Won']

    # 4. Create Preprocessing Pipeline
    # Define which features are numeric and which are categorical
    numeric_features = ['Age', 'Height', 'Weight']
    categorical_features = ['Sex', 'Sport', 'region']

    # Create a transformer for numeric features (scaling)
    numeric_transformer = StandardScaler()

    # Create a transformer for categorical features (one-hot encoding)
    # handle_unknown='ignore' prevents errors if new/unseen data is passed
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Use ColumnTransformer to apply different transformers to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 5. Create and Train the Final Model Pipeline
    # We chain the preprocessor and the classifier (Logistic Regression)
    # class_weight='balanced' is CRITICAL for this imbalanced dataset (way more non-winners than winners)
    # max_iter=1000 ensures the model converges

    # model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)
                               ])

    # Train the model on the entire available dataset
    pipeline.fit(X, y)

    return pipeline, sports_list, region_list