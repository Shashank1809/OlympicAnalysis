import streamlit as st
import pandas as pd
import preprocessor, helper
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
from helper import medal_tally

# df = pd.read_csv('athlete_events.csv')
# region_df = pd.read_csv('noc_regions.csv')
#
# df = preprocessor.preprocess(df, region_df)

# --- Modify the data loading and add model caching ---
@st.cache_data
def load_data():
    df = pd.read_csv('athlete_events.csv')
    region_df = pd.read_csv('noc_regions.csv')
    df = preprocessor.preprocess(df, region_df)
    return df

df = load_data()
#end


# --- ADD THIS NEW SECTION ---
# Cache the trained model and feature lists
# Use @st.cache_resource for non-data objects like models
@st.cache_resource
def get_model_and_lists():
    pipeline, sports_list, region_list = helper.train_prediction_model(df)
    return pipeline, sports_list, region_list

# Load the model once
pipeline, sports_list, region_list = get_model_and_lists()
# --- END OF NEW SECTION ---


st.sidebar.title("Olympics Analysis")
user_menu = st.sidebar.radio(
    'Select option',
    ('Medal Tally','Overall Analysis','Country-wise Analytics','Athlete-wise Analysis','Country Comparison','Medal Predictor')
)

if user_menu == 'Medal Tally':
    st.sidebar.header('Medal Tally')
    years, country = helper.country_year_list(df)

    selected_year = st.sidebar.selectbox("Select Year", years)
    selected_country = st.sidebar.selectbox("Select Country", country)

    medal_tally = helper.fetch_medal_tally(df, selected_year, selected_country)

    if selected_year == 'Overall' and selected_country == 'Overall':
        st.title('Overall Tally')
        st.header("Global Medal Distribution Map")
        fig = px.choropleth(medal_tally,
                            locations='region',
                            locationmode='country names',
                            color='total',
                            hover_name='region',
                            hover_data=['Gold', 'Silver', 'Bronze'],  # Shows medal breakdown on hover
                            color_continuous_scale=px.colors.sequential.Plasma,
                            title='Overall Olympic Medals by Country')
        st.plotly_chart(fig, use_container_width=True)
    if selected_year != 'Overall' and selected_country == 'Overall':
        st.title("Medal Tally in " + str(selected_year) + " Olympics")
    if selected_year == 'Overall' and selected_country != 'Overall':
        st.title(selected_country+ " overall performance")
    if selected_country != 'Overall' and selected_country != 'Overall':
        st.title(selected_country + " performance in " + str(selected_year) + " Olympics")

    st.table(medal_tally)

if user_menu == 'Overall Analysis':
    editions = df['Year'].unique().shape[0] - 1
    cities = df['City'].unique().shape[0]
    sports =  df['Sport'].unique().shape[0]
    events =  df['Event'].unique().shape[0]
    athletes =  df['Name'].unique().shape[0]
    nations =  df['region'].unique().shape[0]

    st.title("Top Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.header('Editions')
        st.title(editions)
    with col2:
        st.header('Hosts')
        st.title(cities)
    with col3:
        st.header('Sport')
        st.title(sports)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.header('Events')
        st.title(events)
    with col2:
        st.header('Nations')
        st.title(nations)
    with col3:
        st.header('Athletes')
        st.title(athletes)

    nations_over_time = helper.data_over_time(df,'region')
    fig = px.line(nations_over_time, x="Year", y="region")
    st.title("Participating Nations Over the Years")
    st.plotly_chart(fig)

    events_over_time = helper.data_over_time(df, 'Event')
    fig = px.line(events_over_time, x="Year", y="Event")
    st.title("Events Over the Years")
    st.plotly_chart(fig)

    athletes_over_time = helper.data_over_time(df, 'Name')
    fig = px.line(athletes_over_time, x="Year", y="Name")
    st.title("Athletes Over the Years")
    st.plotly_chart(fig)

    st.title("No. of Events over time(Every Sport)")
    fig,ax = plt.subplots(figsize = (20,20))
    x = df.drop_duplicates(['Year','Sport','Event'])
    ax = sns.heatmap(x.pivot_table(index='Sport',columns='Year', values='Event', aggfunc='count').fillna(0).astype(int),annot=True)
    st.pyplot(fig)

    st.title("Most Successful Athletes")
    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0,'Overall')

    selected_sport = st.selectbox("Select Sport", sport_list, key='overall_athletes_sport')

    top_n = st.sidebar.slider('Select Number of Athletes', 5, 25, 10)

    x = helper.most_successful(df,selected_sport,top_n)
    st.table(x)

if user_menu == 'Country-wise Analytics':

    st.sidebar.title("Country wise Analytics")

    country_list = df['region'].dropna().unique().tolist()
    country_list.sort()

    selected_country = st.sidebar.selectbox("Select Country", country_list)

    country_df = helper.yearwise_medal_tally(df,selected_country)
    fig = px.line(country_df, x="Year", y="Medal")
    st.title(selected_country + " Medal Tally Over the Years")
    st.plotly_chart(fig)

    st.title(selected_country + " excels in the following Sports")
    pt = helper.country_event_heatmap(df,selected_country)
    fig, ax = plt.subplots(figsize = (20,20))
    ax = sns.heatmap(pt,annot=True)
    st.pyplot(fig)

    st.title("Top athletes of "+selected_country)
    top_n_country = st.sidebar.slider('Select Number of Athletes', 5, 15, 10, key='country_slider')
    top10_df = helper.most_successful_countrywise(df,selected_country,top_n_country)
    st.table(top10_df)

if user_menu == 'Athlete-wise Analysis':
    athlete_df = df.drop_duplicates(subset=['Name','region'])

    x1 = athlete_df['Age'].dropna()
    x2 = athlete_df[athlete_df['Medal'] == 'Gold']['Age'].dropna()
    x3 = athlete_df[athlete_df['Medal'] == 'Silver']['Age'].dropna()
    x4 = athlete_df[athlete_df['Medal'] == 'Bronze']['Age'].dropna()

    fig = ff.create_distplot([x1, x2, x3, x4], ['Overall', 'Gold Medalist', 'Silver Medalist', 'Bronze Medalist'],
                             show_hist=False, show_rug=False)
    fig.update_layout(autosize=False,width=1000,height=600)
    st.title("Distribution of Age")
    st.plotly_chart(fig)

    #NEW CODE
    st.title("Age Distribution for Gold Medalists by Sport")

    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    selected_sports = st.multiselect("Select Sports to Compare", sport_list,
                                     default=['Athletics', 'Swimming', 'Gymnastics'])
    if selected_sports:
        # Filter the data for Gold medalists in the selected sports
        gold_medalists_age_df = athlete_df[
            (athlete_df['Medal'] == 'Gold') & (athlete_df['Sport'].isin(selected_sports))]

        # Create a box plot to show the distribution
        fig = px.box(gold_medalists_age_df, x='Sport', y='Age',
                     title="Age Distribution of Gold Medalists",
                     labels={'Sport': 'Sport', 'Age': 'Age of Gold Medalist'},
                     points="all")  # 'all' shows the individual data points
        st.plotly_chart(fig)
    #NEW CODE

    x=[]
    name = []
    famous_sports = ['Basketball','Judo','Football','Tug-Of-War','Athletics','Swimming','Badminton','Sailing','Gymnastics','Art Competitions',
                     'Handball','Weightlifting','Wrestling','Water Polo','Hockey','Rowing','Fencing','Shooting','Boxing','Taekwondo','Cycling',
                     'Diving','Canoeing','Tennis','Golf','Softball','Archery','Volleyball','Synchronized Swimming','Table Tennis','Baseball',
                     'Rhythmic Gymnastics','Rugby Sevens','Beach Volleyball','Triathlon','Rugby','Polo','Ice Hockey']

    for sport in famous_sports:
        temp_df = athlete_df[athlete_df['Sport'] == sport]
        x.append(temp_df[temp_df['Medal'] == 'Gold']['Age'].dropna())
        name.append(sport)

    fig = ff.create_distplot(x,name,show_hist=False, show_rug=False)
    fig.update_layout(autosize=False, width=1000, height=600)
    st.title("Distribution of Age w.r.t. Sports(Gold Medalist)")
    st.plotly_chart(fig)

    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0, 'Overall')

    st.title('Height VS Weight')
    selected_sport = st.selectbox("Select Sport", sport_list)
    temp_df = helper.weight_v_height(df,selected_sport)
    fig,ax = plt.subplots()
    ax = sns.scatterplot(x=temp_df['Weight'], y=temp_df['Height'],hue = temp_df['Medal'],style=temp_df['Sex'])
    st.pyplot(fig)

    st.title("Men vs Women participation over the Years")
    final = helper.men_vs_women(df)
    fig = px.line(final,x="Year",y=["Male","Female"])
    st.plotly_chart(fig)

# if user_menu == 'Country Comparison':
#     st.sidebar.title('Country Comparison')
#
#     country_list = df['region'].dropna().unique().tolist()
#     country_list.sort()
#
#     st.title("Head-to-Head Country Medal Comparison")
#
#     country1 = st.sidebar.selectbox("Select Country 1", country_list)
#     country2 = st.sidebar.selectbox("Select Country 2", country_list)
#
#     if country1 and country2:
#         if country1 == country2:
#             st.warning("Please select two different countries.")
#         else:
#             comparison_df = helper.country_comparison_data(df, country1, country2)
#
#             # Ensure both countries are in the columns, even if one has 0 medals overall
#             if country1 not in comparison_df.columns:
#                 comparison_df[country1] = 0
#             if country2 not in comparison_df.columns:
#                 comparison_df[country2] = 0
#
#             fig = px.line(comparison_df, x=comparison_df.index, y=[country1, country2],
#                           title=f"Medal Tally Comparison: {country1} vs {country2}",
#                           labels={'value': 'Number of Medals', 'Year': 'Olympic Year'})
#             st.plotly_chart(fig)

if user_menu == 'Country Comparison':
    st.sidebar.title('Country Comparison')

    country_list = df['region'].dropna().unique().tolist()
    country_list.sort()

    st.title("Head-to-Head Country Medal Comparison")

    # Set default indices for a richer user experience
    # (Finds the index for 'USA' and 'China', defaults to 0 if not found)
    default_c1_index = country_list.index("USA") if "USA" in country_list else 0
    default_c2_index = country_list.index("China") if "China" in country_list else 1

    country1 = st.sidebar.selectbox("Select Country 1", country_list, index=default_c1_index)
    country2 = st.sidebar.selectbox("Select Country 2", country_list, index=default_c2_index)

    if country1 and country2:
        if country1 == country2:
            st.warning("Please select two different countries.")
        else:
            # --- PLOT 1: Original Line Chart (Medal Tally Over Time) ---
            st.header(f"Medal Tally Over Time: {country1} vs {country2}")
            comparison_df = helper.country_comparison_data(df, country1, country2)

            # Ensure both countries are columns even if one has 0 medals
            if country1 not in comparison_df.columns:
                comparison_df[country1] = 0
            if country2 not in comparison_df.columns:
                comparison_df[country2] = 0

            fig_line = px.line(comparison_df, x=comparison_df.index, y=[country1, country2],
                               title=f"Total Medals Comparison: {country1} vs {country2}",
                               labels={'value': 'Number of Medals', 'Year': 'Olympic Year'})
            st.plotly_chart(fig_line, use_container_width=True)

            # --- PLOT 2: NEW (Overall Medal Breakdown) ---
            st.header(f"Overall Medal Breakdown: {country1} vs {country2}")
            breakdown_df = helper.country_medal_breakdown(df, country1, country2)
            fig_breakdown = px.bar(breakdown_df, x='region', y='Count', color='Medal',
                                   barmode='group',
                                   title=f"Total Medal Count (Gold, Silver, Bronze)",
                                   labels={'region': 'Country', 'Count': 'Total Medals'},
                                   color_discrete_map={'Gold': 'gold', 'Silver': 'silver', 'Bronze': '#cd7f32'})
            st.plotly_chart(fig_breakdown, use_container_width=True)

            # --- PLOT 3: NEW (Medals by Gender) ---
            st.header(f"Medal Winners by Gender: {country1} vs {country2}")
            gender_df = helper.country_gender_medals(df, country1, country2)
            fig_gender = px.bar(gender_df, x='region', y='Medal', color='Sex',
                                barmode='group',
                                title=f"Medals Won by Men vs. Women",
                                labels={'region': 'Country', 'Medal': 'Number of Medals'})
            st.plotly_chart(fig_gender, use_container_width=True)

            # --- PLOT 4: NEW (Top Sports Comparison) ---
            st.header(f"Top Sports Comparison: {country1} vs {country2}")
            st.info("This chart shows the medal count for any sport where *either* country ranks in their own top 10.")
            sports_df = helper.country_top_sports(df, country1, country2)

            if sports_df.empty:
                st.warning(f"No common or top sports data found for {country1} and {country2}.")
            else:
                fig_sports = px.bar(sports_df, y='Sport', x='Medal', color='region',
                                    barmode='group',
                                    orientation='h',
                                    title=f"Medal Comparison in Top Sports",
                                    labels={'Medal': 'Number of Medals', 'Sport': 'Sport'},
                                    height=len(sports_df['Sport'].unique()) * 40)  # Dynamically set height
                st.plotly_chart(fig_sports, use_container_width=True)



# --- ADD THIS ENTIRE NEW BLOCK at the end of app.py ---
if user_menu == 'Medal Predictor':
    st.title("Medal Win Predictor 🥇")
    st.markdown("""
    This tool uses a Logistic Regression model to predict the probability of an athlete winning *any* medal
    (Gold, Silver, or Bronze) based on their attributes and sport.

    The model was trained on all athlete-event entries from 1896-2016.
    """)

    # Create columns for a cleaner layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Athlete Attributes")
        selected_sex = st.selectbox("Select Sex", ('M', 'F'))
        selected_age = st.number_input("Enter Age", 10, 97, 25)  # min, max, default
        selected_height = st.number_input("Enter Height (cm)", 120, 230, 175)
        selected_weight = st.number_input("Enter Weight (kg)", 25, 220, 70)

    with col2:
        st.subheader("Event & Country")
        # Use the lists generated by the helper function
        selected_sport = st.selectbox("Select Sport", sports_list)
        selected_region = st.selectbox("Select Country (region)", region_list)

    # Button to trigger the prediction
    if st.button("Predict Medal Probability", use_container_width=True, type="primary"):
        # Create a single-row DataFrame from the user's inputs
        # The column names MUST match those used during training
        input_data = pd.DataFrame({
            'Age': [selected_age],
            'Height': [selected_height],
            'Weight': [selected_weight],
            'Sex': [selected_sex],
            'Sport': [selected_sport],
            'region': [selected_region]
        })

        # Use the pipeline to predict probabilities
        # pipeline.predict_proba returns a 2D array: [[prob_class_0, prob_class_1]]
        probability = pipeline.predict_proba(input_data)[0]

        # probability[0] is the probability of 'No Medal' (class 0)
        # probability[1] is the probability of 'Medal Won' (class 1)
        medal_prob = probability[1]

        st.subheader("Prediction Result")

        # Display the result using st.metric for a nice visual
        st.metric(label="Probability of Winning a Medal", value=f"{medal_prob * 100:.2f}%")

        # Give a qualitative interpretation
        if medal_prob > 0.6:
            st.success("This athlete has a very high probability of winning a medal!")
        elif medal_prob > 0.35:
            st.warning("This athlete has an outside chance of winning a medal.")
        else:
            st.error("This athlete is unlikely to win a medal based on historical data.")

        # Add an expander to show more details
        with st.expander("Show Model Details"):
            st.write(f"Probability of **No Medal**: `{probability[0] * 100:.2f}%`")
            st.write(f"Probability of **Medal Won**: `{probability[1] * 100:.2f}%`")
            st.write("""
            **Model:** `LogisticRegression`

            **Features Used:** `Age`, `Height`, `Weight`, `Sex`, `Sport`, `region`

            **Note:** This model is trained on a historically imbalanced dataset (most entries are non-medal winners).
            We use `class_weight='balanced'` to compensate for this.
            """)