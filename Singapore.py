import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
from streamlit_option_menu import option_menu
import pickle
from PIL import Image

       

def town_mapping(town_map):
    if town_map == 'ANG MO KIO':
        town = int(0)
    elif town_map == 'BEDOK':
        town = int(1)
    elif town_map == 'BISHAN':
        town = int(2)
    elif town_map == 'BUKIT BATOK':
        town = int(3)
    elif town_map == 'BUKIT MERAH':
        town = int(4)
    elif town_map == 'BUKIT PANJANG':
        town = int(5)
    elif town_map == 'BUKIT TIMAH':
        town = int(6)
    elif town_map == 'CENTRAL AREA':
        town = int(7)
    elif town_map == 'CHOA CHU KANG':
        town = int(8)
    elif town_map == 'CLEMENTI':
        town = int(9)
    elif town_map == 'GEYLANG':
        town = int(10)
    elif town_map == 'HOUGANG':
        town = int(11)
    elif town_map == 'JURONG EAST':
        town = int(12)
    elif town_map == 'JURONG WEST':
        town = int(13)
    elif town_map == 'KALLANG/WHAMPOA':
        town = int(14)
    elif town_map == 'MARINE PARADE':
        town = int(15)
    elif town_map == 'PASIR RIS':
        town = int(16)
    elif town_map == 'PUNGGOL':
        town = int(17)
    elif town_map == 'QUEENSTOWN':
        town = int(18)
    elif town_map == 'SEMBAWANG':
        town = int(19)
    elif town_map == 'SENGKANG':
        town = int(20)
    elif town_map == 'SERANGOON':
        town == int(21)
    elif town_map == 'TAMPINES':
        town = int(22)
    elif town_map == 'TOA PAYOH':
        town = int(23)
    elif town_map == 'WOODLANDS':
        town = int(24)
    elif town_map == 'YISHUN':
        town = int(25)
    
    return town
      

def flat_type_mapping(flat_type):

    if flat_type == '1 ROOM':
        flt_type = int(0)
    elif flat_type == '2 ROOM':
        flt_type = int(1)
    elif flat_type == '3 ROOM':
        flt_type = int(2)
    elif flat_type == '4 ROOM':
        flt_type = int(3)
    elif flat_type == '5 ROOM':
        flt_type = int(4)
    elif flat_type == 'EXECUTIVE':
        flt_type = int(5)
    elif flat_type == 'MULTI-GENERATION':
        flt_type = int(6)

    
    return flt_type

def flat_model_mapping(flt_model):
 

    if flt_model == '2-room':
        flat_model = int(0)
    elif flt_model == '3Gen':
        flat_model = int(1)
    elif flt_model == 'Adjoined flat':
        flat_model = int(2)
    elif flt_model == 'Apartment':
        flat_model = int(3)
    elif flt_model == 'DBSS':
        flat_model = int(4)
    elif flt_model == 'Improved':
        flat_model = int(5)
    elif flt_model == 'Improved-Maisonette':
        flat_model = int(6)
    elif flt_model == 'Maisonette':
        flat_model = int(7)
    elif flt_model == 'Model A':
        flat_model = int(8)
    elif flt_model == 'Model A-Maisonette':
        flat_model = int(9)
    elif flt_model == 'Model A2':
        flat_model = int(10)
    elif flt_model == 'Multi Generation':
        flat_model = int(11)
    elif flt_model == 'New Generation':
        flat_model = int(12)
    elif flt_model == 'Premium Apartment':
        flat_model = int(13)
    elif flt_model == 'Premium Apartment Loft':
        flat_model = int(14)
    elif flt_model == 'Premium Maisonette':
        flat_model = int(15)
    elif flt_model == 'Simplified':
        flat_model = int(16)
    elif flt_model == 'Standard':
        flat_model = int(17)
    elif flt_model == 'Terrace':
        flat_model = int(18)
    elif flt_model == 'Type S1':
        flat_model = int(19)
    elif flt_model == 'Type S2':
        flat_model = int(20)

    return flat_model 

def predict_price(year, town, flat_type, flr_area_sqm, flat_modl, storey_start, storey_end, re_les_year,
                  re_les_month, les_com_dt):
    
    year_s = int(year)
    town_s = town_mapping(town)
    flat_type_s = flat_type_mapping(flat_type)
    flr_area_sqm_s = int(flr_area_sqm)
    flat_model_s = flat_model_mapping(flat_modl)
    stry_start_s = np.log(int(storey_start))
    stry_end_s = np.log(int(storey_end))
    re_les_year_s = int(re_les_year)
    re_les_month_s = int(re_les_month)
    les_com_dt_s = int(les_com_dt)

    with open(r"C:\Users\ramya\Singapore_Flat_Price.pkl", "rb") as f:
        regg_model = pickle.load(f)

    user_data = np.array([[year_s, town_s, flat_type_s, flr_area_sqm_s, flat_model_s, stry_start_s, stry_end_s,
                          re_les_year_s, re_les_month_s, les_com_dt_s]])
    
    y_pred = regg_model.predict(user_data)
    price= np.exp(y_pred[0])

    return round(price)

# Streamlit

st.set_page_config(layout="wide")

st.title(":violet[SINGAPORE RESALE FLAT PRICES PREDICTING]")
st.write("")

with st.sidebar:
    select = option_menu("Main Menu",["Home","Price Prediction","Conclusion"], 
                         icons=['house','list','star'], menu_icon="cast", default_index=1)

if select == "Home":

    col1,col2 = st.columns(2)
    with col2 :
        st.write("")
        st.write("")
        img = Image.open(r"C:\Users\ramya\Singapore4.jpeg")
        st.image(img, width=450)

    with col1:
        st.write("Welcome to the Singapore Resale Flat Prices Prediction project! This project is dedicated to analyzing and predicting the resale prices of HDB flats in Singapore, providing valuable insights for homeowners, buyers, real estate agents, and policymakers.")
        st.header(":violet[Project Overview:]")
        st.write("Singapore's public housing system, managed by the Housing and Development Board (HDB), is a cornerstone of the nation's real estate market. With a significant portion of the population residing in HDB flats, understanding the factors that influence resale prices is essential for making informed decisions. This project aims to leverage data analysis and machine learning techniques to predict the resale prices of these flats, offering a comprehensive and interactive tool for stakeholders.")
        st.write("")

    st.header("Key Features")
    st.write("")

    st.subheader(":violet[1.Data Wrangling:]")
    st.write("Comprehensive data cleaning and preprocessing to ensure accurate and reliable analysis.")
    st.write("")

    st.subheader(":violet[2.Exploratory Data Analysis (EDA):]")
    st.write("In-depth analysis of historical resale flat prices, identifying key trends and patterns.")
    st.write("")

    st.subheader(":violet[3.Model Building:]")
    st.write("Development of predictive models using advanced machine learning algorithms to forecast future resale prices.")
    st.write("")
    
    st.subheader(":violet[4.Interactive Interface: ]")
    st.write("A user-friendly Streamlit application that allows users to explore the data, visualize trends, and make predictions.")
    st.write("")

    st.header("Outcomes")
    st.subheader(":violet[1.Accurate Predictions]")
    st.write("A model capable of predicting resale prices with high accuracy, aiding buyers and sellers in making informed decisions.")
    st.subheader(":violet[2.Insightful Analysis]")
    st.write("Detailed insights into the factors influencing HDB flat prices, helping stakeholders understand market dynamics.")
    st.subheader(":violet[3.Interactive Tool]")
    st.write("A web-based application allowing users to interact with the model and obtain real-time price predictions.")

elif select == "Price Prediction":
    
    col1, col2 = st.columns(2)
    with col1:
        
        year = st.selectbox(":blue[***Select The Year:***]",["2015","2016","2017","2018","2019","2020","2021","2022","2023","2024"])

        town = st.selectbox(":blue[***Select The Month:***]", ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
                                                'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG',
                                                'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
                                                'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL',
                                                'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES',
                                                'TOA PAYOH', 'WOODLANDS', 'YISHUN'])
        
        flat_type = st.selectbox(":blue[***Select The Flat Type:***]",['3 ROOM', '4 ROOM', '5 ROOM', '2 ROOM', 'EXECUTIVE', '1 ROOM',
                                                        'MULTI-GENERATION'])
        
        flr_area_sqm = st.number_input(":blue[***Enter The Value Of Floor Area sqm : [Min: 31 / Max: 280]***]")

        flat_model = st.selectbox(":blue[***Select The Flat Model:***]", ['Improved', 'New Generation', 'Model A', 'Standard', 'Simplified',
                                                            'Premium Apartment', 'Maisonette', 'Apartment', 'Model A2',
                                                            'Type S1', 'Type S2', 'Adjoined flat', 'Terrace', 'DBSS',
                                                            'Model A-Maisonette', 'Premium Maisonette', 'Multi Generation',
                                                            'Premium Apartment Loft', 'Improved-Maisonette', '2-room', '3Gen'])

    with col2:

        stry_start = st.number_input(":blue[***Enter The Value For Storey Start:***]")
        stry_end = st.number_input(":blue[***Enter The Value For Storey End:***]")
        re_les_year = st.number_input(":blue[***Enter The Value For Remaining Lease Year: [Min: 42 / Max: 97]***]")
        re_les_mon = st.number_input(":blue[***Enter The Value For Remaing Lease Month : [Min: 0 / Max: 11]***]")
        les_com_dt = st.selectbox(":blue[***Select The Lease Commence Date :***]", [str(i) for i in range(1966, 2023)])

    button = st.button(":blue[**Predict The Price**]", use_container_width= True)

    if button:

        pred_price = predict_price(year, town, flat_type, flr_area_sqm, flat_model,
                                   stry_start, stry_end, re_les_year, re_les_mon, les_com_dt)
        
        st.write("## :red[**The Predicted Price is :**]", pred_price)

elif select == "Conclusion":

    st.header(":red[Conclusion]")
    st.write("***In this project, we successfully developed a machine learning model to predict the resale prices of HDB flats in Singapore. The key steps involved in this project were:***")

    st.header(":red[Data Collection and Preprocessing:]")
    st.write("***We collected a comprehensive dataset of HDB resale transactions.***")
    st.write("***The data was cleaned and preprocessed to ensure it was suitable for modeling. This included handling missing values, encoding categorical variables, and normalizing numerical features.***")
    st.header(":red[Feature Engineering:]")
    st.write("***We created additional features to capture the relevant aspects of each flat, such as the flat type, flat model, floor area, remaining lease, and location (town).***")
    st.write("***Transformation of features such as the logarithmic transformation of storey levels to handle skewness and improve model performance.***")
    st.header(":red[Model Training:]")
    st.write("***We trained a RandomForestRegressor model, which was selected due to its ability to handle complex interactions between features and its robustness against overfitting.***")
    st.write("***The model was trained on historical data to learn the patterns and relationships between the features and the resale prices.***")
    st.header(":red[Model Evaluation:]")
    st.write("***The model was evaluated using appropriate metrics to ensure its accuracy and reliability in predicting resale prices.***")
    st.write("***Cross-validation and parameter tuning were performed to optimize the model's performance.***")
    st.header(":red[Deployment:]")
    st.write("***The model was integrated into a Streamlit application, providing an interactive interface for users to input the characteristics of an HDB flat and receive a predicted resale price.***")
    st.write("***The application was designed to be user-friendly and informative, making it accessible to a wide range of users.***")
    st.header(":red[Error Handling and Debugging:]")
    st.write("***Throughout the development process, we implemented various error handling and debugging techniques to ensure the robustness of the application.***")
    st.write("***Issues related to input data, such as handling NaN or infinite values, were addressed to prevent runtime errors.***")


    

    





