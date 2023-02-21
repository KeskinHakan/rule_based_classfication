# Kural Tabanlı Sınıflandırma ile Potansiyel Müşteri Getirisi Hesaplama

# Mission 1

# Q1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Rule Based Classification of Customer's Data", page_icon="🖖")
# st.title("Rule Based Classification of Customer's Data")
st.markdown("<h2 style='text-align: center; color: grey;'>Rule Based Classification of Customer's Data </h2>", unsafe_allow_html=True)
"""
This application is developed to find out Segment and Price of the New Customer' informations.

To use the app you should choose at the following steps below:

    1- Country of the new customer,
    2- Operating system of the new customer,
    3- Gender 
    4- Age

After these choices this app will give the Segment and Price for the new user.

"""
st.subheader("Analysis of the variables")
pd.set_option("display.width", 500)
pd.set_option("display.max_columns", None)

main_file_name = 'persona.csv' # change it to the name of your excel file

df = pd.read_csv(main_file_name)
df.info()
df.head()
#df.describe().T
df.isnull().sum()
#df.columns

### Check df fonksiyonu ile hızlı bir bakış atabiliriz
def check_df(dataframe, head=5):
    print("########## Shape#############")
    print(dataframe.shape)
    print(dataframe.dtypes)
    print(dataframe.head(head))
    print(dataframe.tail(head))
    print(dataframe.isnull().sum())
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

# Q2

def ratio_calculater(dataframe,col):
    print(pd.DataFrame({col: dataframe[col].value_counts(),
                        "Ratio": 100 * dataframe[col].value_counts()/len(dataframe)}))
    print("#################################")
    print("Total number of unique source is: ",dataframe[col].nunique())

for col in df.columns:
    ratio_calculater(df, col)

# Q3

print("Total number of unique Price is: ", df["PRICE"].nunique())

# Q4

ratio_calculater(df, "PRICE")

# Q5

ratio_calculater(df, "COUNTRY")

# Q6
def total_earning(dataframe, col_name):
    print(dataframe.groupby(col_name).agg({"PRICE": "sum"}))

total_earning(df,"COUNTRY")

# Q7

def count_sales(dataframe, col_name):
    print(dataframe.groupby(col_name).agg({"PRICE": "count"}))

count_sales(df,"SOURCE")

# Q8

def mean_prices(dataframe, col_name):
    print(dataframe.groupby(col_name).agg({"PRICE": "mean"}))

mean_prices(df,"COUNTRY")

# Q9

mean_prices(df,"SOURCE")

# Q10

def fragility_curve(dataframe, col_name1, col_name2):
    print(dataframe.groupby([col_name1, col_name2]).agg({"PRICE": "mean"}))

fragility_curve(df,"COUNTRY", "SOURCE")

# Mission 2 & 3

agg_df = pd.DataFrame(df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"})).sort_values("PRICE", ascending= False)

# Mission 4
agg_df = agg_df.reset_index()

# Mission 5


def grab_col_names(df, cat_th=10, car_th=20):

    cat_col = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and str(df[col].dtypes) in ["float64", "int64"]]
    cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

    cat_col = cat_col + num_but_cat

    cat_col = [col for col in cat_col if col not in cat_but_car]

    num_col = [col for col in df.columns if col not in cat_col]

    print(f"Observations: {df.shape[0]}")
    print(f"Variables: {df.shape[1]}")
    print(f"cat_cols: {len(cat_col)}")
    print(f"num_cols: {len(num_col)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat {len(num_but_cat)}")

    return cat_col, num_col, cat_but_car

cat_col, num_col, cat_but_car = grab_col_names(df)


def cat_summary(dataframe, col_name, plot = False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#######################")
    if plot:
        fig = plt.figure(figsize=(10, 4))
        sns.countplot(x=dataframe[col_name], data = dataframe)
        plt.show(block=True)
        st.caption(col_name, unsafe_allow_html=False)
        st.pyplot(fig)

for col in cat_col:
    cat_summary(df, col, plot = True)

def num_summary(dataframe, num_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    print(dataframe[num_col].describe(quantiles).T)
    if plot == True:
        fig = plt.figure(figsize=(10, 4))
        dataframe[num_col].hist()
        plt.xlabel(num_col)
        plt.title(num_col)
        plt.show(block=True)
        st.caption(num_col, unsafe_allow_html=False)
        st.pyplot(fig)

for col in num_col:
    num_summary(df, col, plot=True)

agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], [0, 18, 23, 30, 40, 70], labels=['0_18', '19_23', '24_30','31_40', '41_70'])

agg_df["AGE_CAT"] = agg_df["AGE_CAT"].astype("O")

customers_level_based = [agg_df["COUNTRY"][x] + "_" + agg_df["SOURCE"][x] + "_" + agg_df["SEX"][x] + "_" + agg_df["AGE_CAT"][x]
                         for x in range(len(agg_df))]

customers_level_based = pd.DataFrame(customers_level_based)
customers_level_based.columns = ["customers_level_based"]
customers_level_based["customers_level_based"] = customers_level_based["customers_level_based"].str.upper()
customers_level_based["PRICE"] = agg_df["PRICE"] # Mean Price

customers_level_based.groupby(["customers_level_based"]).agg({"PRICE": "mean"})

agg_df = customers_level_based.groupby(["customers_level_based"]).agg({"PRICE": "mean"})
agg_df2 = agg_df.reset_index()

segment = pd.qcut(agg_df2["PRICE"], 4, labels = ["D","C","B","A"])

agg_df2["SEGMENT"] = segment

agg_df3 = agg_df2.groupby(["SEGMENT"]).agg({"PRICE": ["mean", "max", "sum"]})

# Mission 8

st.sidebar.header("New customer information:")
country = st.sidebar.selectbox("Country of new user: ", {"Bra", "Tur", "Usa", "Can", "Deu","Fra"})
phone_type = st.sidebar.selectbox("Operating System of new user: ", {"Android", "IOS"})
gender = st.sidebar.selectbox("Gender of new user: ", {"Female", "Male"})
age = st.sidebar.number_input("Age of new user",value=18, step=1)

input_dataframe = [[country, phone_type, gender, age]]
input_dataframe = pd.DataFrame(input_dataframe, columns = ["COUNTRY", "SOURCE", "SEX", "AGE"])

input_dataframe["AGE_CAT"] = pd.cut(input_dataframe["AGE"], [0, 18, 23, 30, 40, 70], labels=['0_18', '19_23', '24_30','31_40', '41_70'])
age_cat = input_dataframe["AGE_CAT"][0]

st.sidebar.markdown(("New Customer Definition:"))
new_user_3 = (country + "_"+ phone_type +"_"+ gender+ "_"+age_cat).upper()
price = agg_df2[agg_df2["customers_level_based"] == new_user_3].reset_index(drop=True)
st.sidebar.markdown(new_user_3)

st.info("Customer ID: " + str(new_user_3))
st.info("Mean price for the new customer: " + str(format(price["PRICE"][0], ".2f")) + "$")
st.success("Segment for the user's choice: " + str(price["SEGMENT"][0]))

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

