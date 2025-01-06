import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

import os
os.chdir(os.path.dirname(__file__))
print("Current working directory:", os.getcwd())

df = pd.read_excel('Retail-Supply-Chain-Sales-Dataset.xlsx')
columns_to_drop = [
    "Order Date", "Ship Date", "Order ID",
    "Customer ID", "Customer Name", "Country",
    "Postal Code", "Retail Sales People",
    "Product ID"]

df_cleaned = df.drop(columns=columns_to_drop)
print(df_cleaned.head())


label_encoder = LabelEncoder()
df_cleaned['Sub-Category (Numeric)'] = label_encoder.fit_transform(df_cleaned['Sub-Category'])
df_cleaned['State (Numeric)'] = label_encoder.fit_transform(df_cleaned['State'])

X_Produk= df_cleaned[['Sub-Category (Numeric)', 'Profit']]
X_State = df_cleaned[['State (Numeric)', 'Sales']]


# Memuat model yang sudah dilatih
with open('rfmodel_pipeline.pkl', 'rb') as file:
    rfmodel_pipeline = pickle.load(file)

with open('kmeansproduk_pipeline.pkl', 'rb') as file:
    kmeansproduk_pipeline = pickle.load(file)

with open('kmeansstate_pipeline.pkl', 'rb') as file:
    kmeansstate_pipeline = pickle.load(file)

def prediction(sales, quantity, discount, profit, ship_mode, segment, city, state, region, category, sub_category):
    input_data = pd.DataFrame({
        'Sales': [sales],
        'Quantity': [quantity],
        'Discount': [discount],
        'Profit': [profit],
        'Ship Mode': [ship_mode],
        'Segment': [segment],
        'City': [city],
        'State': [state],
        'Region': [region],
        'Category': [category],
        'Sub-Category': [sub_category],
        'Cluster_Produk': [0],
        'Sub-Category (Numeric)': [0],
        'Cluster_State': [0], 
        'State (Numeric)': [0]
    })

    pred = rfmodel_pipeline.predict(input_data)
    return pred

def prediksiReturned():
    st.title("Prediksi Barang Dikembalikan Atau Tidak")
    st.write("Made with RandomForest Classifier")
    st.write("Akurasi Model dengan Data Test = 0.918")
    st.write("Akurasi Model dengan Data Simulation = 0.909")

    sales = st.number_input("Sales", min_value=0.0, format="%.2f")
    quantity = st.number_input("Quantity", min_value=1, step=1)
    discount = st.number_input("Discount", min_value=0.0, max_value=1.0, format="%.2f")
    profit = st.number_input("Profit", format="%.2f")

    ship_mode = st.selectbox("Ship Mode", options=['First Class', 'Same Day', 'Second Class', 'Standard Class'])
    segment = st.selectbox("Segment", options=['Consumer', 'Corporate', 'Home Office'])
    city = st.selectbox("City", options=['Henderson', 'Los Angeles', 'Fort Lauderdale', 'Concord', 'Seattle', 'Fort Worth', 'Madison', 'West Jordan', 'San Francisco', 'Fremont', 'Philadelphia', 'Orem', 'Houston', 'Richardson', 'Naperville', 'Melbourne', 'Eagan', 'Westland', 'Dover', 'New Albany', 'New York City', 'Troy', 'Chicago', 'Gilbert', 'Springfield', 'Jackson', 'Memphis', 'Decatur', 'Durham', 'Columbia', 'Rochester', 'Minneapolis', 'Portland', 'Saint Paul', 'Aurora', 'Charlotte', 'Orland Park', 'Urbandale', 'Columbus', 'Bristol', 'Wilmington', 'Bloomington', 'Phoenix', 'Roseville', 'Independence', 'Pasadena', 'Newark', 'Franklin', 'Scottsdale', 'San Jose', 'Edmond', 'Carlsbad', 'San Antonio', 'Monroe', 'Fairfield', 'Grand Prairie', 'Redlands', 'Hamilton', 'Westfield', 'Akron', 'Denver', 'Dallas', 'Whittier', 'Saginaw', 'Medina', 'Dublin', 'Detroit', 'Tampa', 'Santa Clara', 'Lakeville', 'San Diego', 'Brentwood', 'Chapel Hill', 'Morristown', 'Cincinnati', 'Inglewood', 'Tamarac', 'Colorado Springs', 'Belleville', 'Taylor', 'Lakewood', 'Arlington', 'Arvada', 'Hackensack', 'Saint Petersburg', 'Long Beach', 'Hesperia', 'Murfreesboro', 'Layton', 'Austin', 'Lowell', 'Manchester', 'Harlingen', 'Tucson', 'Quincy', 'Pembroke Pines', 'Des Moines', 'Peoria', 'Las Vegas', 'Warwick', 'Miami', 'Huntington Beach', 'Richmond', 'Louisville', 'Lawrence', 'Canton', 'New Rochelle', 'Gastonia', 'Jacksonville', 'Auburn', 'Norman', 'Park Ridge', 'Amarillo', 'Lindenhurst', 'Huntsville', 'Fayetteville', 'Costa Mesa', 'Parker', 'Atlanta', 'Gladstone', 'Great Falls', 'Lakeland', 'Montgomery', 'Mesa', 'Green Bay', 'Anaheim', 'Marysville', 'Salem', 'Laredo', 'Grove City', 'Dearborn', 'Warner Robins', 'Vallejo', 'Mission Viejo', 'Rochester Hills', 'Plainfield', 'Sierra Vista', 'Vancouver', 'Cleveland', 'Tyler', 'Burlington', 'Waynesboro', 'Chester', 'Cary', 'Palm Coast', 'Mount Vernon', 'Hialeah', 'Oceanside', 'Evanston', 'Trenton', 'Cottage Grove', 'Bossier City', 'Lancaster', 'Asheville', 'Lake Elsinore', 'Omaha', 'Edmonds', 'Santa Ana', 'Milwaukee', 'Florence', 'Lorain', 'Linden', 'Salinas', 'New Brunswick', 'Garland', 'Norwich', 'Alexandria', 'Toledo', 'Farmington', 'Riverside', 'Torrance', 'Round Rock', 'Boca Raton', 'Virginia Beach', 'Murrieta', 'Olympia', 'Washington', 'Jefferson City', 'Saint Peters', 'Rockford', 'Brownsville', 'Yonkers', 'Oakland', 'Clinton', 'Encinitas', 'Roswell', 'Jonesboro', 'Antioch', 'Homestead', 'La Porte', 'Lansing', 'Cuyahoga Falls', 'Reno', 'Harrisonburg', 'Escondido', 'Royal Oak', 'Rockville', 'Coral Springs', 'Buffalo', 'Boynton Beach', 'Gulfport', 'Fresno', 'Greenville', 'Macon', 'Cedar Rapids', 'Providence', 'Pueblo', 'Deltona', 'Murray', 'Middletown', 'Freeport', 'Pico Rivera', 'Provo', 'Pleasant Grove', 'Smyrna', 'Parma', 'Mobile', 'New Bedford', 'Irving', 'Vineland', 'Glendale', 'Niagara Falls', 'Thomasville', 'Westminster', 'Coppell', 'Pomona', 'North Las Vegas', 'Allentown', 'Tempe', 'Laguna Niguel', 'Bridgeton', 'Everett', 'Watertown', 'Appleton', 'Bellevue', 'Allen', 'El Paso', 'Grapevine', 'Carrollton', 'Kent', 'Lafayette', 'Tigard', 'Skokie', 'Plano', 'Suffolk', 'Indianapolis', 'Bayonne', 'Greensboro', 'Baltimore', 'Kenosha', 'Olathe', 'Tulsa', 'Redmond', 'Raleigh', 'Muskogee', 'Meriden', 'Bowling Green', 'South Bend', 'Spokane', 'Keller', 'Port Orange', 'Medford', 'Charlottesville', 'Missoula', 'Apopka', 'Reading', 'Broomfield', 'Paterson', 'Oklahoma City', 'Chesapeake', 'Lubbock', 'Johnson City', 'San Bernardino', 'Leominster', 'Bozeman', 'Perth Amboy', 'Ontario', 'Rancho Cucamonga', 'Moorhead', 'Mesquite', 'Stockton', 'Ormond Beach', 'Sunnyvale', 'York', 'College Station', 'Saint Louis', 'Manteca', 'San Angelo', 'Salt Lake City', 'Knoxville', 'Little Rock', 'Lincoln Park', 'Marion', 'Littleton', 'Bangor', 'Southaven', 'New Castle', 'Midland', 'Sioux Falls', 'Fort Collins', 'Clarksville', 'Sacramento', 'Thousand Oaks', 'Malden', 'Holyoke', 'Albuquerque', 'Sparks', 'Coachella', 'Elmhurst', 'Passaic', 'North Charleston', 'Newport News', 'Jamestown', 'Mishawaka', 'La Quinta', 'Tallahassee', 'Nashville', 'Bellingham', 'Woodstock', 'Haltom City', 'Wheeling', 'Summerville', 'Hot Springs', 'Englewood', 'Las Cruces', 'Hoover', 'Frisco', 'Vacaville', 'Waukesha', 'Bakersfield', 'Pompano Beach', 'Corpus Christi', 'Redondo Beach', 'Orlando', 'Orange', 'Lake Charles', 'Highland Park', 'Hempstead', 'Noblesville', 'Apple Valley', 'Mount Pleasant', 'Sterling Heights', 'Eau Claire', 'Pharr', 'Billings', 'Gresham', 'Chattanooga', 'Meridian', 'Bolingbrook', 'Maple Grove', 'Woodland', 'Missouri City', 'Pearland', 'San Mateo', 'Grand Rapids', 'Visalia', 'Overland Park', 'Temecula', 'Yucaipa', 'Revere', 'Conroe', 'Tinley Park', 'Dubuque', 'Dearborn Heights', 'Santa Fe', 'Hickory', 'Carol Stream', 'Saint Cloud', 'North Miami', 'Plantation', 'Port Saint Lucie', 'Rock Hill', 'Odessa', 'West Allis', 'Chula Vista', 'Manhattan', 'Altoona', 'Thornton', 'Champaign', 'Texarkana', 'Edinburg', 'Baytown', 'Greenwood', 'Woonsocket', 'Superior', 'Bedford', 'Covington', 'Broken Arrow', 'Miramar', 'Hollywood', 'Deer Park', 'Wichita', 'Mcallen', 'Iowa City', 'Boise', 'Cranston', 'Port Arthur', 'Citrus Heights', 'The Colony', 'Daytona Beach', 'Bullhead City', 'Portage', 'Fargo', 'Elkhart', 'San Gabriel', 'Margate', 'Sandy Springs', 'Mentor', 'Lawton', 'Hampton', 'Rome', 'La Crosse', 'Lewiston', 'Hattiesburg', 'Danville', 'Logan', 'Waterbury', 'Athens', 'Avondale', 'Marietta', 'Yuma', 'Wausau', 'Pasco', 'Oak Park', 'Pensacola', 'League City', 'Gaithersburg', 'Lehi', 'Tuscaloosa', 'Moreno Valley', 'Georgetown', 'Loveland', 'Chandler', 'Helena', 'Kirkwood', 'Waco', 'Frankfort', 'Bethlehem', 'Grand Island', 'Woodbury', 'Rogers', 'Clovis', 'Jupiter', 'Santa Barbara', 'Cedar Hill', 'Norfolk', 'Draper', 'Ann Arbor', 'La Mesa', 'Pocatello', 'Holland', 'Milford', 'Buffalo Grove', 'Lake Forest', 'Redding', 'Chico', 'Utica', 'Conway', 'Cheyenne', 'Owensboro', 'Caldwell', 'Kenner', 'Nashua', 'Bartlett', 'Redwood City', 'Lebanon', 'Santa Maria', 'Des Plaines', 'Longview', 'Hendersonville', 'Waterloo', 'Cambridge', 'Palatine', 'Beverly', 'Eugene', 'Oxnard', 'Renton', 'Glenview', 'Delray Beach', 'Commerce City', 'Texas City', 'Wilson', 'Rio Rancho', 'Goldsboro', 'Montebello', 'El Cajon', 'Beaumont', 'West Palm Beach', 'Abilene', 'Normal', 'Saint Charles', 'Camarillo', 'Hillsboro', 'Burbank', 'Modesto', 'Garden City', 'Atlantic City', 'Longmont', 'Davis', 'Morgan Hill', 'Clifton', 'Sheboygan', 'East Point', 'Rapid City', 'Andover', 'Kissimmee', 'Shelton', 'Danbury', 'Sanford', 'San Marcos', 'Greeley', 'Mansfield', 'Elyria', 'Twin Falls', 'Coral Gables', 'Romeoville', 'Marlborough', 'Laurel', 'Bryan', 'Pine Bluff', 'Aberdeen', 'Hagerstown', 'East Orange', 'Arlington Heights', 'Oswego', 'Coon Rapids', 'San Clemente', 'San Luis Obispo', 'Springdale', 'Lodi', 'Mason'])
    state = st.selectbox("State", options=['Kentucky', 'California', 'Florida', 'North Carolina', 'Washington', 'Texas', 'Wisconsin', 'Utah', 'Nebraska', 'Pennsylvania', 'Illinois', 'Minnesota', 'Michigan', 'Delaware', 'Indiana', 'New York', 'Arizona', 'Virginia', 'Tennessee', 'Alabama', 'South Carolina', 'Oregon', 'Colorado', 'Iowa', 'Ohio', 'Missouri', 'Oklahoma', 'New Mexico', 'Louisiana', 'Connecticut', 'New Jersey', 'Massachusetts', 'Georgia', 'Nevada', 'Rhode Island', 'Mississippi', 'Arkansas', 'Montana', 'New Hampshire', 'Maryland', 'District of Columbia', 'Kansas', 'Vermont', 'Maine', 'South Dakota', 'Idaho', 'North Dakota', 'Wyoming', 'West Virginia'])
    region = st.selectbox("Region", options=['South', 'West', 'Central', 'East'])
    category = st.selectbox("Category", options=['Furniture', 'Office Supplies', 'Technology'])
    sub_category = st.selectbox("Sub-Category", options=['Binders', 'Paper', 'Furnishings', 'Labels', 'Art', 'Phones', 'Chairs', 'Accessories', 'Tables', 'Envelopes', 'Bookcases', 'Appliances', 'Storage', 'Supplies', 'Machines'])

    result = ""

    if st.button("Predict"):
        result = prediction(sales, quantity, discount, profit, ship_mode, segment, city, state, region, category, sub_category)
        st.success(f'Prediksi: {result}')
        
    


def plot_clusters_product():
    df_cleaned['Cluster_Produk'] = kmeansproduk_pipeline.predict(X_Produk)
    plt.figure(figsize=(8, 6))
    for cluster in range(3):
        cluster_data = df_cleaned[df_cleaned['Cluster_Produk'] == cluster]
        plt.scatter(
            cluster_data['Sub-Category (Numeric)'],
            cluster_data['Profit'], 
            label=f'Cluster {cluster}', 
            alpha=0.7
        )

    centers_produk = kmeansproduk_pipeline.named_steps['kmeansprodukmodel'].cluster_centers_
    plt.scatter(
    centers_produk[:, 0],
    centers_produk[:, 1], 
    c='red', 
    s=200, 
    alpha=0.9, 
    label='Centroids'
)

    plt.title('Segmentasi Produk Berdasarkan Profit')
    plt.xlabel('Sub-Category (Numeric)')
    plt.ylabel('Profit')
    plt.legend()
    st.pyplot(plt)

def segmentasiProduk():
    st.title("Segmentasi Produk Berdasarkan Profit")
    st.write("Made with K-Means Clustering")
    st.write("Silhoutte Score = 0.968")

    sub_category_numeric = st.number_input("Sub-Category (Numeric)", min_value=0, step=1)
    profit = st.number_input("Profit", format="%.2f")

    result = ""

    if st.button("Predict Cluster"):
        input_data = pd.DataFrame({
            'Sub-Category (Numeric)': [sub_category_numeric],
            'Profit': [profit]
        })

        # Memuat pipeline yang telah dilatih
        with open('kmeansproduk_pipeline.pkl', 'rb') as file:
            loaded_pipeline = pickle.load(file)

        result = loaded_pipeline.predict(input_data)
        st.success(f'Cluster: {result[0]}')

    if st.button("Show Clustering Plot"):
        plot_clusters_product()

def plot_clusters_state():
    df_cleaned['Cluster_State'] = kmeansstate_pipeline.predict(X_State)
    plt.figure(figsize=(8, 6))
    for cluster in range(3):
        cluster_data = df_cleaned[df_cleaned['Cluster_State'] == cluster]
        plt.scatter(
            cluster_data['State (Numeric)'],
            cluster_data['Sales'], 
            label=f'Cluster {cluster}', 
            alpha=0.7
        )

    centers_state = kmeansstate_pipeline.named_steps['kmeansstatemodel'].cluster_centers_
    plt.scatter(
        centers_state[:, 0],
        centers_state[:, 1], 
        c='red', 
        s=200, 
        alpha=0.9, 
        label='Centroids'
    )

    plt.title('K-Means Clustering State Pelanggan berdasarkan Sales')
    plt.xlabel('State (Numeric)')
    plt.ylabel('Sales')
    plt.legend()
    st.pyplot(plt)

def segmentasiState():
    st.title("Segmentasi State Berdasarkan Sales")
    st.write("Made with K-Means Clustering")
    st.write("Silhoutte Score = 0.854")

    state_numeric = st.number_input("State (Numeric)", min_value=0, step=1)
    sales = st.number_input("Sales", format="%.2f")

    result = ""

    if st.button("Predict Cluster for State"):
        input_data = pd.DataFrame({
            'State (Numeric)': [state_numeric],
            'Sales': [sales]
        })

        # Memuat pipeline yang telah dilatih
        with open('kmeansstate_pipeline.pkl', 'rb') as file:
            loaded_pipeline = pickle.load(file)

        result = loaded_pipeline.predict(input_data)
        st.success(f'Cluster: {result[0]}')

    if st.button("Show State Clustering Plot"):
        plot_clusters_state()

def main():
    st.title("Dashboard")
    st.title("Segmentasi Produk & States dan Prediksi Barang Dikembalikan Studi Kasus Toko Retail Amerika Serikat")
    st.write("Made with Love from Dimas, Gilang, Razak, and Dava. Pilih salah satu opsi di bawah ini untuk melanjutkan.")

    choice = st.selectbox("Pilih Opsi", ["Segmentasi Produk", "Segmentasi State", "Prediksi Returned"])

    if choice == "Segmentasi Produk":
        segmentasiProduk()
    elif choice == "Segmentasi State":
        segmentasiState()
    elif choice == "Prediksi Returned":
        prediksiReturned()

if __name__ == "__main__":
    main()


