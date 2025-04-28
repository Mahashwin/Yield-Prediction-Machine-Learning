import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
import tkinter as tk
from tkinter import ttk, messagebox
import os

warnings.filterwarnings('ignore')


class CropProductionPredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.data = None
        self.target = None
        self.categorical_features = ['State_Name', 'District_Name', 'Season', 'Crop']
        self.numerical_features = ['Area']
        self.model_type = None

    def load_data(self, filepath):
        self.data = pd.read_csv(filepath)
        print(f"Data loaded successfully with shape: {self.data.shape}")
        return self.data.head()

    def preprocess_data(self):
        if self.data.isnull().sum().sum() > 0:
            print("Handling missing values...")
            self.data = self.data.dropna()

        X = self.data[self.categorical_features + self.numerical_features]
        self.target = self.data['Production']

        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        numerical_transformer = StandardScaler()

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, self.categorical_features),
                ('num', numerical_transformer, self.numerical_features)
            ])

        self.preprocessor.fit(X)
        return X, self.target

    def train_model(self, model_type='knn', n_neighbors=5):
        X, y = self.preprocess_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model_type = model_type

        if model_type.lower() == 'knn':
            self.model = Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', KNeighborsRegressor(n_neighbors=n_neighbors))
            ])
        else:
            self.model = Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', LinearRegression())
            ])

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Model trained using {model_type.upper()}")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R² Score: {r2:.2f}")

        return self.model

    def predict_production(self, state_name, district_name, season, crop, area):
        if self.model is None:
            raise Exception("Model not trained. Please train the model first.")

        input_data = pd.DataFrame({
            'State_Name': [state_name],
            'District_Name': [district_name],
            'Season': [season],
            'Crop': [crop],
            'Area': [float(area)]
        })

        prediction = self.model.predict(input_data)[0]
        return prediction

    def save_model(self, filename='crop_production_model.pkl'):
        if self.model is None:
            raise Exception("No model to save. Please train the model first.")
        joblib.dump({'model': self.model}, filename)
        print(f"Model saved as {filename}")

    def load_model(self, filename='crop_production_model.pkl'):
        loaded = joblib.load(filename)
        self.model = loaded['model']
        print(f"Model loaded from {filename}")


def create_prediction_ui():
    model_file = 'crop_production_model.pkl'
    data_path = 'crop_production.csv'

    predictor = CropProductionPredictor()

    if not os.path.exists(model_file):
        predictor.load_data(data_path)
        predictor.train_model()
        predictor.save_model()
    else:
        predictor.load_model()
        if os.path.exists(data_path):
            predictor.load_data(data_path)

    root = tk.Tk()
    root.title("Crop Production Predictor")
    root.geometry("650x600")
    root.configure(bg="#e8f0fe")

    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TLabel', font=('Segoe UI', 12), background="#e8f0fe")
    style.configure('TButton', font=('Segoe UI', 12, 'bold'))
    style.configure('TEntry', font=('Segoe UI', 12))

    main_frame = ttk.Frame(root, padding="20")
    main_frame.pack(fill=tk.BOTH, expand=True)

    title_label = ttk.Label(main_frame, text="Crop Production Prediction", font=('Segoe UI', 18, 'bold'))
    title_label.grid(row=0, column=0, columnspan=2, pady=20)

    state_to_districts = {"Andaman and Nicobar Islands": ["NICOBARS", "NORTH AND MIDDLE ANDAMAN", "SOUTH ANDAMANS"],
                          "Andhra Pradesh": ["ANANTAPUR", "CHITTOOR", "EAST GODAVARI", "GUNTUR", "KADAPA", "KRISHNA",
                                             "KURNOOL",
                                             "PRAKASAM", "SPSR NELLORE", "SRIKAKULAM", "VISAKHAPATANAM", "VIZIANAGARAM",
                                             "WEST GODAVARI"],
                          "Arunachal Pradesh": ["ANJAW", "CHANGLANG", "DIBANG VALLEY", "EAST KAMENG", "EAST SIANG",
                                                "KURUNG KUMEY",
                                                "LOHIT", "LONGDING", "LOWER DIBANG VALLEY", "LOWER SUBANSIRI", "NAMSAI",
                                                "PAPUM PARE",
                                                "TAWANG", "TIRAP", "UPPER SIANG", "UPPER SUBANSIRI", "WEST KAMENG",
                                                "WEST SIANG"],
                          "Assam": ["BAKSA", "BARPETA", "BONGAIGAON", "CACHAR", "CHIRANG", "DARRANG", "DHEMAJI",
                                    "DHUBRI", "DIBRUGARH",
                                    "DIMA HASAO", "GOALPARA", "GOLAGHAT", "HAILAKANDI", "JORHAT", "KAMRUP",
                                    "KAMRUP METRO",
                                    "KARBI ANGLONG", "KARIMGANJ", "KOKRAJHAR", "LAKHIMPUR", "MARIGAON", "NAGAON",
                                    "NALBARI",
                                    "SIVASAGAR", "SONITPUR", "TINSUKIA", "UDALGURI"],
                          "Bihar": ["ARARIA", "ARWAL", "AURANGABAD", "BANKA", "BEGUSARAI", "BHAGALPUR", "BHOJPUR",
                                    "BUXAR",
                                    "DARBHANGA", "GAYA", "GOPALGANJ", "JAMUI", "JEHANABAD", "KAIMUR (BHABUA)",
                                    "KATIHAR", "KHAGARIA",
                                    "KISHANGANJ", "LAKHISARAI", "MADHEPURA", "MADHUBANI", "MUNGER", "MUZAFFARPUR",
                                    "NALANDA",
                                    "NAWADA", "PASHCHIM CHAMPARAN", "PATNA", "PURBI CHAMPARAN", "PURNIA", "ROHTAS",
                                    "SAHARSA",
                                    "SAMASTIPUR", "SARAN", "SHEIKHPURA", "SHEOHAR", "SITAMARHI", "SIWAN", "SUPAUL",
                                    "VAISHALI"],
                          "Chandigarh": ["CHANDIGARH"],
                          "Chhattisgarh": ["BALOD", "BALODA BAZAR", "BALRAMPUR", "BASTAR", "BEMETARA", "BIJAPUR",
                                           "BILASPUR", "DANTEWADA",
                                           "DHAMTARI", "DURG", "GARIYABAND", "JANJGIR-CHAMPA", "JASHPUR", "KABIRDHAM",
                                           "KANKER",
                                           "KONDAGAON", "KORBA", "KOREA", "MAHASAMUND", "MUNGELI", "NARAYANPUR",
                                           "RAIGARH", "RAIPUR",
                                           "RAJNANDGAON", "SUKMA", "SURAJPUR", "SURGUJA"],
                          "Dadra and Nagar Haveli": ["DADRA AND NAGAR HAVELI"],
                          "Goa": ["NORTH GOA", "SOUTH GOA"],
                          "Gujarat": ["AHMADABAD", "AMRELI", "ANAND", "BANAS KANTHA", "BHARUCH", "BHAVNAGAR", "DANG",
                                      "DOHAD",
                                      "GANDHINAGAR", "JAMNAGAR", "JUNAGADH", "KACHCHH", "KHEDA", "MAHESANA", "NARMADA",
                                      "NAVSARI",
                                      "PANCH MAHALS", "PATAN", "PORBANDAR", "RAJKOT", "SABAR KANTHA", "SURAT",
                                      "SURENDRANAGAR",
                                      "TAPI", "VADODARA", "VALSAD"],
                          "Jharkhand": ["BOKARO", "CHATRA", "DEOGHAR", "DHANBAD", "DUMKA", "EAST SINGHBUM", "GARHWA",
                                        "GIRIDIH", "GODDA",
                                        "GUMLA", "HAZARIBAGH", "JAMTARA", "KHUNTI", "KODERMA", "LATEHAR", "LOHARDAGA",
                                        "PAKUR", "PALAMU",
                                        "RAMGARH", "RANCHI", "SAHEBGANJ", "SARAIKELA KHARSAWAN", "SIMDEGA",
                                        "WEST SINGHBUM"],
                          "Karnataka": ["BAGALKOT", "BANGALORE RURAL", "BELGAUM", "BELLARY", "BENGALURU URBAN", "BIDAR",
                                        "BIJAPUR",
                                        "CHAMARAJANAGAR", "CHIKBALLAPUR", "CHIKMAGALUR", "CHITRADURGA",
                                        "DAKSHIN KANNAD", "DAVANGERE",
                                        "DHARWAD", "GADAG", "GULBARGA", "HASSAN", "HAVERI", "KODAGU", "KOLAR", "KOPPAL",
                                        "MANDYA",
                                        "MYSORE", "RAICHUR", "RAMANAGARA", "SHIMOGA", "TUMKUR", "UDUPI", "UTTAR KANNAD",
                                        "YADGIR"],
                          "Kerala": ["ALAPPUZHA", "ERNAKULAM", "IDUKKI", "KANNUR", "KASARAGOD", "KOLLAM", "KOTTAYAM",
                                     "KOZHIKODE",
                                     "MALAPPURAM", "PALAKKAD", "PATHANAMTHITTA", "THIRUVANANTHAPURAM", "THRISSUR",
                                     "WAYANAD"],
                          "Madhya Pradesh": ["AGAR MALWA", "ALIRAJPUR", "ANUPPUR", "ASHOKNAGAR", "BALAGHAT", "BARWANI",
                                             "BETUL",
                                             "BHIND", "BHOPAL", "BURHANPUR", "CHHATARPUR", "CHHINDWARA", "DAMOH",
                                             "DATIA", "DEWAS",
                                             "DHAR", "DINDORI", "GUNA", "GWALIOR", "HARDA", "HOSHANGABAD", "INDORE",
                                             "JABALPUR",
                                             "JHABUA", "KATNI", "KHANDWA", "KHARGONE", "MANDLA", "MANDSAUR", "MORENA",
                                             "NARSINGHPUR",
                                             "NEEMUCH", "PANNA", "RAISEN", "RAJGARH", "RATLAM", "REWA", "SAGAR",
                                             "SATNA", "SEHORE",
                                             "SEONI", "SHAHDOL", "SHAJAPUR", "SHEOPUR", "SHIVPURI", "SIDHI",
                                             "SINGRAULI", "TIKAMGARH",
                                             "UJJAIN", "UMARIA", "VIDISHA"],
                          "Maharashtra": ["AHMEDNAGAR", "AKOLA", "AMRAVATI", "AURANGABAD", "BEED", "BHANDARA",
                                          "BULDHANA", "CHANDRAPUR",
                                          "DHULE", "GADCHIROLI", "GONDIA", "HINGOLI", "JALGAON", "JALNA", "KOLHAPUR",
                                          "LATUR", "MUMBAI",
                                          "NAGPUR", "NANDED", "NANDURBAR", "NASHIK", "OSMANABAD", "PALGHAR", "PARBHANI",
                                          "PUNE",
                                          "RAIGAD", "RATNAGIRI", "SANGLI", "SATARA", "SINDHUDURG", "SOLAPUR", "THANE",
                                          "WARDHA",
                                          "WASHIM", "YAVATMAL"],
                          "Manipur": ["BISHNUPUR", "CHANDEL", "CHURACHANDPUR", "IMPHAL EAST", "IMPHAL WEST", "SENAPATI",
                                      "TAMENGLONG", "THOUBAL", "UKHRUL"],
                          "Meghalaya": ["EAST GARO HILLS", "EAST JAINTIA HILLS", "EAST KHASI HILLS", "NORTH GARO HILLS",
                                        "RI BHOI",
                                        "SOUTH GARO HILLS", "SOUTH WEST GARO HILLS", "SOUTH WEST KHASI HILLS",
                                        "WEST GARO HILLS",
                                        "WEST JAINTIA HILLS", "WEST KHASI HILLS"],
                          "Mizoram": ["AIZAWL", "CHAMPHAI", "KOLASIB", "LAWNGTLAI", "LUNGLEI", "MAMIT", "SAIHA",
                                      "SERCHHIP"],
                          "Nagaland": ["DIMAPUR", "KIPHIRE", "KOHIMA", "LONGLENG", "MOKOKCHUNG", "MON", "PEREN", "PHEK",
                                       "TUENSANG",
                                       "WOKHA", "ZUNHEBOTO"],
                          "Odisha": ["ANUGUL", "BALANGIR", "BALESHWAR", "BARGARH", "BHADRAK", "BOUDH", "CUTTACK",
                                     "DEOGARH", "DHENKANAL",
                                     "GAJAPATI", "GANJAM", "JAGATSINGHAPUR", "JAJAPUR", "JHARSUGUDA", "KALAHANDI",
                                     "KANDHAMAL",
                                     "KENDRAPARA", "KENDUJHAR", "KHORDHA", "KORAPUT", "MALKANGIRI", "MAYURBHANJ",
                                     "NABARANGPUR",
                                     "NAYAGARH", "NUAPADA", "PURI", "RAYAGADA", "SAMBALPUR", "SONEPUR", "SUNDARGARH"],
                          "Puducherry": ["KARAIKAL", "MAHE", "PONDICHERRY", "YANAM"],
                          "Punjab": ["AMRITSAR", "BARNALA", "BATHINDA", "FARIDKOT", "FATEHGARH SAHIB", "FAZILKA",
                                     "FIROZEPUR",
                                     "GURDASPUR", "HOSHIARPUR", "JALANDHAR", "KAPURTHALA", "LUDHIANA", "MANSA", "MOGA",
                                     "MUKTSAR",
                                     "NAWANSHAHR", "PATHANKOT", "PATIALA", "RUPNAGAR", "S.A.S NAGAR", "SANGRUR",
                                     "TARN TARAN"],
                          "Rajasthan": ["AJMER", "ALWAR", "BANSWARA", "BARAN", "BARMER", "BHARATPUR", "BHILWARA",
                                        "BIKANER", "BUNDI",
                                        "CHITTORGARH", "CHURU", "DAUSA", "DHOLPUR", "DUNGARPUR", "GANGANAGAR",
                                        "HANUMANGARH", "JAIPUR",
                                        "JAISALMER", "JALORE", "JHALAWAR", "JHUNJHUNU", "JODHPUR", "KARAULI", "KOTA",
                                        "NAGAUR",
                                        "PALI", "PRATAPGARH", "RAJSAMAND", "SAWAI MADHOPUR", "SIKAR", "SIROHI", "TONK",
                                        "UDAIPUR"],
                          "Sikkim": ["EAST DISTRICT", "NORTH DISTRICT", "SOUTH DISTRICT", "WEST DISTRICT"],
                          "Tamil Nadu": ["ARIYALUR", "COIMBATORE", "CUDDALORE", "DHARMAPURI", "DINDIGUL", "ERODE",
                                         "KANCHIPURAM",
                                         "KANNIYAKUMARI", "KARUR", "KRISHNAGIRI", "MADURAI", "NAGAPATTINAM", "NAMAKKAL",
                                         "PERAMBALUR",
                                         "PUDUKKOTTAI", "RAMANATHAPURAM", "SALEM", "SIVAGANGA", "THANJAVUR",
                                         "THE NILGIRIS", "THENI",
                                         "THIRUVALLUR", "THIRUVARUR", "TIRUCHIRAPPALLI", "TIRUNELVELI", "TIRUPPUR",
                                         "TIRUVANNAMALAI",
                                         "TUTICORIN", "VELLORE", "VILLUPURAM", "VIRUDHUNAGAR"],
                          "Telangana": ["ADILABAD", "HYDERABAD", "KARIMNAGAR", "KHAMMAM", "MAHBUBNAGAR", "MEDAK",
                                        "NALGONDA",
                                        "NIZAMABAD", "RANGAREDDI", "WARANGAL"],
                          "Tripura": ["DHALAI", "GOMATI", "KHOWAI", "NORTH TRIPURA", "SEPAHIJALA", "SOUTH TRIPURA",
                                      "UNAKOTI", "WEST TRIPURA"],
                          "Uttar Pradesh": ["AGRA", "ALIGARH", "ALLAHABAD", "AMBEDKAR NAGAR", "AMETHI", "AMROHA",
                                            "AURAIYA",
                                            "AZAMGARH", "BAGHPAT", "BAHRAICH", "BALLIA", "BALRAMPUR", "BANDA",
                                            "BARABANKI", "BAREILLY",
                                            "BASTI", "BIJNOR", "BUDAUN", "BULANDSHAHR", "CHANDAULI", "CHITRAKOOT",
                                            "DEORIA", "ETAH",
                                            "ETAWAH", "FAIZABAD", "FARRUKHABAD", "FATEHPUR", "FIROZABAD",
                                            "GAUTAM BUDDHA NAGAR",
                                            "GHAZIABAD", "GHAZIPUR", "GONDA", "GORAKHPUR", "HAMIRPUR", "HAPUR",
                                            "HARDOI", "HATHRAS",
                                            "JALAUN", "JAUNPUR", "JHANSI", "KANNAUJ", "KANPUR DEHAT", "KANPUR NAGAR",
                                            "KASGANJ",
                                            "KAUSHAMBI", "KHERI", "KUSHI NAGAR", "LALITPUR", "LUCKNOW", "MAHARAJGANJ",
                                            "MAHOBA",
                                            "MAINPURI", "MATHURA", "MAU", "MEERUT", "MIRZAPUR", "MORADABAD",
                                            "MUZAFFARNAGAR",
                                            "PILIBHIT", "PRATAPGARH", "RAE BARELI", "RAMPUR", "SAHARANPUR", "SAMBHAL",
                                            "SANT KABEER NAGAR", "SANT RAVIDAS NAGAR", "SHAHJAHANPUR", "SHAMLI",
                                            "SHRAVASTI",
                                            "SIDDHARTH NAGAR", "SITAPUR", "SONBHADRA", "SULTANPUR", "UNNAO",
                                            "VARANASI"],
                          "Uttarakhand": ["ALMORA", "BAGESHWAR", "CHAMOLI", "CHAMPAWAT", "DEHRADUN", "HARIDWAR",
                                          "NAINITAL",
                                          "PAURI GARHWAL", "PITHORAGARH", "RUDRA PRAYAG", "TEHRI GARHWAL",
                                          "UDAM SINGH NAGAR",
                                          "UTTAR KASHI"],
                          "West Bengal": ["24 PARAGANAS NORTH", "24 PARAGANAS SOUTH", "BANKURA", "BARDHAMAN", "BIRBHUM",
                                          "COOCHBEHAR", "DARJEELING", "DINAJPUR DAKSHIN", "DINAJPUR UTTAR", "HOOGHLY",
                                          "HOWRAH",
                                          "JALPAIGURI", "MALDAH", "MEDINIPUR EAST", "MEDINIPUR WEST", "MURSHIDABAD",
                                          "NADIA",
                                          "PURULIA"]}  # <-- Paste your big state_to_districts dictionary here
    crops = ["Arecanut", "Other Kharif pulses", "Rice", "Banana", "Cashewnut", "Coconut", "Dry ginger", "Sugarcane",
             "Sweet potato", "Tapioca", "Black pepper", "Dry chillies", "other oilseeds", "Turmeric", "Maize",
             "Moong(Green Gram)", "Urad", "Arhar/Tur", "Groundnut", "Sunflower", "Bajra", "Castor seed", "Cotton(lint)",
             "Horse-gram", "Jowar", "Korra", "Ragi", "Tobacco", "Gram", "Wheat", "Masoor", "Sesamum", "Linseed",
             "Safflower", "Onion", "other misc. pulses", "Samai", "Small millets", "Coriander", "Potato",
             "Other  Rabi pulses", "Soyabean", "Beans & Mutter(Vegetable)", "Bhindi", "Brinjal", "Citrus Fruit",
             "Cucumber", "Grapes", "Mango", "Orange", "other fibres", "Other Fresh Fruits", "Other Vegetables",
             "Papaya", "Pome Fruit", "Tomato", "Rapeseed &Mustard", "Mesta", "Cowpea(Lobia)", "Lemon", "Pome Granet",
             "Sapota", "Cabbage", "Peas  (vegetable)", "Niger seed", "Bottle Gourd", "Sannhamp", "Varagu", "Garlic",
             "Ginger", "Oilseeds total", "Pulses total", "Jute", "Peas & beans (Pulses)", "Blackgram", "Paddy",
             "Pineapple", "Barley", "Khesari", "Guar seed", "Moth", "Other Cereals & Millets", "Cond-spcs other",
             "Turnip", "Carrot", "Redish", "Arcanut (Processed)", "Atcanut (Raw)", "Cashewnut Processed",
             "Cashewnut Raw", "Cardamom", "Rubber", "Bitter Gourd", "Drum Stick", "Jack Fruit", "Snak Guard",
             "Pump Kin", "Tea", "Coffee", "Cauliflower", "Other Citrus Fruit", "Water Melon", "Total foodgrain",
             "Kapas", "Colocosia", "Lentil", "Bean", "Jobster", "Perilla", "Rajmash Kholar", "Ricebean (nagadal)",
             "Ash Gourd", "Beet Root", "Lab-Lab", "Ribed Guard", "Yam", "Apple", "Peach", "Pear", "Plums",
             "Litchi", "Ber", "Other Dry Fruit", "Jute & mesta"]  # <-- Paste your crops list here

    seasons = ["Kharif", "Rabi", "Zaid", "Whole Year", "Autumn", "Summer", "Winter"]

    ttk.Label(main_frame, text="State Name:").grid(row=1, column=0, sticky=tk.W, pady=10)
    state_var = tk.StringVar()
    state_dropdown = ttk.Combobox(main_frame, textvariable=state_var, values=list(state_to_districts.keys()), width=35,
                                  state="readonly")
    state_dropdown.grid(row=1, column=1, pady=10)

    ttk.Label(main_frame, text="District Name:").grid(row=2, column=0, sticky=tk.W, pady=10)
    district_var = tk.StringVar()
    district_dropdown = ttk.Combobox(main_frame, textvariable=district_var, width=35, state="readonly")
    district_dropdown.grid(row=2, column=1, pady=10)

    ttk.Label(main_frame, text="Season:").grid(row=3, column=0, sticky=tk.W, pady=10)
    season_var = tk.StringVar()
    season_dropdown = ttk.Combobox(main_frame, textvariable=season_var, values=seasons, width=35, state="readonly")
    season_dropdown.grid(row=3, column=1, pady=10)

    ttk.Label(main_frame, text="Crop:").grid(row=4, column=0, sticky=tk.W, pady=10)
    crop_var = tk.StringVar()
    crop_dropdown = ttk.Combobox(main_frame, textvariable=crop_var, values=crops, width=35, state="readonly")
    crop_dropdown.grid(row=4, column=1, pady=10)

    ttk.Label(main_frame, text="Area (hectares):").grid(row=5, column=0, sticky=tk.W, pady=10)
    area_var = tk.StringVar()
    area_entry = ttk.Entry(main_frame, textvariable=area_var, width=38)
    area_entry.grid(row=5, column=1, pady=10)
    area_entry.insert(0, "100")

    result_frame = ttk.LabelFrame(main_frame, text="Prediction Result", padding="10")
    result_frame.grid(row=6, column=0, columnspan=2, pady=20, sticky="ew")

    production_label = ttk.Label(result_frame, text="Predicted Production: ")
    production_label.grid(row=0, column=0, sticky=tk.W, pady=5)
    production_value = ttk.Label(result_frame, text="--", font=('Segoe UI', 12, 'bold'))
    production_value.grid(row=0, column=1, sticky=tk.W, pady=5)

    def update_districts(*args):
        selected_state = state_var.get()
        districts = state_to_districts.get(selected_state, [])
        district_dropdown['values'] = districts
        district_var.set("")

    state_var.trace("w", update_districts)

    def predict():
        try:
            area = float(area_var.get())
            if area <= 0:
                messagebox.showerror("Invalid Input", "Area must be a positive number.")
                return

            production = predictor.predict_production(
                state_name=state_var.get(),
                district_name=district_var.get(),
                season=season_var.get(),
                crop=crop_var.get(),
                area=area
            )

            production_value.config(text=f"{production:.2f} units")

        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number for Area.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    predict_btn = ttk.Button(main_frame, text="Predict Production", command=predict)
    predict_btn.grid(row=7, column=0, columnspan=2, pady=20)

    root.mainloop()


if __name__ == "__main__":
    create_prediction_ui()
