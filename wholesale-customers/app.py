from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model
model_filename = "models/BaggingClassifier_n_estimators_3.pkl"
with open(model_filename, "rb") as file:
    model = pickle.load(file)

# Encoding dictionary
encoding_dict = {
    "Region": {1: 0, 2: 1, 3: 2}
}

# Contoh data yang di-encode
data = {
    "Region": 1,
    "Fresh": 12669,
    "Milk": 9656,
    "Grocery": 7561,
    "Frozen": 214,
    "Detergents_Paper": 2674,
    "Delicassen": 1338
}




# Class names
class_names = {
    0: "(Horeca)",
    1: "(Retail)"
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Mengambil nilai input dari form
        input_data = {
            "Region": encoding_dict["Region"][data["Region"]],
            "Fresh": data["Fresh"],
            "Milk": data["Milk"],
            "Grocery": data["Grocery"],
            "Frozen": data["Frozen"],
            "Detergents_Paper": data["Detergents_Paper"],
            "Delicassen": data["Delicassen"]
        }
        
        # Convert ke DataFrame
        mydata = pd.DataFrame([input_data])
        
        # Convert type numeric ke jenis data yang sesuai
        # mydata["doors"] = mydata["doors"].replace({"2": 2, "3": 3, "4": 4, "5more": "5more"})
        # mydata["persons"] = mydata["persons"].replace({"2": 2, "4": 4, "more": "more"})
        
        # Apply encoding
        encoded_data = mydata.copy()
        for col, mapping in encoding_dict.items():
            encoded_data.loc[0, col] = mapping[encoded_data.loc[0, col]]
        
        # Prediksi
        predictions = model.predict(encoded_data)
        predicted_class = predictions[0]
        predicted_class_name = class_names.get(predicted_class, "Unknown")
        
        return render_template("index.html", prediction=predicted_class_name, form_data=input_data)
    
    return render_template("index.html", prediction=None, form_data=None)

if __name__ == "__main__":
    app.run(debug=True)
