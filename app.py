from flask import Flask, render_template, request
import pickle
import numpy as np

# Load model bundle
with open("kmeans_model_bundle.pkl", "rb") as f:
    bundle = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = []
        for feature in bundle["features"]:
            val = request.form.get(feature)

            # Apply label encoding if needed
            if feature in bundle["label_encoders"]:
                le = bundle["label_encoders"][feature]
                if val not in le.classes_:
                    # Handle unseen values
                    return render_template("index.html", prediction=f"Error: Unknown value '{val}' for {feature}")
                val = le.transform([val])[0]
            else:
                val = float(val)

            input_data.append(val)

        # Convert to array and reshape
        input_array = np.array(input_data).reshape(1, -1)

        # Scale and predict
        X_scaled = bundle["scaler"].transform(input_array)
        # X_pca = bundle["pca"].transform(X_scaled)
        cluster = bundle["kmeans"].predict(X_pca)[0]

        return render_template("index.html", prediction=f"Predicted Cluster: {cluster}")

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {e}")


if __name__ == "__main__":
    app.run(debug=True)
