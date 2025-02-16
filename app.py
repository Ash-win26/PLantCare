from flask import Flask, request, jsonify, render_template, url_for, send_from_directory, session, redirect, flash
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import pandas as pd
from werkzeug.utils import secure_filename
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps  # For login_required decorator
from flask_mail import Mail, Message

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Flask-Mail Configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'  # Replace with your SMTP server
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'plantcareai@gmail.com'  # Replace with your email
app.config['MAIL_PASSWORD'] = 'bsay zzgk wskb ysku'  # Use an app password for security
app.config['MAIL_DEFAULT_SENDER'] = 'your_email@gmail.com'  # Default sender

mail = Mail(app)

app.secret_key = 'your_secret_key'  # Change this to a secure random key

# Database configuration (SQLite for simplicity)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# Load the Keras model
MODEL_PATH = "plant_disease_model.keras"
model = load_model(MODEL_PATH)

# Define target size for input images
IMG_SIZE = (128, 128)

# Upload folder setup
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load disease information CSV
CSV_PATH = "disease_info.csv"
df = pd.read_csv(CSV_PATH)

# Load supplement information CSV
SUPPLEMENT_CSV_PATH = "supplement_info.csv"
supplement_df = pd.read_csv(SUPPLEMENT_CSV_PATH)

# Class labels (adjust according to your model)
CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",  
    "Apple___healthy", "UPLOADED IMAGE IS NOT PLANT RELATED!", "Blueberry___healthy", 
    "Cherry___healthy", "Cherry___Powdery_mildew", "Corn___Cercospora_leaf_spot Gray_leaf_spot", 
    "Corn___Common_rust", "Corn___healthy", "Corn___Northern_Leaf_Blight", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___healthy", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy", "Pepper_bell___Bacterial_spot",
    "Pepper_bell___healthy", "Potato___Early_blight", "Potato___healthy", "Potato___Late_blight",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew", "Strawberry___healthy",
    "Strawberry___Leaf_scorch", "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___healthy",
    "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_mosaic_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]

# Ensure database is created
with app.app_context():
    db.create_all()

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please log in first!", "danger")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/market')
def market():
    return render_template('market.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        # Create email message
        msg = Message("New Contact Form Submission",
                      sender=email,
                      recipients=['plantcareai@gmail.com'])  # Replace with your email
        msg.body = f"Name: {name}\nEmail: {email}\n\nMessage:\n{message}"

        try:
            mail.send(msg)
            flash("Message sent successfully!", "success")
        except Exception as e:
            flash(f"Error sending message: {str(e)}", "danger")

        return redirect(url_for('contact'))

    return render_template('contact.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        if User.query.filter_by(email=email).first():
            flash("Email already registered. Please log in.", "danger")
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(name=name, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful! Please log in.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['user_name'] = user.name
            flash("Login successful!", "success")
            return redirect(url_for('result'))  # Redirect to result page
        else:
            flash("Invalid credentials. Please try again.", "danger")

    return render_template('login.html')

@app.route('/logout', methods=['GET'])
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for('home'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/result')
@login_required
def result():
    return render_template('result.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        image = load_img(file_path, target_size=IMG_SIZE)
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_class_index]
        confidence = f"{np.max(predictions) * 100:.2f}%"

        disease_info = df[df['Disease'] == predicted_class].to_dict(orient='records')
        disease_info = disease_info[0] if disease_info else {
            "Description": "No description available for this disease.",
            "Possible Steps": "N/A",
            "image_url": ""
        }

        # Fetch supplement info
        supplement = supplement_df[supplement_df["disease_name"] == predicted_class]
        supplement_info = supplement.iloc[0].to_dict() if not supplement.empty else None

        return jsonify({
            "uploaded_image_url": url_for('uploaded_file', filename=filename, _external=True),
            "predicted_class": predicted_class,
            "confidence": confidence,
            "description": disease_info.get("Description", ""),
            "steps": disease_info.get("Possible Steps", ""),
            "image_url": disease_info.get("image_url", ""),
            "supplement": {
                "name": supplement_info.get("supplement_name", "") if supplement_info else "",
                "image": supplement_info.get("supplement_image", "") if supplement_info else "",
                "link": supplement_info.get("buy_link", "") if supplement_info else ""
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
