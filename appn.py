from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from werkzeug.utils import secure_filename
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import cv2
from pathlib import Path
from google.cloud import vision
import google.generativeai as genai
import os
import exifread
from geopy.distance import geodesic
from decimal import Decimal

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'hello'
app.config['UPLOAD_FOLDER'] = 'static/uploaded'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Configure Google Vision API client
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'PATH OF GOOGLE APP CRED'
vision_client = vision.ImageAnnotatorClient()
db = SQLAlchemy(app)

# Define a User model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    phone_number = db.Column(db.String(20), nullable=False)

class Upload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    latitude = db.Column(db.Numeric(precision=18, scale=15), nullable=False)  # Increase precision to 15 digits
    longitude = db.Column(db.Numeric(precision=18, scale=15), nullable=False)  # Increase precision to 15 digits
    filepath = db.Column(db.String(200), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email, password=password).first()
        if user:
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password', 'error')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        email = request.form['email']
        password = request.form['password']
        phone_number = request.form['phone_number']

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already exists', 'error')
            return redirect(url_for('register'))

        new_user = User(first_name=first_name, last_name=last_name, email=email, password=password, phone_number=phone_number)
        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for('login'))
    return render_template('register.html')


# Flask WTForms Upload Form
class UploadForm(FlaskForm):
    file = FileField(validators=[FileRequired()])

# Function to check image dimensions
def is_above_540p(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return False
    height, width, _ = image.shape
    return width >= 960 and height >= 540

# Function to check image blur
def is_blurry(image_path, threshold=100.0):
    image = cv2.imread(image_path)
    if image is None:
        return False, 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    is_blur = variance < threshold
    return is_blur, variance

# Function to perform web detection using Google Vision API
def annotate(path: str) -> vision.WebDetection:
    if os.path.exists(path):
        with open(path, "rb") as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
    else:
        raise FileNotFoundError(f"File {path} not found.")

    web_detection = vision_client.web_detection(image=image).web_detection
    return web_detection

# Function to find duplicates
def find_duplicates(upload_folder, filepath):
    files = {}
    for root, dirs, file_names in os.walk(upload_folder):
        for filename in file_names:
            path = os.path.join(root, filename)
            if os.path.isfile(path) and path != filepath:
                with open(path, 'rb') as file:
                    file_hash = hash(file.read())
                    if file_hash in files:
                        return True  # Duplicate found
                    files[file_hash] = filename
    return False  # No duplicates found

# EXIF extraction functions
def get_exif_data(image_path):
    """Get embedded EXIF data from image file using ExifRead."""
    with open(image_path, 'rb') as image_file:
        tags = exifread.process_file(image_file)
    return tags

def get_gps_data(tags):
    """Extract GPS data from EXIF tags."""
    gps_data = {}
    gps_keys = ['GPS GPSLatitude', 'GPS GPSLatitudeRef', 'GPS GPSLongitude', 'GPS GPSLongitudeRef']
    for key in gps_keys:
        if key in tags:
            gps_data[key] = tags[key]
    return gps_data

def convert_to_degrees(value):
    """Convert GPS coordinates to degrees, avoiding zero division."""
    def safe_div(num, den):
        return float(num) / float(den) if den != 0 else 0

    if len(value.values) != 3:
        print("Incomplete GPS coordinate data.")
        return None

    d = safe_div(value.values[0].num, value.values[0].den)
    m = safe_div(value.values[1].num, value.values[1].den)
    s = safe_div(value.values[2].num, value.values[2].den)
    return d + (m / 60.0) + (s / 3600.0)

def get_lat_lon(gps_data):
    """Extract latitude and longitude from GPS data."""
    lat = None
    lon = None

    gps_latitude = gps_data.get('GPS GPSLatitude')
    gps_latitude_ref = gps_data.get('GPS GPSLatitudeRef')
    gps_longitude = gps_data.get('GPS GPSLongitude')
    gps_longitude_ref = gps_data.get('GPS GPSLongitudeRef')

    if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
        lat = convert_to_degrees(gps_latitude)
        lon = convert_to_degrees(gps_longitude)

        if lat is None or lon is None:
            return None, None

        if gps_latitude_ref.values[0] != 'N':
            lat = -lat
        if gps_longitude_ref.values[0] != 'E':
            lon = -lon

    return lat, lon

def extract_gps_from_image(image_path):
    tags = get_exif_data(image_path)
    gps_data = get_gps_data(tags)
    lat, lon = get_lat_lon(gps_data)
    return lat, lon

def get_google_maps_url(latitude, longitude, api_key):
    # Generate the Google Maps URL with a marker at the specified location
    map_url = f"https://www.google.com/maps/embed/v1/place?key={api_key}&q={latitude},{longitude}&zoom=15&maptype=satellite"
    return map_url

# Main upload route
@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_form():
    form = UploadForm()

    if form.validate_on_submit():
        file = form.file.data
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Check if image dimensions are above 540p
        if not is_above_540p(filepath):
            os.remove(filepath)
            flash('Image dimensions should be at least 960x540 pixels.', 'error')
            return redirect(url_for('error2'))

        # Check if image is blurry
        is_blur, variance = is_blurry(filepath)
        if is_blur:
            os.remove(filepath)
            flash('The image is too blurry.', 'error')
            return redirect(url_for('error2'))

        # Check for duplicate images
        if find_duplicates(app.config['UPLOAD_FOLDER'], filepath):
            os.remove(filepath)
            flash('Duplicate image found.', 'error')
            return redirect(url_for('error2'))

        # Perform recognition using Gemini
        image_path = Path(filepath)
        image_part = {
            "mime_type": "image/jpeg",
            "data": image_path.read_bytes()
        }
        genai.configure(api_key="GEMINI-API-KEY")  # Replace with your Gemini API key
        generation_config = {"temperature": 0.9, "top_p": 1, "top_k": 1, "max_output_tokens": 2048}
        model = genai.GenerativeModel("gemini-1.5-pro-001", generation_config=generation_config)
        prompt_parts = [
            "Is it a plant or sapling? Answer YES or NO:\n",
            image_part
        ]
        response = model.generate_content(prompt_parts)
        if "error" in response.text.lower():
            os.remove(filepath)
            flash("Error recognizing the image content.", "error")
            return redirect(url_for('error1'))

        recognition_result = response.text.strip().lower()
        if recognition_result != "yes":
            os.remove(filepath)
            flash("The image does not contain a plant or sapling.", 'error')
            return redirect(url_for('error1'))

        prompt_partsa = [
            "Recognize the plant and give output as name only:\n",
            image_part
        ]
        responsee = model.generate_content(prompt_partsa)
        plant_name = responsee.text.strip()

        # Extract GPS coordinates from the uploaded image
        latitude, longitude = extract_gps_from_image(filepath)
        if latitude is None or longitude is None:
            os.remove(filepath)
            flash('No GPS data found or data is incomplete.', 'error')
            return redirect(url_for('error3'))

        # Check if the GPS coordinates are at least 3 meters away from existing coordinates
        uploads = Upload.query.all()
        for upload in uploads:
            distance = geodesic((latitude, longitude), (upload.latitude, upload.longitude)).meters
            if distance < 3:
                os.remove(filepath)
                flash('The location is too close to an existing upload.', 'error')
                return redirect(url_for('error3'))

        # Save the upload record
        new_upload = Upload(user_id=current_user.id, latitude=latitude, longitude=longitude, filepath=filepath)
        db.session.add(new_upload)
        db.session.commit()

        # Store plant_name in session
        session['plant_name'] = plant_name

        flash('File successfully uploaded', 'success')
        return redirect(url_for('success'))

    return render_template('upload.html', form=form)

# Error routes
@app.route('/error')
def error():
    return render_template('error.html')

@app.route('/error1')
def error1():
    return render_template('error1.html')

@app.route('/error2')
def error2():
    return render_template('error2.html')

@app.route('/error3')
def error3():
    return render_template('error3.html')

# Success route
@app.route('/success')
def success():
    return render_template('passed.html')

# Route to display map with all upload locations
@app.route('/plantmap')
def plantmap():
    uploads = Upload.query.all()
    api_key = 'GOOGLE-MAPS-API-KEY'  # Replace with your actual Google Maps API key
    markers = [{'lat': "{:.8f}".format(upload.latitude), 'lng': "{:.8f}".format(upload.longitude)} for upload in uploads]
    return render_template('plantmap.html', markers=markers, api_key=api_key)

# HTML template for displaying map
MAP_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Google Maps Embed</title>
    <style>
        #map {
            height: 450px;
            width: 100%;
        }
    </style>
</head>
<body>
    <h1>Plant Locations</h1>
    <div id="map"></div>
    <script>
        function initMap() {
            var center = {lat: 0, lng: 0}; // Default center of the map
            var markers = {{ markers | tojson }};
            var map = new google.maps.Map(document.getElementById('map'), {
                zoom: 2,
                center: center
            });
            markers.forEach(function(marker) {
                new google.maps.Marker({
                    position: {lat: marker.lat, lng: marker.lng},
                    map: map
                });
            });
        }
    </script>
    <script async defer
        src="https://maps.googleapis.com/maps/api/js?key={{ api_key }}&callback=initMap">
    </script>
</body>
</html>
'''

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    with app.app_context():
        db.create_all()
    app.run()
