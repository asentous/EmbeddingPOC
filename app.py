from flask import Flask, render_template, url_for, redirect, flash, request, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, PasswordField, SubmitField
from wtforms.validators import input_required, length, ValidationError, DataRequired
from flask_bcrypt import Bcrypt
from dotenv import load_dotenv
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.chains import LLMChain 
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Initialize the SentenceTransformer model for semantic embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Flask app and configuration
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'your_secret_key'

# Initialize database, login manager, and bcrypt for password hashing
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# Initialize language model for JD generation
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Template for generating job descriptions
job_template = """
You are a helpful assistant that generates complete job descriptions based on the user's input. The user may provide information in any order or form, but the job description should include the following details:

- Company Name
- Location
- Salary Range
- Job Role
- Main Skill
- Job Type (Full-time/Part-time)
- Number of Openings
- Experience Required

Use the information provided by the user to create a structured job description. If any details are missing, use placeholders or fill in reasonable defaults.

User Input: 
{user_input}

Complete Job Description:
"""

# Create Langchain LLM Chain for job description generation
job_prompt_template = PromptTemplate(input_variables=["user_input"], template=job_template)
job_chain = LLMChain(llm=model, prompt=job_prompt_template)

# Function to embed resume and job description using the same model
def embed_text(text):
    return embedding_model.encode(text).tolist()

# Function to find the best candidate based on semantic embeddings
def find_best_candidate(job_desc_embedding):
    resumes = Resume.query.all()
    best_match = None
    best_similarity = -1

    # Iterate through each resume, calculate similarity, and find the best match
    for resume in resumes:
        embedding = np.array(resume.embedding)
        if embedding.shape == (384,):  # Check for correct embedding dimension
            similarity = cosine_similarity([job_desc_embedding], [embedding])[0][0]
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = resume
    
    return best_match

# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Define User model for authentication
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)

# Model for storing job descriptions and embeddings
class JobDescription(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    job_description = db.Column(db.Text, nullable=False)
    embedding = db.Column(db.PickleType, nullable=False)  # Storing embedding as a pickled list

# Model for storing resumes and embeddings
class Resume(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    candidate_name = db.Column(db.String(100), nullable=False)
    resume_text = db.Column(db.Text, nullable=False)
    embedding = db.Column(db.PickleType, nullable=False)  # Storing embedding as a pickled list

# Registration form
class RegisterForm(FlaskForm):
    username = StringField(validators=[input_required(), length(min=4, max=20)], render_kw={"placeholder": "Username"})
    password = PasswordField(validators=[input_required(), length(min=4, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField("Register")

    def validate_username(self, username):
        existing_user = User.query.filter_by(username=username.data).first()
        if existing_user:
            raise ValidationError("That username already exists. Please choose a different one.")

# Login form
class LoginForm(FlaskForm):
    username = StringField(validators=[input_required(), length(min=4, max=20)], render_kw={"placeholder": "Username"})
    password = PasswordField(validators=[input_required(), length(min=4, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField("Login")

# Resume upload form
class ResumeUploadForm(FlaskForm):
    candidate_name = StringField('Candidate Name', validators=[DataRequired()])
    resume_text = TextAreaField('Resume Text', validators=[DataRequired()])
    submit = SubmitField('Upload Resume')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect(url_for('dashboard'))
    return render_template('login.html', form=form)

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        existing_user = User.query.filter_by(username=form.username.data).first()
        if existing_user:
            flash('That username already exists. Please choose a different username.', 'Warning')
        else:
            hashed_password = bcrypt.generate_password_hash(form.password.data)
            new_user = User(username=form.username.data, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            flash('Account created successfully! Please log in.', 'Success')
            return redirect(url_for('login'))
    return render_template('register.html', form=form)

# Route for generating and storing job descriptions
@app.route('/chatbot', methods=['GET', 'POST'])
@login_required
def chatbot():
    if request.method == 'POST':
        user_input = request.form.get('user_input')

        # Check if user_input is empty or None
        if not user_input:
            flash("Input cannot be empty. Please provide the required information.", "warning")
            return redirect(url_for('chatbot'))

        try:
            # Use Langchain to generate the job description with the free-form user input
            response = job_chain.invoke({"user_input": user_input})
            generated_text = response.get("text", "")

            if generated_text:
                # Generate semantic embedding for the job description
                job_description_embedding = embed_text(generated_text)

                # Store job description and embedding
                new_job_desc = JobDescription(job_description=generated_text, embedding=job_description_embedding)
                db.session.add(new_job_desc)
                db.session.commit()

                session['generated_job_id'] = new_job_desc.id
                flash(f"Generated Job Description: {generated_text}", "success")
                flash("Do you want to use this job description to find suitable candidates?", "info")
            else:
                flash("Failed to generate a job description. Please try again.", "danger")
        except Exception as e:
            flash(f"Error generating job description: {str(e)}", "danger")

        return render_template('chatbot.html')

    return render_template('chatbot.html')


# Route for uploading resumes
@app.route('/upload_resume', methods=['GET', 'POST'])
@login_required
def upload_resume():
    form = ResumeUploadForm()
    if form.validate_on_submit():
        candidate_name = form.candidate_name.data
        resume_text = form.resume_text.data

        resume_embedding = embed_text(resume_text)
        new_resume = Resume(candidate_name=candidate_name, resume_text=resume_text, embedding=resume_embedding)
        db.session.add(new_resume)
        db.session.commit()

        flash("Resume uploaded and stored successfully!", "success")
        return redirect(url_for('upload_resume'))

    return render_template('upload_resume.html', form=form)

# Route for finding suitable candidates based on the generated job description
@app.route('/find_candidates', methods=['GET', 'POST'])
@login_required
def find_candidates():
    if request.method == 'POST':
        job_id = request.form.get('job_id')

        if not job_id:
            flash("Job ID is missing. Please try generating a job description again.", "warning")
            return redirect(url_for('chatbot'))

        job_desc = JobDescription.query.get(job_id)

        if not job_desc:
            flash("No job description found for matching. Please try again.", "danger")
            return redirect(url_for('chatbot'))

        job_description_embedding = job_desc.embedding

        if not job_description_embedding:
            flash("The job description embedding is missing. Please regenerate the job description.", "warning")
            return redirect(url_for('chatbot'))

        # Find the best matching candidate based on the job description embedding
        best_candidate = find_best_candidate(job_description_embedding)

        if best_candidate:
            flash(f"Best Candidate for the Job Description: {job_desc.job_description}", "success")
            flash(f"Candidate Name: {best_candidate.candidate_name}, Resume: {best_candidate.resume_text}", "info")
        else:
            flash("No matching candidates found.", "warning")

    return render_template('find_candidates.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()

    app.run(debug=True)
