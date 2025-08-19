import eventlet
eventlet.monkey_patch()

from flask import Flask, request, jsonify, session, send_file, abort, redirect, render_template_string
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room, disconnect
import os
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
import traceback
import secrets
import re
from pathlib import Path
import json
import hashlib
import mimetypes
import uuid
from werkzeug.utils import secure_filename

# Import models from separate file
from models import (
    db, bcrypt, User, StudyGroup, GroupMember, GroupMessage, GroupThread, GroupFile,
    UltraAcademicConversationHistory, UltraAcademicDocumentStats, 
    UltraQueryPerformanceLog, DocumentUploadSession
)

# Load environment variables
load_dotenv()

# Configure enhanced logging for academic content
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', secrets.token_hex(32))

from group_study import group_study_bp

# Register the group study blueprint
app.register_blueprint(group_study_bp)


socketio = SocketIO(app, 
                   cors_allowed_origins="*", 
                   async_mode='eventlet',
                   logger=False,
                   engineio_logger=False)


# Then declare your global variables
active_calls = {}
call_participants = {}

# Enhanced CORS configuration for academic environment
CORS(app,
     supports_credentials=True,
     origins=os.getenv('ALLOWED_ORIGINS', 'http://localhost,http://localhost:80,http://127.0.0.1:5000').split(','),
     methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization', 'X-Requested-With'])

database_url = os.getenv('DATABASE_URL')

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 300,
    'pool_timeout': 10,
    'connect_args': {
        'charset': 'utf8mb4',
        'connect_timeout': 30,
        'read_timeout': 30,
        'write_timeout': 30,
        'autocommit': False,  # Important for MySQL
    }
}

# Security configurations
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour
app.config['MAX_CONTENT_LENGTH'] = 150 * 1024 * 1024  # 150MB max file size

# Initialize extensions with app
db.init_app(app)
bcrypt.init_app(app)

# Admin credentials from environment
ADMIN_USERNAME = os.getenv('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', 'admin123')

# Initialize ultra-robust academic document processor
doc_processor = None
processor_error = None

try:
    # Import the enhanced ultra-robust academic document processor
    from document_processor import UltraRobustAcademicDocumentProcessor
    doc_processor = UltraRobustAcademicDocumentProcessor()
    logger.info("Ultra-Robust Academic Document Processor initialized successfully")
except Exception as e:
    processor_error = str(e)
    logger.error(f"Failed to initialize Ultra-Robust Academic Document Processor: {e}")
    logger.error(f"Traceback: {traceback.format_exc()}")

# Create necessary directories (including group_uploads for file sharing)
directories = ['uploads', 'vectorstore', 'logs', 'temp', 'academic_cache', 'user_sessions', 'group_uploads']
for directory in directories:
    os.makedirs(directory, exist_ok=True)
    try:
        os.chmod(directory, 0o755)
    except:
        pass  # Skip permission setting on Windows

# Create database tables
with app.app_context():
    try:
        db.create_all()
        logger.info("Enhanced academic database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")

def _get_session_id() -> str:
    """Return a stable session identifier for memory scoping."""
    if 'session_id' not in session:
        session['session_id'] = secrets.token_hex(16)
    return session['session_id']

def require_auth():
    """Check if user is authenticated"""
    if 'user_id' not in session and 'is_admin' not in session:
        return False
    return True

def get_current_user():
    """Get current user object - FIXED SQLAlchemy warning"""
    if session.get('is_admin'):
        return None  # Admin user, handle separately
    elif session.get('user_id'):
        return db.session.get(User, session['user_id'])  # FIXED: Using db.session.get()
    return None

def safe_filename(filename: str) -> str:
    """Create a safe filename without problematic characters"""
    filename = os.path.basename(filename)
    safe_chars = []
    for char in filename:
        if char.isalnum() or char in '.-_':
            safe_chars.append(char)
        else:
            safe_chars.append('_')
    safe_name = ''.join(safe_chars).strip('._')
    safe_name = re.sub(r'_+', '_', safe_name)
    if not safe_name:
        safe_name = 'unnamed_file'
    return safe_name[:100]

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f}{size_names[i]}"

def validate_file_type(filename: str) -> bool:
    """Validate if file type is supported"""
    allowed_extensions = {'.pdf', '.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt', '.txt', '.html', '.htm'}
    file_ext = os.path.splitext(filename)[1].lower()
    return file_ext in allowed_extensions

def allowed_file_for_sharing(filename):
    """Check if file is allowed for sharing in groups"""
    allowed_extensions = {
        # Images
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg',
        # Documents
        '.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt',
        '.xls', '.xlsx', '.ppt', '.pptx', '.csv',
        # Videos
        '.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv',
        # Audio
        '.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a',
        # Archives
        '.zip', '.rar', '.7z', '.tar', '.gz',
        # Code
        '.py', '.js', '.html', '.css', '.json', '.xml'
    }
    file_ext = os.path.splitext(filename)[1].lower()
    return file_ext in allowed_extensions

def get_file_type(filename):
    """Determine file type based on extension"""
    file_ext = os.path.splitext(filename)[1].lower()
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg'}
    video_extensions = {'.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv'}
    audio_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'}
    document_extensions = {'.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt', '.xls', '.xlsx', '.ppt', '.pptx'}
    
    if file_ext in image_extensions:
        return 'image'
    elif file_ext in video_extensions:
        return 'video'
    elif file_ext in audio_extensions:
        return 'audio'
    elif file_ext in document_extensions:
        return 'document'
    else:
        return 'other'

def create_group_upload_folder(group_id):
    """Create upload folder for a specific group"""
    folder_path = os.path.join('group_uploads', str(group_id))
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def detect_document_category_enhanced(filename: str, content_sample: str = "") -> str:
    """Enhanced academic document category detection"""
    filename_lower = filename.lower()
    content_lower = content_sample.lower()

    # Enhanced question paper indicators
    question_indicators = [
        'question', 'exam', 'test', 'quiz', 'assessment', 'assignment',
        'q.', 'que.', 'mcq', 'objective', 'subjective', 'marks', 'points'
    ]

    # Enhanced answer script indicators
    answer_indicators = [
        'answer', 'solution', 'ans.', 'sol.', 'key', 'solved', 'response',
        'explanation', 'result', 'outcome', 'resolution'
    ]

    # Enhanced textbook indicators
    textbook_indicators = [
        'textbook', 'book', 'chapter', 'unit', 'reference', 'manual',
        'guide', 'handbook', 'course material', 'study material'
    ]

    # Enhanced notes indicators
    notes_indicators = [
        'notes', 'lecture', 'summary', 'study', 'revision', 'review',
        'outline', 'synopsis', 'brief', 'memo'
    ]

    # Research/academic paper indicators
    research_indicators = [
        'research', 'paper', 'journal', 'article', 'thesis', 'dissertation',
        'report', 'analysis', 'study', 'investigation'
    ]

    # Filename-based detection with enhanced scoring
    scores = {
        'question_paper': 0,
        'answer_script': 0,
        'textbook': 0,
        'notes': 0,
        'research_paper': 0
    }

    # Score based on filename
    for indicator in question_indicators:
        if indicator in filename_lower:
            scores['question_paper'] += 2
    for indicator in answer_indicators:
        if indicator in filename_lower:
            scores['answer_script'] += 2
    for indicator in textbook_indicators:
        if indicator in filename_lower:
            scores['textbook'] += 2
    for indicator in notes_indicators:
        if indicator in filename_lower:
            scores['notes'] += 2
    for indicator in research_indicators:
        if indicator in filename_lower:
            scores['research_paper'] += 2

    # Content-based detection with enhanced patterns
    if content_sample:
        # Question patterns in content
        question_patterns = [
            r'\bq\.?\s*\d+', r'\bquestion\s*\d+', r'\b\d+\s*marks?\b',
            r'\bwhat\s+is\b', r'\bwhy\s+is\b', r'\bhow\s+does\b',
            r'\bexplain\b', r'\bdiscuss\b', r'\bdefine\b', r'\bcompare\b'
        ]

        for pattern in question_patterns:
            if re.search(pattern, content_lower):
                scores['question_paper'] += 1

        # Answer patterns in content
        answer_patterns = [
            r'\banswer\s*:?', r'\bsolution\s*:?', r'\bexplanation\s*:?',
            r'\bsteps?\s*:?', r'\bmethod\s*:?', r'\bapproach\s*:?'
        ]

        for pattern in answer_patterns:
            if re.search(pattern, content_lower):
                scores['answer_script'] += 1

        # Academic structure indicators
        if any(term in content_lower for term in ['chapter', 'section', 'definition', 'theorem']):
            scores['textbook'] += 1
        if any(term in content_lower for term in ['abstract', 'methodology', 'conclusion', 'references']):
            scores['research_paper'] += 1

    # Return category with highest score
    if max(scores.values()) > 0:
        return max(scores, key=scores.get)
    return 'general'

def extract_subject_enhanced(filename: str, content_sample: str = "") -> str:
    """Enhanced subject extraction from filename and content"""
    # Enhanced academic subjects mapping
    subjects_mapping = {
        'cloud computing': ['cloud', 'aws', 'azure', 'virtualization', 'saas', 'paas', 'iaas'],
        'machine learning': ['ml', 'machine learning', 'artificial intelligence', 'ai', 'neural', 'deep learning'],
        'data structures': ['data structure', 'algorithm', 'sorting', 'searching', 'tree', 'graph', 'stack', 'queue'],
        'database management': ['database', 'dbms', 'sql', 'nosql', 'mongodb', 'mysql', 'postgresql'],
        'computer networks': ['network', 'networking', 'tcp', 'ip', 'protocol', 'router', 'switch'],
        'operating systems': ['os', 'operating system', 'linux', 'windows', 'unix', 'kernel'],
        'software engineering': ['software', 'engineering', 'sdlc', 'agile', 'scrum', 'testing'],
        'web development': ['web', 'html', 'css', 'javascript', 'react', 'node', 'angular'],
        'cybersecurity': ['security', 'cyber', 'encryption', 'firewall', 'malware', 'vulnerability'],
        'data science': ['data science', 'analytics', 'statistics', 'visualization', 'pandas', 'numpy'],
        'mathematics': ['math', 'calculus', 'algebra', 'geometry', 'statistics', 'probability'],
        'physics': ['physics', 'mechanics', 'thermodynamics', 'electromagnetism', 'quantum'],
        'chemistry': ['chemistry', 'organic', 'inorganic', 'physical chemistry', 'biochemistry'],
        'biology': ['biology', 'genetics', 'ecology', 'molecular', 'cell biology'],
        'english': ['english', 'literature', 'grammar', 'writing', 'communication']
    }

    combined_text = f"{filename} {content_sample}".lower()

    # Score each subject
    subject_scores = {}
    for subject, keywords in subjects_mapping.items():
        score = 0
        for keyword in keywords:
            score += combined_text.count(keyword)
        if score > 0:
            subject_scores[subject] = score

    # Return subject with highest score
    if subject_scores:
        return max(subject_scores, key=subject_scores.get).title()

    # Fallback: try to extract from filename patterns
    patterns = [
        r'([a-zA-Z\s]+)(?:_|-|\s)(?:exam|test|question|answer|notes|book)',
        r'([a-zA-Z\s]+)(?:_|-|\s)(?:chapter|unit|module)',
        r'([a-zA-Z\s]+)(?:_|-|\s)(?:tutorial|assignment|homework)'
    ]

    for pattern in patterns:
        match = re.search(pattern, filename.lower())
        if match:
            subject = match.group(1).strip().title()
            if len(subject) > 2:
                return subject

    return 'General Academic'

def calculate_query_hash(query: str) -> str:
    """Calculate hash for query deduplication"""
    return hashlib.sha256(query.encode()).hexdigest()[:64]

# Security middleware
@app.before_request
def before_request():
    """Enhanced security checks before each request"""
    if request.method == 'OPTIONS':
        return

@app.after_request
def after_request(response):
    """Add enhanced security headers after request"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['X-Academic-System'] = 'Ultra-Robust-Academic-Chatbot'
    return response

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found', 'system': 'ultra-robust-academic'}), 404

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 150MB', 'system': 'ultra-robust-academic'}), 413

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error', 'system': 'ultra-robust-academic'}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {e}\n{traceback.format_exc()}")
    return jsonify({'error': 'An unexpected error occurred', 'system': 'ultra-robust-academic'}), 500

# AUTHENTICATION ROUTES
@app.route('/api/auth/register', methods=['POST'])
def api_register():
    """Enhanced registration for ultra-academic users"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'}), 400

        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        full_name = data.get('full_name', '').strip()
        role = data.get('role', 'student').strip()
        institution = data.get('institution', '').strip()
        course = data.get('course', '').strip()
        academic_year = data.get('academic_year', '').strip()
        subjects_of_interest = data.get('subjects_of_interest', [])

        # Enhanced validation
        if not all([username, email, password, full_name]):
            return jsonify({'success': False, 'message': 'Required fields: username, email, password, full_name'}), 400

        if len(username) < 3 or len(username) > 30:
            return jsonify({'success': False, 'message': 'Username must be 3-30 characters long'}), 400

        if len(password) < 8:
            return jsonify({'success': False, 'message': 'Password must be at least 8 characters long'}), 400

        if role not in ['student', 'teacher']:
            return jsonify({'success': False, 'message': 'Role must be student or teacher'}), 400

        # Email validation
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_regex, email):
            return jsonify({'success': False, 'message': 'Invalid email format'}), 400

        # Check for existing users
        if User.query.filter_by(username=username).first():
            return jsonify({'success': False, 'message': 'Username already exists'}), 409

        if User.query.filter_by(email=email).first():
            return jsonify({'success': False, 'message': 'Email already registered'}), 409

        # Create new ultra-academic user
        user = User(
            username=username,
            email=email,
            full_name=full_name,
            role=role,
            institution=institution,
            course=course,
            academic_year=academic_year,
            subjects_of_interest=json.dumps(subjects_of_interest) if subjects_of_interest else None
        )

        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        logger.info(f"New ultra-academic {role} registered: {username}")

        return jsonify({
            'success': True,
            'message': 'Ultra-academic account created successfully',
            'user': user.to_dict()
        }), 201

    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({'success': False, 'message': 'Registration failed. Please try again.'}), 500

@app.route('/api/auth/login', methods=['POST'])
def api_login():
    """Enhanced authentication for ultra-academic users"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'}), 400

        role = data.get('role', '').strip()
        username = data.get('username', '').strip()
        password = data.get('password', '')

        if not username or not password:
            return jsonify({'success': False, 'message': 'Username and password required'}), 400

        if role == 'admin':
            if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                session.permanent = True
                session['is_admin'] = True
                session['admin_username'] = username
                session['login_time'] = datetime.utcnow().isoformat()
                session['role'] = 'admin'
                _get_session_id()

                logger.info(f"Ultra-academic admin login successful: {username}")

                return jsonify({
                    'success': True,
                    'role': 'admin',
                    'username': username,
                    'message': 'Ultra-academic admin login successful',
                    'system': 'ultra-robust-academic'
                })
            else:
                logger.warning(f"Failed ultra-academic admin login attempt: {username}")
                return jsonify({'success': False, 'message': 'Invalid admin credentials'}), 401

        else:  # student or teacher
            user = User.query.filter_by(username=username, is_active=True).first()

            if not user:
                logger.warning(f"Login attempt for non-existent ultra-academic user: {username}")
                return jsonify({'success': False, 'message': 'Invalid username or password'}), 401

            if user.locked_until and datetime.utcnow() < user.locked_until:
                return jsonify({'success': False, 'message': 'Account temporarily locked'}), 423

            if user.check_password(password):
                user.reset_login_attempts()
                user.last_login = datetime.utcnow()
                db.session.commit()

                session.permanent = True
                session['user_id'] = user.id
                session['username'] = user.username
                session['role'] = user.role
                session['login_time'] = datetime.utcnow().isoformat()
                _get_session_id()

                logger.info(f"Ultra-academic {user.role} login successful: {username}")

                return jsonify({
                    'success': True,
                    'role': user.role,
                    'user': user.to_dict(),
                    'message': f'Ultra-academic {user.role} login successful',
                    'system': 'ultra-robust-academic'
                })
            else:
                user.login_attempts += 1
                if user.login_attempts >= 5:
                    user.lock_account()
                    logger.warning(f"Ultra-academic account locked due to multiple failed attempts: {username}")
                db.session.commit()

                logger.warning(f"Failed ultra-academic login attempt: {username}")
                return jsonify({'success': False, 'message': 'Invalid username or password'}), 401

    except Exception as e:
        logger.error(f"Ultra-academic login error: {e}")
        return jsonify({'success': False, 'message': 'Login failed. Please try again.'}), 500

@app.route('/api/auth/logout', methods=['POST'])
def api_logout():
    """Enhanced logout """
    try:
        username = session.get('username') or session.get('admin_username')
        role = session.get('role', 'unknown')

        session.clear()

        if username:
            logger.info(f"Ultra-academic {role} logged out: {username}")

        return jsonify({
            'success': True,
            'message': 'Ultra-academic logout successful',
            'system': 'ultra-robust-academic'
        })

    except Exception as e:
        logger.error(f"Ultra-academic logout error: {e}")
        return jsonify({'success': False, 'message': 'Logout failed'}), 500

@app.route('/api/auth/status', methods=['GET'])
def api_auth_status():
    """Check ultra-academic authentication status - FIXED user display issue"""
    try:
        if session.get('is_admin'):
            return jsonify({
                'authenticated': True,
                'role': 'admin',
                'username': session.get('admin_username'),
                'user': {
                    'username': session.get('admin_username'),
                    'full_name': 'Administrator',
                    'role': 'admin'
                },
                'login_time': session.get('login_time'),
                'session_id': session.get('session_id'),
                'system': 'ultra-robust-academic'
            })

        elif session.get('user_id'):
            user = db.session.get(User, session['user_id'])  # FIXED: Using db.session.get()
            if user and user.is_active:
                return jsonify({
                    'authenticated': True,
                    'role': user.role,
                    'username': user.username,  # FIXED: Added direct username
                    'user': user.to_dict(),
                    'login_time': session.get('login_time'),
                    'session_id': session.get('session_id'),
                    'system': 'ultra-robust-academic'
                })

        return jsonify({'authenticated': False, 'system': 'ultra-robust-academic'}), 401

    except Exception as e:
        logger.error(f"Ultra-academic auth status error: {e}")
        return jsonify({'authenticated': False, 'system': 'ultra-robust-academic'}), 401

# FILE MANAGEMENT ROUTES (For Admin Processing)
@app.route('/api/files/upload', methods=['POST'])
def api_upload_files():
    """Ultra-enhanced file upload with advanced academic document classification"""
    if not session.get('is_admin') and session.get('role') != 'teacher':
        return jsonify({'error': 'Unauthorized. Admin or teacher access required.'}), 403

    if not doc_processor:
        return jsonify({'error': 'Ultra-robust document processor unavailable'}), 503

    if 'files' not in request.files:
        return jsonify({'error': 'No files selected'}), 400

    files = request.files.getlist('files')

    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No files selected'}), 400

    # Create upload session
    upload_session = DocumentUploadSession(
        session_id=_get_session_id(),
        user_id=session.get('user_id'),
        total_files=len(files)
    )

    db.session.add(upload_session)
    db.session.commit()

    uploaded_files = []
    errors = []
    max_file_size = 150 * 1024 * 1024  # 150MB limit

    for file in files:
        if file.filename == '':
            continue

        try:
            # Validate file size
            file.seek(0, 2)
            file_size = file.tell()
            file.seek(0)

            if file_size > max_file_size:
                error_msg = f'File {file.filename} exceeds size limit (150MB)'
                errors.append(error_msg)
                upload_session.failed_files += 1
                continue

            # Validate file type
            if not validate_file_type(file.filename):
                file_ext = os.path.splitext(file.filename)[1].lower()
                error_msg = f'File type {file_ext} not supported for {file.filename}'
                errors.append(error_msg)
                upload_session.failed_files += 1
                continue

            # Create safe filename
            safe_name = safe_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
            filename = f"{timestamp}{safe_name}"
            filepath = os.path.join('uploads', filename)

            # Save file
            file.save(filepath)

            # Read sample content for enhanced classification
            try:
                sample_content = ""
                if file.filename.lower().endswith('.txt'):
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        sample_content = f.read(2000)  # Read first 2000 chars
                elif file.filename.lower().endswith(('.docx', '.doc')):
                    # Quick DOCX sample extraction
                    try:
                        import docx
                        doc = docx.Document(filepath)
                        sample_content = " ".join([para.text for para in doc.paragraphs[:10]])
                    except:
                        pass
            except:
                sample_content = ""

            # Enhanced document category and subject detection
            doc_category = detect_document_category_enhanced(file.filename, sample_content)
            subject = extract_subject_enhanced(file.filename, sample_content)

            # Process document with ultra-robust processor
            processing_start = datetime.now()
            result = doc_processor.process_document(filepath, filename)
            processing_time = (datetime.now() - processing_start).total_seconds()

            if result.success:
                # Calculate enhanced academic relevance score
                academic_relevance = 0.8  # Default high score for academic content
                if doc_category == 'question_paper':
                    academic_relevance = 0.95
                elif doc_category == 'answer_script':
                    academic_relevance = 0.90
                elif doc_category == 'textbook':
                    academic_relevance = 0.85
                elif doc_category == 'research_paper':
                    academic_relevance = 0.80

                # Count academic patterns
                academic_patterns = 0
                question_patterns = 0
                answer_patterns = 0
                academic_terms = 0

                if sample_content:
                    # Enhanced pattern counting
                    question_patterns = len(re.findall(r'\b(?:question|q\.?)\s*\d+|\bmarks?\b|\bpoints?\b', sample_content.lower()))
                    answer_patterns = len(re.findall(r'\banswer\s*:?|\bsolution\s*:?|\bexplanation\s*:?', sample_content.lower()))
                    academic_terms = len(re.findall(r'\b(?:chapter|unit|module|lecture|course|study|exam|test)\b', sample_content.lower()))
                    academic_patterns = question_patterns + answer_patterns + academic_terms

                # Save ultra-enhanced processing stats
                try:
                    stats = UltraAcademicDocumentStats(
                        filename=filename,
                        original_filename=file.filename,
                        file_type=result.document_type,
                        document_category=doc_category,
                        subject_detected=subject,
                        academic_relevance_score=academic_relevance,
                        file_size=file_size,
                        processing_time=processing_time,
                        chunks_created=result.chunks_added,
                        content_length=result.metadata.get('content_length', 0),
                        extraction_methods=','.join(result.extraction_methods),
                        chunk_types_used=json.dumps(list(set([chunk.metadata.get('chunk_type', 'unknown') for chunk in getattr(result, 'chunks', [])]))),
                        academic_patterns_detected=academic_patterns,
                        question_patterns_found=question_patterns,
                        answer_patterns_found=answer_patterns,
                        academic_terms_count=academic_terms,
                        parallel_processing_used=result.parallel_processing_used,
                        thread_count=result.thread_count,
                        embedding_model=result.metadata.get('embedding_model', 'text-embedding-3-large'),
                        embedding_dimensions=result.metadata.get('embedding_dimensions', 3072),
                        processed_by=session.get('user_id')
                    )

                    db.session.add(stats)
                    upload_session.processed_files += 1
                    upload_session.total_chunks += result.chunks_added
                    upload_session.total_processing_time += processing_time

                    if result.academic_content_detected:
                        upload_session.academic_documents_detected += 1

                    db.session.commit()

                except Exception as stats_error:
                    logger.error(f"Failed to save ultra-academic processing stats: {stats_error}")

                uploaded_files.append({
                    'filename': filename,
                    'original_filename': file.filename,
                    'size': file_size,
                    'size_formatted': format_file_size(file_size),
                    'processed': True,
                    'chunks_added': result.chunks_added,
                    'document_type': result.document_type,
                    'document_category': doc_category,
                    'subject_detected': subject,
                    'academic_relevance_score': academic_relevance,
                    'processing_time': processing_time,
                    'extraction_methods': result.extraction_methods,
                    'academic_patterns_detected': academic_patterns,
                    'question_patterns_found': question_patterns,
                    'answer_patterns_found': answer_patterns,
                    'academic_terms_count': academic_terms,
                    'parallel_processing_used': result.parallel_processing_used,
                    'thread_count': result.thread_count,
                    'embedding_model': result.metadata.get('embedding_model', 'text-embedding-3-large'),
                    'embedding_dimensions': result.metadata.get('embedding_dimensions', 3072),
                    'academic_content_detected': result.academic_content_detected
                })

            else:
                error_msg = f'Processing failed for {file.filename}: {result.error}'
                errors.append(error_msg)
                upload_session.failed_files += 1

                # Remove failed file
                if os.path.exists(filepath):
                    os.remove(filepath)

            logger.info(f"Ultra-academic file processing completed: {filename} - Success: {result.success} - Category: {doc_category} - Subject: {subject}")

        except Exception as e:
            error_msg = f'Error processing {file.filename}: {str(e)}'
            errors.append(error_msg)
            upload_session.failed_files += 1
            logger.error(error_msg)

            # Clean up file if processing failed
            if 'filepath' in locals() and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception as cleanup_error:
                    logger.error(f"Failed to cleanup file {filepath}: {cleanup_error}")

    # Update upload session
    upload_session.status = 'completed'
    upload_session.completed_at = datetime.utcnow()
    upload_session.error_messages = json.dumps(errors) if errors else None
    db.session.commit()

    return jsonify({
        'success': True,
        'message': f'Successfully uploaded and processed {len(uploaded_files)} ultra-academic files',
        'uploaded_files': uploaded_files,
        'errors': errors,
        'upload_session': {
            'total_files': upload_session.total_files,
            'processed_files': upload_session.processed_files,
            'failed_files': upload_session.failed_files,
            'total_chunks': upload_session.total_chunks,
            'total_processing_time': upload_session.total_processing_time,
            'academic_documents_detected': upload_session.academic_documents_detected
        },
        'system_stats': doc_processor.get_system_stats() if doc_processor else {},
        'system': 'ultra-robust-academic'
    })

@app.route('/api/files/list', methods=['GET'])
def api_get_files():
    """Ultra-enhanced file listing with comprehensive academic metadata"""
    if 'user_id' not in session and 'is_admin' not in session:
        return jsonify({'error': 'Unauthorized'}), 403

    try:
        files_info = []
        uploads_dir = 'uploads'

        if os.path.exists(uploads_dir):
            for filename in os.listdir(uploads_dir):
                filepath = os.path.join(uploads_dir, filename)
                if os.path.isfile(filepath):
                    try:
                        stat = os.stat(filepath)
                        in_vectorstore = doc_processor.is_file_in_vectorstore(filename) if doc_processor else False

                        # Get ultra-academic processing stats from database
                        ultra_stats = UltraAcademicDocumentStats.query.filter_by(filename=filename).first()

                        file_info = {
                            'name': filename,
                            'size': stat.st_size,
                            'size_formatted': format_file_size(stat.st_size),
                            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            'type': filename.split('.')[-1].lower() if '.' in filename else 'unknown',
                            'in_vectorstore': in_vectorstore,
                            'can_delete': session.get('is_admin', False) or session.get('role') == 'teacher'
                        }

                        if ultra_stats:
                            file_info.update({
                                'original_filename': ultra_stats.original_filename,
                                'document_category': ultra_stats.document_category,
                                'subject_detected': ultra_stats.subject_detected,
                                'academic_relevance_score': ultra_stats.academic_relevance_score,
                                'processing_time': ultra_stats.processing_time,
                                'chunks_created': ultra_stats.chunks_created,
                                'content_length': ultra_stats.content_length,
                                'extraction_methods': ultra_stats.extraction_methods.split(',') if ultra_stats.extraction_methods else [],
                                'chunk_types_used': json.loads(ultra_stats.chunk_types_used) if ultra_stats.chunk_types_used else [],
                                'academic_patterns_detected': ultra_stats.academic_patterns_detected,
                                'question_patterns_found': ultra_stats.question_patterns_found,
                                'answer_patterns_found': ultra_stats.answer_patterns_found,
                                'academic_terms_count': ultra_stats.academic_terms_count,
                                'parallel_processing_used': ultra_stats.parallel_processing_used,
                                'thread_count': ultra_stats.thread_count,
                                'embedding_model': ultra_stats.embedding_model,
                                'embedding_dimensions': ultra_stats.embedding_dimensions,
                                'processed_at': ultra_stats.processed_at.isoformat()
                            })

                        files_info.append(file_info)

                    except Exception as e:
                        logger.error(f"Error getting ultra-academic file info for {filename}: {e}")
                        continue

        # Enhanced file statistics
        category_stats = {}
        subject_stats = {}
        for file_info in files_info:
            category = file_info.get('document_category', 'unknown')
            subject = file_info.get('subject_detected', 'unknown')
            category_stats[category] = category_stats.get(category, 0) + 1  
            subject_stats[subject] = subject_stats.get(subject, 0) + 1

        return jsonify({
            'success': True,
            'files': sorted(files_info, key=lambda x: x['modified'], reverse=True),
            'total_files': len(files_info),
            'category_distribution': category_stats,
            'subject_distribution': subject_stats,
            'system_stats': doc_processor.get_system_stats() if doc_processor else {},
            'system': 'ultra-robust-academic'
        })

    except Exception as e:
        logger.error(f"Error listing ultra-academic files: {e}")
        return jsonify({'error': 'Failed to list files', 'system': 'ultra-robust-academic'}), 500

@app.route('/api/files/delete', methods=['DELETE'])
def api_delete_file():
    """Enhanced file deletion with ultra-academic cleanup"""
    if not session.get('is_admin') and session.get('role') != 'teacher':
        return jsonify({'error': 'Unauthorized. Admin or teacher access required.'}), 403
    
    if not doc_processor:
        return jsonify({'error': 'Ultra-robust document processor unavailable'}), 503
    
    try:
        data = request.get_json()
        filename = data.get('filename', '').strip()
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        filepath = os.path.join('uploads', filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        # Remove from ultra-robust vectorstore
        removed_count = doc_processor.remove_document(filename)
        
        # Remove physical file
        os.remove(filepath)
        
        # Remove ultra-academic processing stats from database
        try:
            UltraAcademicDocumentStats.query.filter_by(filename=filename).delete()
            db.session.commit()
        except Exception as db_error:
            logger.error(f"Failed to remove ultra-academic processing stats: {db_error}")
        
        logger.info(f"Ultra-academic file deleted successfully: {filename}")
        
        return jsonify({
            'success': True,
            'message': f'Ultra-academic file "{filename}" deleted successfully',
            'chunks_removed': removed_count,
            'system_stats': doc_processor.get_system_stats() if doc_processor else {},
            'system': 'ultra-robust-academic'
        })

    except Exception as e:
        logger.error(f'Error deleting ultra-academic file: {e}')
        return jsonify({'error': f'Error deleting file: {str(e)}'}), 500

# CHAT ROUTE
@app.route('/api/chat', methods=['POST'])
def api_chat():
    """Ultra-enhanced chat endpoint with advanced academic query processing"""
    if 'user_id' not in session and 'is_admin' not in session:
        return jsonify({'error': 'Unauthorized'}), 403

    if not doc_processor:
        return jsonify({'error': 'Ultra-robust document processor unavailable'}), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        question = data.get('question', '').strip()
        context_mode = data.get('context_mode', 'ultra_academic')

        if not question:
            return jsonify({'error': 'No question provided'}), 400

        if len(question) > 2000:
            return jsonify({'error': 'Question too long (max 2000 characters)'}), 400

        start_time = datetime.now()
        query_hash = calculate_query_hash(question)

        # Generate response with ultra-robust academic processor
        response_data = doc_processor.generate_response(
            query=question,
            user_context={
                'user_id': session.get('user_id'),
                'username': session.get('username') or session.get('admin_username'),
                'role': session.get('role', 'user'),
                'session_id': _get_session_id(),
                'context_mode': context_mode
            }
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        # Extract ultra-academic information
        analysis = response_data.get('analysis', {})
        academic_type = analysis.get('academic_type', 'general')
        complexity_level = analysis.get('complexity_level', 'moderate')
        expected_length = analysis.get('expected_length', 'moderate')

        # Save ultra-enhanced conversation history
        try:
            conversation = UltraAcademicConversationHistory(
                user_id=session.get('user_id'),
                session_id=session.get('session_id', 'anonymous'),
                query_text=question,
                query_hash=query_hash,
                response=response_data['response'],
                query_type=response_data.get('query_type'),
                academic_type=academic_type,
                complexity_level=complexity_level,
                expected_length=expected_length,
                confidence_score=response_data.get('confidence_score'),
                processing_time=processing_time,
                sources_used=json.dumps(response_data.get('sources_used', [])),
                search_methods=json.dumps(analysis.get('search_methods_used', [])),
                academic_relevance=analysis.get('academic_relevance', 0.0),
                follow_up_context=analysis.get('follow_up_context', False),
                key_concepts=json.dumps(analysis.get('key_concepts', [])),
                max_tokens_used=analysis.get('max_tokens_used', 0),
                documents_searched=analysis.get('documents_searched', 0)
            )

            db.session.add(conversation)
            db.session.commit()

        except Exception as e:
            logger.error(f"Failed to save ultra-academic conversation history: {e}")

        # Log ultra-enhanced query performance
        try:
            perf_log = UltraQueryPerformanceLog(
                query_text=question,
                query_hash=query_hash,
                academic_type=academic_type,
                complexity_level=complexity_level,
                search_strategies_used=json.dumps(analysis.get('search_methods_used', [])),
                results_found=len(response_data.get('sources_used', [])),
                unique_sources=len(set(response_data.get('sources_used', []))),
                processing_time=processing_time,
                confidence_score=response_data.get('confidence_score', 0.0),
                academic_relevance_avg=analysis.get('academic_relevance_avg', 0.0),
                session_id=session.get('session_id'),
                user_id=session.get('user_id')
            )

            db.session.add(perf_log)
            db.session.commit()

        except Exception as e:
            logger.error(f"Failed to log ultra-academic query performance: {e}")

        return jsonify({
            'success': True,
            'response': response_data['response'],
            'metadata': {
                'query_type': response_data.get('query_type'),
                'academic_type': academic_type,
                'complexity_level': complexity_level,
                'expected_length': expected_length,
                'confidence_score': response_data.get('confidence_score'),
                'sources_used': response_data.get('sources_used', []),
                'processing_time': processing_time,
                'context_mode': context_mode,
                'analysis': analysis,
                'query_hash': query_hash
            },
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'system_stats': doc_processor.get_system_stats() if doc_processor else {},
            'system': 'ultra-robust-academic'
        })

    except Exception as e:
        logger.error(f'Error generating ultra-academic response: {e}')
        return jsonify({
            'error': f'Error generating response: {str(e)}',
            'system': 'ultra-robust-academic'
        }), 500

@app.route('/api/chat/history', methods=['GET'])
def api_get_chat_history():
    try:
        session_id = _get_session_id()
        user_id = session.get('user_id')
        
        # Query chat history for current session/user
        query = UltraAcademicConversationHistory.query
        
        if user_id:
            query = query.filter_by(user_id=user_id)
        else:
            query = query.filter_by(session_id=session_id)
        
        # Get recent conversations (last 50)
        conversations = query.order_by(
            UltraAcademicConversationHistory.created_at.desc()
        ).limit(50).all()  # ← Make sure .all() is called here
        
        # Convert to chat format
        chat_history = []
        for conv in reversed(conversations):
            # Add user message
            chat_history.append({
                'id': f"user_{conv.id}",
                'type': 'user',
                'message': conv.query_text,  # ← Changed from conv.query to conv.query_text
                'timestamp': conv.created_at.isoformat(),
                'query_type': conv.query_type,
                'academic_type': conv.academic_type
            })
            
            # Add AI response
            chat_history.append({
                'id': f"ai_{conv.id}",
                'type': 'ai',
                'message': conv.response,
                'timestamp': conv.created_at.isoformat(),
                'confidence_score': conv.confidence_score,
                'processing_time': conv.processing_time,
                'sources_used': json.loads(conv.sources_used) if conv.sources_used else []
            })
        
        return jsonify({
            'success': True,
            'chat_history': chat_history,
            'total_conversations': len(conversations),
            'system': 'ultra-robust-academic'
        })

    except Exception as e:
        logger.error(f"Error retrieving chat history: {e}")
        return jsonify({'error': 'Failed to retrieve chat history'}), 500



@app.route('/api/chat/clear', methods=['DELETE'])
def api_clear_chat_history():
    try:
        session_id = _get_session_id()
        user_id = session.get('user_id')
        
        # Use db.session.query() instead of Model.query
        if user_id:
            deleted_count = db.session.query(UltraAcademicConversationHistory).filter_by(
                user_id=user_id
            ).delete()
        else:
            deleted_count = db.session.query(UltraAcademicConversationHistory).filter_by(
                session_id=session_id
            ).delete()
        
        # Clear processor memory
        if doc_processor:
            doc_processor.clear_conversation_memory(session_id)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Cleared {deleted_count} conversations',
            'deleted_count': deleted_count,
            'system': 'ultra-robust-academic'
        })

    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        return jsonify({'error': 'Failed to clear chat history'}), 500



@app.route('/api/chat/stats', methods=['GET'])
def api_get_chat_stats():
    """Get chat statistics for current user"""
    if 'user_id' not in session and 'is_admin' not in session:
        return jsonify({'error': 'Unauthorized'}), 403

    try:
        user_id = session.get('user_id')
        session_id = _get_session_id()
        
        # Get conversation count
        if user_id:
            total_conversations = UltraAcademicConversationHistory.query.filter_by(user_id=user_id).count()
            recent_conversations = UltraAcademicConversationHistory.query.filter(
                UltraAcademicConversationHistory.user_id == user_id,
                UltraAcademicConversationHistory.created_at >= datetime.utcnow() - timedelta(days=7)
            ).count()
        else:
            total_conversations = UltraAcademicConversationHistory.query.filter_by(session_id=session_id).count()
            recent_conversations = UltraAcademicConversationHistory.query.filter(
                UltraAcademicConversationHistory.session_id == session_id,
                UltraAcademicConversationHistory.created_at >= datetime.utcnow() - timedelta(days=7)
            ).count()
        
        return jsonify({
            'success': True,
            'stats': {
                'total_conversations': total_conversations,
                'recent_conversations': recent_conversations,
                'session_id': session_id
            },
            'system': 'ultra-robust-academic'
        })

    except Exception as e:
        logger.error(f"Error getting chat stats: {e}")
        return jsonify({'error': 'Failed to get chat statistics'}), 500


@app.route('/api/chat/feedback', methods=['POST'])
def api_chat_feedback():
    """Enhanced user feedback collection for ultra-academic improvement"""
    if 'user_id' not in session and 'is_admin' not in session:
        return jsonify({'error': 'Unauthorized'}), 403

    try:
        data = request.get_json()
        query_hash = data.get('query_hash')
        feedback = data.get('feedback')  # helpful, not_helpful, partially_helpful
        satisfaction = data.get('satisfaction')  # satisfied, neutral, dissatisfied

        if not query_hash or feedback not in ['helpful', 'not_helpful', 'partially_helpful']:
            return jsonify({'error': 'Invalid feedback data'}), 400

        # Update ultra-academic conversation history
        conversation = UltraAcademicConversationHistory.query.filter_by(query_hash=query_hash).first()
        if conversation:
            conversation.user_feedback = feedback

        # Update ultra-academic query performance log
        perf_log = UltraQueryPerformanceLog.query.filter_by(query_hash=query_hash).first()
        if perf_log:
            perf_log.user_satisfaction = satisfaction or feedback

        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'Ultra-academic feedback recorded',
            'system': 'ultra-robust-academic'
        })

    except Exception as e:
        logger.error(f"Error recording ultra-academic feedback: {e}")
        return jsonify({'error': 'Failed to record feedback'}), 500

@app.route('/api/vectorstore/rebuild', methods=['POST'])
def api_rebuild_vectorstore():
    """Ultra-enhanced vectorstore rebuild"""
    if not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized. Admin access required.'}), 403
    
    if not doc_processor:
        return jsonify({'error': 'Ultra-robust document processor unavailable'}), 503
    
    try:
        rebuild_start = datetime.now()
        result = doc_processor.rebuild_vectorstore()
        rebuild_time = (datetime.now() - rebuild_start).total_seconds()
        
        logger.info(f"Ultra-academic vectorstore rebuilt successfully in {rebuild_time:.2f} seconds")
        
        return jsonify({
            'success': True,
            'message': 'Ultra-academic vectorstore rebuilt successfully',
            'processed_files': result['processed_files'],
            'total_chunks': result['total_chunks'],
            'processing_time': result.get('processing_time', rebuild_time),
            'errors': result['errors'],
            'embedding_model': result.get('embedding_model', 'text-embedding-3-large'),
            'embedding_dimensions': result.get('embedding_dimensions', 3072),
            'academic_enhanced': result.get('academic_enhanced', True),
            'system': 'ultra-robust-academic'
        })
    
    except Exception as e:
        logger.error(f'Error rebuilding ultra-academic vectorstore: {e}')
        return jsonify({'error': f'Error rebuilding vectorstore: {str(e)}'}), 500



# ============= WEBSOCKET EVENT HANDLERS =============

@socketio.on('connect')
def handle_connect():
    print(f'Client connected: {request.sid}')
    
    # Get user info if authenticated
    current_user = get_current_user()
    if current_user:
        print(f'User {current_user.username} connected with socket ID: {request.sid}')

@socketio.on('disconnect')
def handle_disconnect():
    print(f'Client disconnected: {request.sid}')
    
@socketio.on('join-group-room')
def handle_join_group_room(data):
    """Join a group room for real-time messaging"""
    try:
        group_id = data.get('group_id')
        user_id = session.get('user_id')
        
        if not user_id or not group_id:
            emit('error', {'message': 'Invalid request'})
            return
        
        # Verify user is a member of the group
        from models import StudyGroup
        group = StudyGroup.query.get(group_id)
        if not group or not group.is_member(user_id):
            emit('error', {'message': 'Access denied'})
            return
        
        room = f'group_{group_id}'
        join_room(room)
        
        # Get user info
        current_user = get_current_user()
        if current_user:
            # Notify others in the group
            emit('user-joined-group', {
                'user_id': current_user.id,
                'username': current_user.username,
                'full_name': current_user.full_name,
                'timestamp': datetime.now().isoformat()
            }, room=room, include_self=False)
        
        # Confirm successful join
        emit('group-room-joined', {
            'group_id': group_id,
            'room': room,
            'message': 'Successfully joined group room'
        })
        
        print(f'User {current_user.username if current_user else user_id} joined group room {room}')
        
    except Exception as e:
        logger.error(f"Error joining group room: {e}")
        emit('error', {'message': 'Failed to join group room'})

@socketio.on('leave-group-room')
def handle_leave_group_room(data):
    """Leave a group room"""
    try:
        group_id = data.get('group_id')
        room = f'group_{group_id}'
        leave_room(room)
        
        current_user = get_current_user()
        if current_user:
            # Notify others in the group
            emit('user-left-group', {
                'user_id': current_user.id,
                'username': current_user.username,
                'timestamp': datetime.now().isoformat()
            }, room=room)
            
            print(f'User {current_user.username} left group room {room}')
        
    except Exception as e:
        logger.error(f"Error leaving group room: {e}")

@socketio.on('send-group-message')
def handle_group_message(data):
    """Handle real-time group messages - NO POLLING"""
    try:
        group_id = data.get('group_id')
        thread_id = data.get('thread_id')
        message_text = data.get('message', '').strip()
        parent_message_id = data.get('parent_message_id')
        
        if not message_text or not group_id or not thread_id:
            emit('error', {'message': 'Invalid message data'})
            return
        
        if len(message_text) > 4000:
            emit('error', {'message': 'Message too long (max 4000 characters)'})
            return
        
        current_user = get_current_user()
        if not current_user:
            emit('error', {'message': 'User not found'})
            return
        
        # Verify access
        from models import StudyGroup, GroupThread, GroupMessage
        group = StudyGroup.query.get(group_id)
        if not group or not group.is_member(current_user.id):
            emit('error', {'message': 'Access denied'})
            return
        
        thread = GroupThread.query.get(thread_id)
        if not thread or thread.group_id != group_id:
            emit('error', {'message': 'Thread not found'})
            return
        
        # Use shared GPT processing function
        from group_study import process_group_gpt_query
        is_gpt_query, gpt_response = process_group_gpt_query(
            message_text, group, thread, current_user
        )
        
        # Create the message
        message = GroupMessage(
            group_id=group_id,
            thread_id=thread_id,
            user_id=current_user.id,
            message=message_text,
            parent_message_id=parent_message_id,
            is_gpt_query=is_gpt_query,
            gpt_response=gpt_response,
            message_type='gpt_query' if is_gpt_query else 'text'
        )
        
        db.session.add(message)
        db.session.commit()
        
        # Emit to all users in the group room - REAL-TIME ONLY
        room = f'group_{group_id}'
        message_data = message.to_dict()
        
        emit('new-group-message', {
            'message': message_data,
            'thread_id': thread_id,
            'group_id': group_id
        }, room=room)
        
        # Confirm successful send to sender
        emit('message-sent-confirm', {
            'message_id': message.id,
            'timestamp': message.created_at.isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error sending group message: {e}")
        emit('error', {'message': 'Failed to send message'})

@socketio.on('group-file-uploaded')
def handle_group_file_uploaded(data):
    """Notify group members about new file upload"""
    try:
        group_id = data.get('group_id')
        file_id = data.get('file_id')
        thread_id = data.get('thread_id')
        
        if not group_id or not file_id:
            return
        
        from models import GroupFile
        group_file = GroupFile.query.get(file_id)
        if not group_file:
            return
        
        room = f'group_{group_id}'
        emit('new-group-file', {
            'file': group_file.to_dict(),
            'thread_id': thread_id,
            'group_id': group_id
        }, room=room, include_self=True)
        
    except Exception as e:
        logger.error(f"Error notifying file upload: {e}")

@socketio.on('typing-in-group')
def handle_group_typing(data):
    """Handle typing indicators in group"""
    try:
        group_id = data.get('group_id')
        thread_id = data.get('thread_id')
        is_typing = data.get('is_typing', False)
        
        current_user = get_current_user()
        if not current_user or not group_id:
            return
        
        room = f'group_{group_id}'
        emit('user-typing', {
            'user_id': current_user.id,
            'username': current_user.username,
            'thread_id': thread_id,
            'is_typing': is_typing
        }, room=room, include_self=False)
        
    except Exception as e:
        logger.error(f"Error handling typing indicator: {e}")

# ANALYTICS ROUTE
@app.route('/api/analytics/ultra-academic', methods=['GET'])
def api_ultra_academic_analytics():
    """Ultra-comprehensive analytics for academic system optimization"""
    if not session.get('is_admin') and session.get('role') != 'teacher':
        return jsonify({'error': 'Unauthorized. Admin or teacher access required.'}), 403

    try:
        # Ultra-comprehensive conversation analytics
        total_conversations = UltraAcademicConversationHistory.query.count()
        recent_conversations = UltraAcademicConversationHistory.query.filter(
            UltraAcademicConversationHistory.created_at >= datetime.utcnow() - timedelta(days=7)
        ).count()

        # Academic type distribution
        academic_types = db.session.query(
            UltraAcademicConversationHistory.academic_type,
            db.func.count(UltraAcademicConversationHistory.id)
        ).group_by(UltraAcademicConversationHistory.academic_type).all()

        # Complexity level distribution
        complexity_levels = db.session.query(
            UltraAcademicConversationHistory.complexity_level,
            db.func.count(UltraAcademicConversationHistory.id)
        ).group_by(UltraAcademicConversationHistory.complexity_level).all()

        # Query type distribution
        query_types = db.session.query(
            UltraAcademicConversationHistory.query_type,
            db.func.count(UltraAcademicConversationHistory.id)
        ).group_by(UltraAcademicConversationHistory.query_type).all()

        # Document category distribution
        doc_categories = db.session.query(
            UltraAcademicDocumentStats.document_category,
            db.func.count(UltraAcademicDocumentStats.id)
        ).group_by(UltraAcademicDocumentStats.document_category).all()

        # Subject distribution
        subjects = db.session.query(
            UltraAcademicDocumentStats.subject_detected,
            db.func.count(UltraAcademicDocumentStats.id)
        ).group_by(UltraAcademicDocumentStats.subject_detected).all()

        # Performance metrics
        avg_confidence = db.session.query(
            db.func.avg(UltraAcademicConversationHistory.confidence_score)
        ).scalar() or 0

        avg_processing_time = db.session.query(
            db.func.avg(UltraAcademicConversationHistory.processing_time)
        ).scalar() or 0

        avg_academic_relevance = db.session.query(
            db.func.avg(UltraAcademicConversationHistory.academic_relevance)
        ).scalar() or 0

        # User statistics
        total_users = User.query.count()
        students = User.query.filter_by(role='student', is_active=True).count()
        teachers = User.query.filter_by(role='teacher', is_active=True).count()

        # Ultra-academic document statistics
        total_documents = UltraAcademicDocumentStats.query.count()
        total_chunks = db.session.query(
            db.func.sum(UltraAcademicDocumentStats.chunks_created)
        ).scalar() or 0

        avg_doc_processing_time = db.session.query(
            db.func.avg(UltraAcademicDocumentStats.processing_time)
        ).scalar() or 0

        avg_academic_relevance_docs = db.session.query(
            db.func.avg(UltraAcademicDocumentStats.academic_relevance_score)
        ).scalar() or 0

        # Feedback statistics
        feedback_stats = db.session.query(
            UltraAcademicConversationHistory.user_feedback,
            db.func.count(UltraAcademicConversationHistory.id)
        ).filter(UltraAcademicConversationHistory.user_feedback.isnot(None)).group_by(UltraAcademicConversationHistory.user_feedback).all()

        # Search method effectiveness
        search_effectiveness = db.session.query(
            UltraQueryPerformanceLog.search_strategies_used,
            db.func.avg(UltraQueryPerformanceLog.confidence_score),
            db.func.count(UltraQueryPerformanceLog.id)
        ).group_by(UltraQueryPerformanceLog.search_strategies_used).all()

        # Group study statistics
        total_groups = StudyGroup.query.filter_by(is_active=True).count()
        total_group_messages = GroupMessage.query.filter_by(is_deleted=False).count()
        total_group_members = GroupMember.query.filter_by(is_active=True).count()
        total_shared_files = GroupFile.query.filter_by(is_deleted=False).count()

        return jsonify({
            'success': True,
            'ultra_academic_analytics': {
                'conversations': {
                    'total': total_conversations,
                    'recent_week': recent_conversations,
                    'average_confidence': round(avg_confidence, 3),
                    'average_processing_time': round(avg_processing_time, 3),
                    'average_academic_relevance': round(avg_academic_relevance, 3)
                },
                'academic_types': dict(academic_types),
                'complexity_levels': dict(complexity_levels),
                'query_types': dict(query_types),
                'users': {
                    'total': total_users,
                    'students': students,
                    'teachers': teachers,
                    'admin_count': 1
                },
                'documents': {
                    'total': total_documents,
                    'total_chunks': total_chunks,
                    'average_processing_time': round(avg_doc_processing_time, 3),
                    'average_academic_relevance': round(avg_academic_relevance_docs, 3),
                    'categories': dict(doc_categories),
                    'subjects': dict(subjects)
                },
                'feedback': dict(feedback_stats),
                'search_effectiveness': [
                    {
                        'methods': methods,
                        'avg_confidence': round(confidence, 3),
                        'usage_count': count
                    }
                    for methods, confidence, count in search_effectiveness
                ],
                'group_study': {
                    'total_groups': total_groups,
                    'total_messages': total_group_messages,
                    'total_members': total_group_members,
                    'total_shared_files': total_shared_files,
                    'avg_members_per_group': round(total_group_members / max(total_groups, 1), 2)
                },
                'system_performance': {
                    'vectorstore_stats': doc_processor.get_system_stats() if doc_processor else {},
                    'system_uptime': datetime.now().isoformat()
                }
            },
            'system': 'ultra-robust-academic'
        })

    except Exception as e:
        logger.error(f"Error generating ultra-academic analytics: {e}")
        return jsonify({'error': 'Failed to generate analytics'}), 500

# HEALTH CHECK ROUTE
@app.route('/api/health', methods=['GET'])
def health_check():
    """Ultra-enhanced health check for academic system"""
    try:
        # Check document processor
        doc_processor_status = 'healthy' if doc_processor else 'unavailable'

        # Check database
        try:
            db.session.execute(db.text('SELECT 1'))
            db_status = 'healthy'
        except Exception:
            db_status = 'unhealthy'

        # Get comprehensive system stats
        stats = doc_processor.get_system_stats() if doc_processor else {}

        # Get database statistics
        try:
            total_conversations = UltraAcademicConversationHistory.query.count()
            total_documents = UltraAcademicDocumentStats.query.count()
            total_users = User.query.count()
            active_sessions = len(set(row[0] for row in db.session.query(UltraAcademicConversationHistory.session_id).distinct().all()))
            
            # Group study statistics
            total_groups = StudyGroup.query.filter_by(is_active=True).count()
            total_group_messages = GroupMessage.query.filter_by(is_deleted=False).count()
            total_shared_files = GroupFile.query.filter_by(is_deleted=False).count()
            
        except Exception:
            total_conversations = 0
            total_documents = 0
            total_users = 0
            active_sessions = 0
            total_groups = 0
            total_group_messages = 0
            total_shared_files = 0

        health_info = {
            'status': 'healthy' if doc_processor_status == 'healthy' and db_status == 'healthy' else 'degraded',
            'timestamp': datetime.now().isoformat(),
            'service': 'ultra-robust-academic-chatbot',
            'version': '9.0.0',
            'components': {
                'ultra_robust_document_processor': doc_processor_status,
                'database': db_status,
                'vectorstore': stats.get('vectorstore', {}).get('vectorstore_status', 'unknown'),
                'nltk_components': 'initialized' if doc_processor else 'unavailable',
                'group_study_system': 'active',
                'file_sharing_system': 'active',
            },
            'system_stats': stats,
            'database_stats': {
                'total_conversations': total_conversations,
                'total_documents': total_documents,
                'total_users': total_users,
                'active_sessions': active_sessions,
                'total_study_groups': total_groups,
                'total_group_messages': total_group_messages,
                'total_shared_files': total_shared_files
            },
            'processor_error': processor_error,
            'ultra_academic_capabilities': {
                'supported_formats': ['.pdf', '.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt', '.txt', '.html', '.htm'],
                'document_categories': ['textbook', 'question_paper', 'answer_script', 'notes', 'research_paper', 'general'],
                'embedding_model': 'text-embedding-3-large',
                'embedding_dimensions': 3072,
                'llm_model': 'gpt-4o-mini',
                'group_study_features': [
                    'real_time_group_chat',
                    'threaded_discussions',
                    'gpt_integration_via_@sana',
                    'file_sharing_system',
                    'group_admin_controls',
                    'member_management',
                    'academic_context_awareness'
                ],
                'file_sharing_features': [
                    'multi_format_support',
                    'image_preview',
                    'download_tracking',
                    'file_categorization',
                    'permission_based_deletion',
                    '50mb_file_limit'
                ]
            }
        }
        return jsonify(health_info)

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'error',
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'system': 'ultra-robust-academic'
        }), 500

# STATIC FILE ROUTES
@app.route('/group-study')
def group_study_page():
    """Serve the group study page"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="refresh" content="0;url=/group_study.html">
    </head>
    <body>
        <p>Redirecting to Group Study...</p>
    </body>
    </html>
    '''

@app.route('/group_study.html')
def serve_group_study():
    """Serve group study HTML directly"""
    return app.send_static_file('group_study.html') if os.path.exists('static/group_study.html') else '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Group Study</title>
    </head>
    <body>
        <h1>Group Study Page</h1>
        <p>Group study functionality will be loaded here.</p>
        <script>
            window.location.href = '/group_study.html';
        </script>
    </body>
    </html>
    '''

@app.route('/')
def index():
    """Serve the main index page"""
    return app.send_static_file('index.html') if os.path.exists('static/index.html') else '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ultra-Robust Academic Chatbot</title>
    </head>
    <body>
        <h1>Ultra-Robust Academic Chatbot</h1>
        <p>Please ensure your index.html file is in the static directory.</p>
    </body>
    </html>
    '''

# ADDITIONAL UTILITY ROUTES
@app.route('/api/system/status', methods=['GET'])
def system_status():
    """Get comprehensive system status"""
    try:
        return jsonify({
            'success': True,
            'system': 'ultra-robust-academic-chatbot',
            'version': '9.0.0',
            'timestamp': datetime.now().isoformat(),
            'features': {
                'authentication': True,
                'file_management': True,
                'chat_system': True,
                'group_study': True,
                'file_sharing': True,
                'video_calling': True,
                'analytics': True,
                'document_processing': doc_processor is not None
            },
            'database_status': 'connected',
            'processor_status': 'available' if doc_processor else 'unavailable'
        })
    except Exception as e:
        logger.error(f"System status error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'system': 'ultra-robust-academic'
        }), 500

@app.route('/api/groups/<int:group_id>/statistics', methods=['GET'])
def get_group_statistics(group_id):
    """Get detailed statistics for a specific group"""
    if not require_auth():
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        current_user = get_current_user()
        if not current_user:
            return jsonify({'error': 'User not found'}), 404
        
        group = db.session.get(StudyGroup, group_id)
        if not group or not group.is_member(current_user.id):
            return jsonify({'error': 'Group not found or access denied'}), 403
        
        # Get group statistics
        total_messages = GroupMessage.query.filter_by(group_id=group_id, is_deleted=False).count()
        total_threads = GroupThread.query.filter_by(group_id=group_id, is_active=True).count()
        total_files = GroupFile.query.filter_by(group_id=group_id, is_deleted=False).count()
        total_downloads = db.session.query(db.func.sum(GroupFile.download_count)).filter_by(group_id=group_id, is_deleted=False).scalar() or 0
        
        # Get file type distribution
        file_types = db.session.query(
            GroupFile.file_type,
            db.func.count(GroupFile.id)
        ).filter_by(group_id=group_id, is_deleted=False).group_by(GroupFile.file_type).all()
        
        # Get most active members
        most_active_members = db.session.query(
            GroupMessage.user_id,
            User.full_name,
            db.func.count(GroupMessage.id).label('message_count')
        ).join(User).filter(
            GroupMessage.group_id == group_id,
            GroupMessage.is_deleted == False
        ).group_by(GroupMessage.user_id, User.full_name).order_by(
            db.func.count(GroupMessage.id).desc()
        ).limit(5).all()
        
        return jsonify({
            'success': True,
            'group_statistics': {
                'group_info': group.to_dict(),
                'totals': {
                    'messages': total_messages,
                    'threads': total_threads,
                    'shared_files': total_files,
                    'file_downloads': total_downloads
                },
                'file_types': dict(file_types),
                'most_active_members': [
                    {
                        'user_id': member[0],
                        'name': member[1],
                        'message_count': member[2]
                    }
                    for member in most_active_members
                ]
            },
            'system': 'ultra-robust-academic'
        })
        
    except Exception as e:
        logger.error(f"Error getting group statistics: {e}")
        return jsonify({'error': 'Failed to get group statistics'}), 500

# ERROR HANDLING AND CLEANUP
@app.teardown_appcontext
def close_db(error):
    """Close database connections properly without data loss"""
    try:
        db.session.close()
    except Exception as cleanup_error:
        # Log cleanup errors but don't interfere with data
        logger.error(f"Session cleanup error: {cleanup_error}")
        pass


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() in ('true', '1', 'yes')  # or just debug = True

    try:
            socketio.run(app, 
                debug=debug, 
                host='0.0.0.0', 
                port=port,
                log_output=True,
                use_reloader=False)

    except Exception as e:
        logger.error(f"❌ Failed to start application: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
