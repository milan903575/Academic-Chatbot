from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from datetime import datetime, timedelta, timezone
import json
import secrets
import uuid
import pytz

db = SQLAlchemy()
bcrypt = Bcrypt()

# IST timezone
IST = pytz.timezone('Asia/Kolkata')

def get_ist_now():
    """Get current time in IST"""
    return datetime.now(IST)

def to_ist(dt):
    """Convert UTC datetime to IST"""
    if dt is None:
        return None
    # If dt is naive, assume UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    # Convert to IST
    return dt.astimezone(IST)

def ist_isoformat(dt):
    """Convert datetime to IST ISO format string"""
    if dt is None:
        return None
    ist_dt = to_ist(dt)
    return ist_dt.isoformat()

# Enhanced User model for academic environment
class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(128), nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(20), default='student')  # student, teacher, admin
    institution = db.Column(db.String(100))
    course = db.Column(db.String(100))
    academic_year = db.Column(db.String(20))
    subjects_of_interest = db.Column(db.Text)  # JSON string
    created_at = db.Column(db.DateTime, default=get_ist_now)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    login_attempts = db.Column(db.Integer, default=0)
    locked_until = db.Column(db.DateTime)
    preferences = db.Column(db.Text)  # JSON string for user preferences
    
    # Relationships
    created_groups = db.relationship('StudyGroup', backref='admin_user', lazy='dynamic', foreign_keys='StudyGroup.admin_id')
    group_memberships = db.relationship('GroupMember', backref='user', lazy='dynamic')
    group_messages = db.relationship('GroupMessage', backref='author', lazy='dynamic')
    
    def set_password(self, password):
        """Hash and set password with enhanced security"""
        self.password_hash = bcrypt.generate_password_hash(password, rounds=12).decode('utf-8')
    
    def check_password(self, password):
        """Check password with rate limiting consideration"""
        if self.locked_until and get_ist_now() < to_ist(self.locked_until):
            return False
        return bcrypt.check_password_hash(self.password_hash, password)
    
    def lock_account(self):
        """Lock account for security"""
        self.locked_until = get_ist_now() + timedelta(minutes=15)
        self.login_attempts = 0
    
    def reset_login_attempts(self):
        """Reset failed login attempts"""
        self.login_attempts = 0
        self.locked_until = None
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'role': self.role,
            'institution': self.institution,
            'course': self.course,
            'academic_year': self.academic_year,
            'subjects_of_interest': json.loads(self.subjects_of_interest) if self.subjects_of_interest else [],
            'created_at': ist_isoformat(self.created_at),
            'last_login': ist_isoformat(self.last_login),
            'is_active': self.is_active
        }

# Study Group model
class StudyGroup(db.Model):
    __tablename__ = 'study_groups'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    group_code = db.Column(db.String(10), unique=True, nullable=False, index=True)
    admin_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    subject = db.Column(db.String(100))
    max_members = db.Column(db.Integer, default=50)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=get_ist_now)
    
    # Relationships
    members = db.relationship('GroupMember', backref='group', lazy='dynamic', cascade='all, delete-orphan')
    messages = db.relationship('GroupMessage', backref='group', lazy='dynamic', cascade='all, delete-orphan')
    threads = db.relationship('GroupThread', backref='group', lazy='dynamic', cascade='all, delete-orphan')
    shared_files = db.relationship('GroupFile', backref='group', lazy='dynamic', cascade='all, delete-orphan')
    
    def __init__(self, **kwargs):
        super(StudyGroup, self).__init__(**kwargs)
        if not self.group_code:
            self.group_code = self.generate_group_code()
    
    @staticmethod
    def generate_group_code():
        """Generate unique 8-character group code"""
        while True:
            code = secrets.token_urlsafe(6)[:8].upper()
            if not StudyGroup.query.filter_by(group_code=code).first():
                return code
    
    def get_member_count(self):
        return self.members.filter_by(is_active=True).count()
    
    def is_admin(self, user_id):
        return self.admin_id == user_id
    
    def is_member(self, user_id):
        return self.members.filter_by(user_id=user_id, is_active=True).first() is not None
    
    def can_join(self):
        return self.is_active and self.get_member_count() < self.max_members
    
    def to_dict(self, include_members=False):
        result = {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'group_code': self.group_code,
            'admin_id': self.admin_id,
            'admin_name': self.admin_user.full_name,
            'subject': self.subject,
            'max_members': self.max_members,
            'member_count': self.get_member_count(),
            'is_active': self.is_active,
            'created_at': ist_isoformat(self.created_at)
        }
        
        if include_members:
            result['members'] = [member.to_dict() for member in self.members.filter_by(is_active=True)]
        
        return result

# Group Member model
class GroupMember(db.Model):
    __tablename__ = 'group_members'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    group_id = db.Column(db.Integer, db.ForeignKey('study_groups.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    joined_at = db.Column(db.DateTime, default=get_ist_now)
    is_active = db.Column(db.Boolean, default=True)
    role = db.Column(db.String(20), default='member')  # member, moderator
    
    __table_args__ = (db.UniqueConstraint('group_id', 'user_id', name='unique_group_member'),)
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'username': self.user.username,
            'full_name': self.user.full_name,
            'role': self.role,
            'joined_at': ist_isoformat(self.joined_at),
            'is_active': self.is_active
        }

# Group Thread model for organized discussions
class GroupThread(db.Model):
    __tablename__ = 'group_threads'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    group_id = db.Column(db.Integer, db.ForeignKey('study_groups.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=get_ist_now)
    is_active = db.Column(db.Boolean, default=True)
    is_pinned = db.Column(db.Boolean, default=False)
    
    # Relationships
    messages = db.relationship('GroupMessage', backref='thread', lazy='dynamic')
    creator = db.relationship('User', backref='created_threads')
    
    def get_message_count(self):
        return self.messages.count()
    
    def get_last_message(self):
        return self.messages.order_by(GroupMessage.created_at.desc()).first()
    
    def to_dict(self):
        last_message = self.get_last_message()
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'created_by': self.created_by,
            'creator_name': self.creator.full_name,
            'created_at': ist_isoformat(self.created_at),
            'message_count': self.get_message_count(),
            'last_message': last_message.to_dict() if last_message else None,
            'is_pinned': self.is_pinned,
            'is_active': self.is_active
        }

# Group Message model
class GroupMessage(db.Model):
    __tablename__ = 'group_messages'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    group_id = db.Column(db.Integer, db.ForeignKey('study_groups.id'), nullable=False)
    thread_id = db.Column(db.Integer, db.ForeignKey('group_threads.id'), nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    message = db.Column(db.Text, nullable=False)
    message_type = db.Column(db.String(20), default='text')  # text, gpt_query, gpt_response, system, file
    parent_message_id = db.Column(db.Integer, db.ForeignKey('group_messages.id'), nullable=True)
    is_gpt_query = db.Column(db.Boolean, default=False)
    gpt_response = db.Column(db.Text)
    file_id = db.Column(db.Integer, db.ForeignKey('group_files.id'), nullable=True)  # NEW: For file messages
    created_at = db.Column(db.DateTime, default=get_ist_now)
    edited_at = db.Column(db.DateTime)
    is_deleted = db.Column(db.Boolean, default=False)
    
    # Self-referential relationship for replies
    replies = db.relationship('GroupMessage', backref=db.backref('parent', remote_side=[id]), lazy='dynamic')
    
    # Relationship to shared file
    shared_file = db.relationship('GroupFile', backref='message', lazy='select')
    
    def to_dict(self, include_replies=False):
        # ✅ CRITICAL FIX: Ensure timestamps are timezone-aware
        created_at_ist = self.created_at
        if created_at_ist.tzinfo is None:
            # Convert naive datetime to IST-aware
            created_at_ist = IST.localize(created_at_ist)
        
        edited_at_ist = None
        if self.edited_at:
            if self.edited_at.tzinfo is None:
                edited_at_ist = IST.localize(self.edited_at)
            else:
                edited_at_ist = self.edited_at
        
        result = {
            'id': self.id,
            'group_id': self.group_id,
            'thread_id': self.thread_id,
            'user_id': self.user_id,
            'username': self.author.username,
            'full_name': self.author.full_name,
            'message': self.message,
            'message_type': self.message_type,
            'parent_message_id': self.parent_message_id,
            'is_gpt_query': self.is_gpt_query,
            'gpt_response': self.gpt_response,
            'file_id': self.file_id,
            'shared_file': self.shared_file.to_dict() if self.shared_file else None,
            'created_at': created_at_ist.isoformat(),  # ✅ Now includes +05:30
            'edited_at': edited_at_ist.isoformat() if edited_at_ist else None,
            'is_deleted': self.is_deleted
        }
        
        if include_replies:
            result['replies'] = [reply.to_dict() for reply in self.replies.filter_by(is_deleted=False)]
        return result


# NEW: Group File model for file sharing
class GroupFile(db.Model):
    __tablename__ = 'group_files'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    group_id = db.Column(db.Integer, db.ForeignKey('study_groups.id'), nullable=False)
    thread_id = db.Column(db.Integer, db.ForeignKey('group_threads.id'), nullable=True)
    uploaded_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    stored_filename = db.Column(db.String(255), nullable=False)
    file_size = db.Column(db.Integer, nullable=False)
    file_type = db.Column(db.String(50))  # image, document, video, audio, other
    mime_type = db.Column(db.String(100))
    file_path = db.Column(db.String(500), nullable=False)
    description = db.Column(db.Text)
    download_count = db.Column(db.Integer, default=0)
    is_deleted = db.Column(db.Boolean, default=False)
    uploaded_at = db.Column(db.DateTime, default=get_ist_now)
    
    # Relationships
    uploader = db.relationship('User', backref='uploaded_files')
    
    def to_dict(self):
        # ✅ Fix file upload timestamps as well
        uploaded_at_ist = self.uploaded_at
        if uploaded_at_ist.tzinfo is None:
            uploaded_at_ist = IST.localize(uploaded_at_ist)
        
        return {
            'id': self.id,
            'group_id': self.group_id,
            'thread_id': self.thread_id,
            'uploaded_by': self.uploaded_by,
            'uploader_name': self.uploader.full_name if self.uploader else 'Unknown User',
            'original_filename': self.original_filename,
            'stored_filename': self.stored_filename,
            'file_size': self.file_size,
            'file_size_formatted': self.format_file_size(),
            'file_type': self.file_type,
            'mime_type': self.mime_type,
            'description': self.description,
            'download_count': self.download_count,
            'uploaded_at': uploaded_at_ist.isoformat(),  # ✅ Now includes +05:30
            'is_image': self.is_image(),
            'is_video': self.is_video(),
            'is_audio': self.is_audio()
        }

    def format_file_size(self):
        """Format file size in human readable format"""
        if self.file_size == 0:
            return "0B"
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        size = float(self.file_size)
        while size >= 1024 and i < len(size_names) - 1:
            size /= 1024.0
            i += 1
        return f"{size:.1f}{size_names[i]}"
    
    def is_image(self):
        """Check if file is an image"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg'}
        return any(self.original_filename.lower().endswith(ext) for ext in image_extensions)
    
    def is_video(self):
        """Check if file is a video"""
        video_extensions = {'.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv'}
        return any(self.original_filename.lower().endswith(ext) for ext in video_extensions)
    
    def is_audio(self):
        """Check if file is audio"""
        audio_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'}
        return any(self.original_filename.lower().endswith(ext) for ext in audio_extensions)

# Enhanced conversation history for academic tracking
class UltraAcademicConversationHistory(db.Model):
    __tablename__ = 'ultra_academic_conversation_history'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    session_id = db.Column(db.String(128), nullable=False, index=True)
    query_text = db.Column(db.Text, nullable=False)
    query_hash = db.Column(db.String(64), index=True)  # For deduplication
    response = db.Column(db.Text, nullable=False)
    query_type = db.Column(db.String(50))
    academic_type = db.Column(db.String(50))  # textbook, question_paper, answer_script, general
    complexity_level = db.Column(db.String(20))
    expected_length = db.Column(db.String(20))
    confidence_score = db.Column(db.Float)
    processing_time = db.Column(db.Float)
    sources_used = db.Column(db.Text)  # JSON string of source files
    search_methods = db.Column(db.Text)  # JSON string of search methods used
    academic_relevance = db.Column(db.Float)
    follow_up_context = db.Column(db.Boolean, default=False)
    key_concepts = db.Column(db.Text)  # JSON string
    max_tokens_used = db.Column(db.Integer)
    documents_searched = db.Column(db.Integer)
    user_feedback = db.Column(db.String(20))  # helpful, not_helpful, partially_helpful
    created_at = db.Column(db.DateTime, default=get_ist_now)

# Enhanced document processing stats for ultra-robust academic content
class UltraAcademicDocumentStats(db.Model):
    __tablename__ = 'ultra_academic_document_stats'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(10), nullable=False)
    document_category = db.Column(db.String(50))  # textbook, question_paper, answer_script, notes, general
    subject_detected = db.Column(db.String(100))
    academic_relevance_score = db.Column(db.Float)
    file_size = db.Column(db.Integer, nullable=False)
    processing_time = db.Column(db.Float, nullable=False)
    chunks_created = db.Column(db.Integer, nullable=False)
    content_length = db.Column(db.Integer, nullable=False)
    extraction_methods = db.Column(db.Text)  # JSON string
    chunk_types_used = db.Column(db.Text)  # JSON string
    academic_patterns_detected = db.Column(db.Integer, default=0)
    question_patterns_found = db.Column(db.Integer, default=0)
    answer_patterns_found = db.Column(db.Integer, default=0)
    academic_terms_count = db.Column(db.Integer, default=0)
    parallel_processing_used = db.Column(db.Boolean, default=False)
    thread_count = db.Column(db.Integer, default=1)
    embedding_model = db.Column(db.String(50), default='text-embedding-3-large')
    embedding_dimensions = db.Column(db.Integer, default=3072)
    processed_at = db.Column(db.DateTime, default=get_ist_now)
    processed_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)

# Query performance tracking for ultra-robust optimization
class UltraQueryPerformanceLog(db.Model):
    __tablename__ = 'ultra_query_performance_log'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    query_text = db.Column(db.Text, nullable=False)
    query_hash = db.Column(db.String(64), index=True)
    academic_type = db.Column(db.String(50))
    complexity_level = db.Column(db.String(20))
    search_strategies_used = db.Column(db.Text)  # JSON
    results_found = db.Column(db.Integer)
    unique_sources = db.Column(db.Integer)
    processing_time = db.Column(db.Float)
    confidence_score = db.Column(db.Float)
    academic_relevance_avg = db.Column(db.Float)
    user_satisfaction = db.Column(db.String(20))  # satisfied, neutral, dissatisfied
    follow_up_questions = db.Column(db.Integer, default=0)
    session_id = db.Column(db.String(128))
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=get_ist_now)

# Document upload session tracking
class DocumentUploadSession(db.Model):
    __tablename__ = 'document_upload_sessions'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    session_id = db.Column(db.String(128), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    total_files = db.Column(db.Integer, nullable=False)
    processed_files = db.Column(db.Integer, default=0)
    failed_files = db.Column(db.Integer, default=0)
    total_chunks = db.Column(db.Integer, default=0)
    total_processing_time = db.Column(db.Float, default=0.0)
    academic_documents_detected = db.Column(db.Integer, default=0)
    status = db.Column(db.String(20), default='processing')  # processing, completed, failed
    error_messages = db.Column(db.Text)  # JSON string
    created_at = db.Column(db.DateTime, default=get_ist_now)
    completed_at = db.Column(db.DateTime)
