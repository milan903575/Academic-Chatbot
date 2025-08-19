from flask import Blueprint, request, jsonify, session, current_app, send_file
from models import db, StudyGroup, GroupMember, GroupMessage, GroupThread, GroupFile, User, get_ist_now
from datetime import datetime
import logging
import json
import re
import uuid
import os
from werkzeug.utils import secure_filename
import mimetypes

logger = logging.getLogger(__name__)

# Create blueprint for group study routes
group_study_bp = Blueprint('group_study', __name__)

def require_auth():
    """Check if user is authenticated"""
    if 'user_id' not in session and 'is_admin' not in session:
        return False
    return True

def get_current_user():
    """Get current user object"""
    if session.get('is_admin'):
        return None # Admin user, handle separately
    elif session.get('user_id'):
        return User.query.get(session['user_id'])
    return None

def get_file_type(filename):
    """Determine file type based on extension"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg'}
    video_extensions = {'.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv'}
    audio_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'}
    document_extensions = {'.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt'}
    
    ext = os.path.splitext(filename.lower())[1]
    
    if ext in image_extensions:
        return 'image'
    elif ext in video_extensions:
        return 'video'
    elif ext in audio_extensions:
        return 'audio'
    elif ext in document_extensions:
        return 'document'
    else:
        return 'other'
    
def process_group_gpt_query(message_text, group, thread, current_user):
    """Process GPT query for group messages - shared with SocketIO handlers"""
    is_gpt_query = '@sana' in message_text.lower()
    gpt_response = None

    if is_gpt_query:
        query = re.sub(r'@sana\s*', '', message_text, flags=re.IGNORECASE).strip()
        if query:
            try:
                # Import here to avoid circular imports
                from app import doc_processor
                if doc_processor:
                    response_data = doc_processor.generate_group_response(
                        query=query,
                        group_context={
                            'group_id': group.id,
                            'group_name': group.name,
                            'thread_id': thread.id,
                            'thread_title': thread.title,
                            'member_count': group.get_member_count(),
                            'context_mode': 'group_study_discussion',
                            'is_group_message': True
                        },
                        session_id=f"user_{current_user.id}"
                    )
                    gpt_response = response_data.get('response', 'Sorry, I could not generate a response for the group discussion.')
                else:
                    gpt_response = 'GPT service is currently unavailable for group discussions.'
            except Exception as gpt_error:
                logger.error(f"Group GPT processing error: {gpt_error}")
                gpt_response = 'Error processing GPT request in group discussion.'

    return is_gpt_query, gpt_response



@group_study_bp.route('/api/groups/create', methods=['POST'])
def create_group():
    """Create a new study group"""
    if not require_auth():
        return jsonify({'error': 'Unauthorized'}), 403

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        name = data.get('name', '').strip()
        description = data.get('description', '').strip()
        subject = data.get('subject', '').strip()
        max_members = data.get('max_members', 50)

        if not name:
            return jsonify({'error': 'Group name is required'}), 400

        if len(name) > 100:
            return jsonify({'error': 'Group name too long (max 100 characters)'}), 400

        if max_members < 2 or max_members > 100:
            return jsonify({'error': 'Max members must be between 2 and 100'}), 400

        current_user = get_current_user()
        if not current_user:
            return jsonify({'error': 'User not found'}), 404

        # Create new group
        group = StudyGroup(
            name=name,
            description=description,
            subject=subject,
            max_members=max_members,
            admin_id=current_user.id
        )

        db.session.add(group)
        db.session.flush() # Get the group ID

        # Add creator as first member
        member = GroupMember(
            group_id=group.id,
            user_id=current_user.id,
            role='admin'
        )
        db.session.add(member)

        # Create default general thread
        general_thread = GroupThread(
            group_id=group.id,
            title='General Discussion',
            description='General discussion for all group members',
            created_by=current_user.id,
            is_pinned=True
        )
        db.session.add(general_thread)

        db.session.commit()

        logger.info(f"Study group created: {name} by user {current_user.username}")

        return jsonify({
            'success': True,
            'message': 'Study group created successfully',
            'group': group.to_dict(include_members=True)
        }), 201

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error creating study group: {e}")
        return jsonify({'error': 'Failed to create group'}), 500

@group_study_bp.route('/api/groups/join', methods=['POST'])
def join_group():
    """Join a study group using group code"""
    if not require_auth():
        return jsonify({'error': 'Unauthorized'}), 403

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        group_code = data.get('group_code', '').strip().upper()
        if not group_code:
            return jsonify({'error': 'Group code is required'}), 400

        current_user = get_current_user()
        if not current_user:
            return jsonify({'error': 'User not found'}), 404

        # Find group by code
        group = StudyGroup.query.filter_by(group_code=group_code, is_active=True).first()
        if not group:
            return jsonify({'error': 'Invalid group code'}), 404

        # Check if user is already a member
        existing_member = GroupMember.query.filter_by(
            group_id=group.id,
            user_id=current_user.id,
            is_active=True
        ).first()

        if existing_member:
            return jsonify({'error': 'You are already a member of this group'}), 400

        # Check if group can accept more members
        if not group.can_join():
            return jsonify({'error': 'Group is full or inactive'}), 400

        # Add user to group
        member = GroupMember(
            group_id=group.id,
            user_id=current_user.id
        )
        db.session.add(member)
        db.session.commit()

        logger.info(f"User {current_user.username} joined group {group.name}")

        return jsonify({
            'success': True,
            'message': 'Successfully joined the group',
            'group': group.to_dict()
        }), 200

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error joining study group: {e}")
        return jsonify({'error': 'Failed to join group'}), 500

@group_study_bp.route('/api/groups/my-groups', methods=['GET'])
def get_my_groups():
    """Get all groups user is a member of"""
    if not require_auth():
        return jsonify({'error': 'Unauthorized'}), 403

    try:
        current_user = get_current_user()
        if not current_user:
            return jsonify({'error': 'User not found'}), 404

        # Get all active memberships
        memberships = GroupMember.query.filter_by(
            user_id=current_user.id,
            is_active=True
        ).join(StudyGroup).filter(StudyGroup.is_active == True).all()

        groups = []
        for membership in memberships:
            group_data = membership.group.to_dict()
            group_data['my_role'] = membership.role
            group_data['joined_at'] = membership.joined_at.isoformat()
            groups.append(group_data)

        return jsonify({
            'success': True,
            'groups': groups
        }), 200

    except Exception as e:
        logger.error(f"Error getting user groups: {e}")
        return jsonify({'error': 'Failed to get groups'}), 500

@group_study_bp.route('/api/groups/<int:group_id>', methods=['GET'])
def get_group_details(group_id):
    """Get detailed information about a specific group"""
    if not require_auth():
        return jsonify({'error': 'Unauthorized'}), 403

    try:
        current_user = get_current_user()
        if not current_user:
            return jsonify({'error': 'User not found'}), 404

        group = StudyGroup.query.get(group_id)
        if not group or not group.is_active:
            return jsonify({'error': 'Group not found'}), 404

        # Check if user is a member
        if not group.is_member(current_user.id):
            return jsonify({'error': 'You are not a member of this group'}), 403

        # Get threads
        threads = GroupThread.query.filter_by(
            group_id=group_id,
            is_active=True
        ).order_by(GroupThread.is_pinned.desc(), GroupThread.created_at.desc()).all()

        group_data = group.to_dict(include_members=True)
        group_data['threads'] = [thread.to_dict() for thread in threads]
        group_data['is_admin'] = group.is_admin(current_user.id)

        return jsonify({
            'success': True,
            'group': group_data
        }), 200

    except Exception as e:
        logger.error(f"Error getting group details: {e}")
        return jsonify({'error': 'Failed to get group details'}), 500

@group_study_bp.route('/api/groups/<int:group_id>/delete', methods=['DELETE'])
def delete_group(group_id):
    """Delete a study group (admin only)"""
    if not require_auth():
        return jsonify({'error': 'Unauthorized'}), 403

    try:
        current_user = get_current_user()
        if not current_user:
            return jsonify({'error': 'User not found'}), 404

        group = StudyGroup.query.get(group_id)
        if not group:
            return jsonify({'error': 'Group not found'}), 404

        # Check if user is admin of the group
        if not group.is_admin(current_user.id):
            return jsonify({'error': 'Only group admin can delete the group'}), 403

        # Soft delete the group
        group.is_active = False
        db.session.commit()

        logger.info(f"Study group deleted: {group.name} by user {current_user.username}")

        return jsonify({
            'success': True,
            'message': 'Group deleted successfully'
        }), 200

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting study group: {e}")
        return jsonify({'error': 'Failed to delete group'}), 500

@group_study_bp.route('/api/groups/<int:group_id>/members/<int:user_id>/remove', methods=['DELETE'])
def remove_member_restful(group_id, user_id):
    """Remove a member from the group (admin only) - RESTful endpoint"""
    if not require_auth():
        return jsonify({'error': 'Unauthorized'}), 403

    try:
        current_user = get_current_user()
        if not current_user:
            return jsonify({'error': 'User not found'}), 404

        group = StudyGroup.query.get(group_id)
        if not group:
            return jsonify({'error': 'Group not found'}), 404

        # Check if current user is admin
        if not group.is_admin(current_user.id):
            return jsonify({'error': 'Only group admin can remove members'}), 403

        # Can't remove yourself
        if user_id == current_user.id:
            return jsonify({'error': 'Cannot remove yourself from the group'}), 400

        # Find and remove member
        member = GroupMember.query.filter_by(
            group_id=group_id,
            user_id=user_id,
            is_active=True
        ).first()

        if not member:
            return jsonify({'error': 'User is not a member of this group'}), 404

        member.is_active = False
        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'Member removed successfully'
        }), 200

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error removing group member: {e}")
        return jsonify({'error': 'Failed to remove member'}), 500


    except Exception as e:
        db.session.rollback()
        logger.error(f"Error removing group member: {e}")
        return jsonify({'error': 'Failed to remove member'}), 500

@group_study_bp.route('/api/groups/<int:group_id>/threads', methods=['GET'])
def get_group_threads(group_id):
    """Get all threads for a group"""
    if not require_auth():
        return jsonify({'error': 'Unauthorized'}), 403

    try:
        current_user = get_current_user()
        if not current_user:
            return jsonify({'error': 'User not found'}), 404

        group = StudyGroup.query.get(group_id)
        if not group or not group.is_member(current_user.id):
            return jsonify({'error': 'Group not found or access denied'}), 403

        threads = GroupThread.query.filter_by(
            group_id=group_id,
            is_active=True
        ).order_by(GroupThread.is_pinned.desc(), GroupThread.created_at.desc()).all()

        return jsonify({
            'success': True,
            'threads': [thread.to_dict() for thread in threads]
        }), 200

    except Exception as e:
        logger.error(f"Error getting group threads: {e}")
        return jsonify({'error': 'Failed to get threads'}), 500

@group_study_bp.route('/api/groups/<int:group_id>/threads/create', methods=['POST'])
def create_thread(group_id):
    """Create a new thread in a group"""
    if not require_auth():
        return jsonify({'error': 'Unauthorized'}), 403

    try:
        data = request.get_json()
        title = data.get('title', '').strip()
        description = data.get('description', '').strip()

        if not title:
            return jsonify({'error': 'Thread title is required'}), 400

        current_user = get_current_user()
        if not current_user:
            return jsonify({'error': 'User not found'}), 404

        group = StudyGroup.query.get(group_id)
        if not group or not group.is_member(current_user.id):
            return jsonify({'error': 'Group not found or access denied'}), 403

        thread = GroupThread(
            group_id=group_id,
            title=title,
            description=description,
            created_by=current_user.id
        )

        db.session.add(thread)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'Thread created successfully',
            'thread': thread.to_dict()
        }), 201

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error creating thread: {e}")
        return jsonify({'error': 'Failed to create thread'}), 500

@group_study_bp.route('/api/groups/<int:group_id>/threads/<int:thread_id>/messages', methods=['GET'])
def get_messages(group_id, thread_id):
    """Get messages for a specific thread - NEEDED for initial load"""
    if not require_auth():
        return jsonify({'error': 'Unauthorized'}), 403

    try:
        current_user = get_current_user()
        if not current_user:
            return jsonify({'error': 'User not found'}), 404

        group = StudyGroup.query.get(group_id)
        if not group or not group.is_member(current_user.id):
            return jsonify({'error': 'Group not found or access denied'}), 403

        thread = GroupThread.query.get(thread_id)
        if not thread or thread.group_id != group_id:
            return jsonify({'error': 'Thread not found'}), 404

        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 50, type=int), 100)

        # FIXED: Get messages in DESCENDING order (newest first)
        messages_query = GroupMessage.query.filter_by(
            group_id=group_id,
            thread_id=thread_id,
            is_deleted=False
        ).order_by(GroupMessage.created_at.desc())  # DESC = newest first

        messages_pagination = messages_query.paginate(
            page=page, per_page=per_page, error_out=False
        )

        return jsonify({
            'success': True,
            'messages': [message.to_dict() for message in messages_pagination.items],
            'pagination': {
                'page': messages_pagination.page,
                'pages': messages_pagination.pages,
                'per_page': messages_pagination.per_page,
                'total': messages_pagination.total,
                'has_next': messages_pagination.has_next,
                'has_prev': messages_pagination.has_prev
            }
        }), 200

    except Exception as e:
        logger.error(f"Error getting messages: {e}")
        return jsonify({'error': 'Failed to load messages'}), 500



@group_study_bp.route('/api/groups/<int:group_id>/threads/<int:thread_id>/messages', methods=['POST'])
def send_message_disabled(group_id, thread_id):
    """DISABLED: HTTP message sending - Use SocketIO only"""
    return jsonify({
        'success': False,
        'error': 'HTTP messaging disabled. Use SocketIO real-time messaging instead.',
        'socket_event': 'send-group-message'
    }), 400


@group_study_bp.route('/api/groups/<int:group_id>/leave', methods=['POST'])
def leave_group(group_id):
    """Leave a study group"""
    if not require_auth():
        return jsonify({'error': 'Unauthorized'}), 403

    try:
        current_user = get_current_user()
        if not current_user:
            return jsonify({'error': 'User not found'}), 404

        group = StudyGroup.query.get(group_id)
        if not group:
            return jsonify({'error': 'Group not found'}), 404

        # Check if user is admin
        if group.is_admin(current_user.id):
            return jsonify({'error': 'Group admin cannot leave. Please delete the group or transfer admin rights.'}), 400

        # Find and deactivate membership
        member = GroupMember.query.filter_by(
            group_id=group_id,
            user_id=current_user.id,
            is_active=True
        ).first()

        if not member:
            return jsonify({'error': 'You are not a member of this group'}), 404

        member.is_active = False
        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'Successfully left the group'
        }), 200

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error leaving group: {e}")
        return jsonify({'error': 'Failed to leave group'}), 500

@group_study_bp.route('/api/groups/search', methods=['GET'])
def search_groups():
    """Search for public groups"""
    if not require_auth():
        return jsonify({'error': 'Unauthorized'}), 403

    try:
        query = request.args.get('q', '').strip()
        subject = request.args.get('subject', '').strip()
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 20, type=int), 50)

        # Base query for active groups
        groups_query = StudyGroup.query.filter(StudyGroup.is_active == True)

        # Apply filters
        if query:
            groups_query = groups_query.filter(
                db.or_(
                    StudyGroup.name.ilike(f'%{query}%'),
                    StudyGroup.description.ilike(f'%{query}%')
                )
            )

        if subject:
            groups_query = groups_query.filter(StudyGroup.subject.ilike(f'%{subject}%'))

        # Paginate results
        groups_pagination = groups_query.order_by(StudyGroup.created_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )

        return jsonify({
            'success': True,
            'groups': [group.to_dict() for group in groups_pagination.items],
            'pagination': {
                'page': groups_pagination.page,
                'pages': groups_pagination.pages,
                'per_page': groups_pagination.per_page,
                'total': groups_pagination.total,
                'has_next': groups_pagination.has_next,
                'has_prev': groups_pagination.has_prev
            }
        }), 200

    except Exception as e:
        logger.error(f"Error searching groups: {e}")
        return jsonify({'error': 'Failed to search groups'}), 500

@group_study_bp.route('/api/groups/<int:group_id>/members', methods=['GET'])
def get_group_members(group_id):
    """Get all members of a group"""
    if not require_auth():
        return jsonify({'error': 'Unauthorized'}), 403

    try:
        current_user = get_current_user()
        if not current_user:
            return jsonify({'error': 'User not found'}), 404

        group = StudyGroup.query.get(group_id)
        if not group or not group.is_member(current_user.id):
            return jsonify({'error': 'Group not found or access denied'}), 403

        members = GroupMember.query.filter_by(
            group_id=group_id,
            is_active=True
        ).join(User).all()

        members_data = []
        for member in members:
            member_data = {
                'user_id': member.user.id,
                'username': member.user.username,
                'email': member.user.email,
                'full_name': member.user.full_name,
                'role': member.role,
                'joined_at': member.joined_at.isoformat(),
                'is_online': hasattr(member.user, 'last_seen') and member.user.last_login
            }
            members_data.append(member_data)

        return jsonify({
            'success': True,
            'members': members_data,
            'total_members': len(members_data)
        }), 200

    except Exception as e:
        logger.error(f"Error getting group members: {e}")
        return jsonify({'error': 'Failed to get group members'}), 500

@group_study_bp.route('/api/groups/<int:group_id>/invite', methods=['POST'])
def generate_invite_link(group_id):
    """Generate an invite link for a group"""
    if not require_auth():
        return jsonify({'error': 'Unauthorized'}), 403

    try:
        current_user = get_current_user()
        if not current_user:
            return jsonify({'error': 'User not found'}), 404

        group = StudyGroup.query.get(group_id)
        if not group:
            return jsonify({'error': 'Group not found'}), 404

        # Check if user is admin or has permission to invite
        if not group.is_admin(current_user.id):
            return jsonify({'error': 'Only group admin can generate invite links'}), 403

        # Generate invite link
        invite_link = f"{request.host_url}join/{group.group_code}"

        return jsonify({
            'success': True,
            'invite_link': invite_link,
            'group_code': group.group_code,
            'expires_at': None
        }), 200

    except Exception as e:
        logger.error(f"Error generating invite link: {e}")
        return jsonify({'error': 'Failed to generate invite link'}), 500

@group_study_bp.route('/api/groups/<int:group_id>/update', methods=['PUT'])
def update_group(group_id):
    """Update group details (admin only)"""
    if not require_auth():
        return jsonify({'error': 'Unauthorized'}), 403

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        current_user = get_current_user()
        if not current_user:
            return jsonify({'error': 'User not found'}), 404

        group = StudyGroup.query.get(group_id)
        if not group:
            return jsonify({'error': 'Group not found'}), 404

        # Check if user is admin
        if not group.is_admin(current_user.id):
            return jsonify({'error': 'Only group admin can update group details'}), 403

        # Update fields if provided
        if 'name' in data:
            name = data['name'].strip()
            if not name:
                return jsonify({'error': 'Group name cannot be empty'}), 400
            if len(name) > 100:
                return jsonify({'error': 'Group name too long (max 100 characters)'}), 400
            group.name = name

        if 'description' in data:
            group.description = data['description'].strip()

        if 'subject' in data:
            group.subject = data['subject'].strip()

        if 'max_members' in data:
            max_members = data['max_members']
            if max_members < 2 or max_members > 100:
                return jsonify({'error': 'Max members must be between 2 and 100'}), 400
            group.max_members = max_members

        # group.updated_at = datetime.utcnow()  # Add this field to your model if needed
        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'Group updated successfully',
            'group': group.to_dict()
        }), 200

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error updating group: {e}")
        return jsonify({'error': 'Failed to update group'}), 500

# ============= FILE MANAGEMENT ENDPOINTS =============

@group_study_bp.route('/api/groups/<int:group_id>/files', methods=['GET'])
def get_group_files(group_id):
    """Get all files shared in a group"""
    if not require_auth():
        return jsonify({'error': 'Unauthorized'}), 403

    try:
        current_user = get_current_user()
        if not current_user:
            return jsonify({'error': 'User not found'}), 404

        group = StudyGroup.query.get(group_id)
        if not group or not group.is_member(current_user.id):
            return jsonify({'error': 'Group not found or access denied'}), 403

        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 20, type=int), 50)
        file_type = request.args.get('type', '').strip()
        thread_id = request.args.get('thread_id', type=int)

        # Base query for group files
        files_query = GroupFile.query.filter_by(
            group_id=group_id,
            is_deleted=False
        )

        # Apply filters
        if file_type:
            files_query = files_query.filter(GroupFile.file_type == file_type)
        
        if thread_id:
            files_query = files_query.filter(GroupFile.thread_id == thread_id)

        # Order by upload date (newest first)
        files_query = files_query.order_by(GroupFile.uploaded_at.desc())

        # Paginate results
        files_pagination = files_query.paginate(
            page=page, per_page=per_page, error_out=False
        )

        return jsonify({
            'success': True,
            'files': [file.to_dict() for file in files_pagination.items],
            'pagination': {
                'page': files_pagination.page,
                'pages': files_pagination.pages,
                'per_page': files_pagination.per_page,
                'total': files_pagination.total,
                'has_next': files_pagination.has_next,
                'has_prev': files_pagination.has_prev
            }
        }), 200

    except Exception as e:
        logger.error(f"Error getting group files: {e}")
        return jsonify({'error': 'Failed to get group files'}), 500

@group_study_bp.route('/api/groups/<int:group_id>/files/upload', methods=['POST'])
def upload_group_file(group_id):
    """Upload a file to a group thread with real-time notifications"""
    if not require_auth():
        return jsonify({'error': 'Unauthorized'}), 403

    try:
        current_user = get_current_user()
        if not current_user:
            return jsonify({'error': 'User not found'}), 404

        group = StudyGroup.query.get(group_id)
        if not group or not group.is_member(current_user.id):
            return jsonify({'error': 'Group not found or access denied'}), 403

        thread_id = request.form.get('thread_id', type=int)
        description = request.form.get('description', '').strip()

        if thread_id:
            thread = GroupThread.query.get(thread_id)
            if not thread or thread.group_id != group_id:
                return jsonify({'error': 'Thread not found'}), 404

        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Check file size (50MB limit - increased for better group sharing)
        MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        if file_size > MAX_FILE_SIZE:
            return jsonify({'error': 'File too large (max 50MB for group sharing)'}), 400

        # Secure the filename
        original_filename = secure_filename(file.filename)
        
        # Generate unique filename
        file_extension = os.path.splitext(original_filename)[1]
        stored_filename = f"{uuid.uuid4()}{file_extension}"
        
        # Create upload directory
        upload_dir = os.path.join(
            current_app.config.get('UPLOAD_FOLDER', 'uploads'), 
            'groups', 
            str(group_id)
        )
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, stored_filename)
        
        # Save the file
        file.save(file_path)

        # Get file type and mime type
        file_type = get_file_type(original_filename)
        mime_type, _ = mimetypes.guess_type(original_filename)

        # Create GroupFile record
        group_file = GroupFile(
            group_id=group_id,
            thread_id=thread_id,
            uploaded_by=current_user.id,
            original_filename=original_filename,
            stored_filename=stored_filename,
            file_size=file_size,
            file_type=file_type,
            mime_type=mime_type,
            file_path=file_path,
            description=description
        )

        db.session.add(group_file)
        db.session.flush()  # Get the file ID

        # Create a message about the file upload if thread_id is provided
        message = None
        if thread_id:
            message_text = f"📎 Shared a file: {original_filename}"
            if description:
                message_text += f"\n📝 {description}"

            message = GroupMessage(
                group_id=group_id,
                thread_id=thread_id,
                user_id=current_user.id,
                message=message_text,
                message_type='file',
                file_id=group_file.id
            )
            db.session.add(message)

        db.session.commit()

        # =================== REAL-TIME SOCKETIO NOTIFICATION ===================
        try:
            from app import socketio
            room = f'group_{group_id}'
            
            # Create complete file data with all necessary fields
            file_data = {
                'id': group_file.id,
                'original_filename': group_file.original_filename,
                'stored_filename': group_file.stored_filename,
                'file_size': group_file.file_size,
                'file_size_formatted': f"{group_file.file_size / (1024*1024):.1f} MB" if group_file.file_size > 1024*1024 else f"{group_file.file_size / 1024:.1f} KB",
                'file_type': group_file.file_type,
                'mime_type': group_file.mime_type,
                'description': group_file.description,
                'uploaded_by': group_file.uploaded_by,
                'uploader_name': current_user.full_name,
                'uploaded_at': datetime.utcnow().isoformat(),
                'download_count': 0,
                'is_image': group_file.file_type == 'image' if hasattr(group_file, 'file_type') else False
            }
            
            # Emit file upload notification to ALL users (including uploader)
            socketio.emit('new-group-file', {
                'file': file_data,
                'thread_id': thread_id,
                'group_id': group_id,
                'uploader_name': current_user.full_name,
                'uploaded_at': datetime.utcnow().isoformat()
            }, room=room)
            
            # ALSO create a message notification for the chat if in a thread
            if message and thread_id:
                message_data = {
                    'id': message.id,
                    'user_id': current_user.id,
                    'username': current_user.username,
                    'full_name': current_user.full_name,
                    'message': message.message,
                    'message_type': 'file',
                    'shared_file': file_data,
                    'created_at': message.created_at.isoformat(),
                    'is_gpt_query': False,
                    'gpt_response': None
                }
                
                socketio.emit('new-group-message', {
                    'message': message_data,
                    'thread_id': thread_id,
                    'group_id': group_id
                }, room=room)
                
            logger.info(f"Real-time file upload notification sent to room {room} for file {original_filename}")
            
        except Exception as socket_error:
            # Don't fail the upload if SocketIO fails
            logger.warning(f"SocketIO notification failed for file upload: {socket_error}")

        # =================== END REAL-TIME NOTIFICATION ===================


        return jsonify({
            'success': True,
            'message': 'File uploaded successfully',
            'file': group_file.to_dict()
        }), 201

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error uploading group file: {e}")
        return jsonify({'error': 'Failed to upload file'}), 500


@group_study_bp.route('/api/groups/<int:group_id>/files/<int:file_id>/download', methods=['GET'])
def download_group_file(group_id, file_id):
    """Download a file from a group"""
    if not require_auth():
        return jsonify({'error': 'Unauthorized'}), 403

    try:
        current_user = get_current_user()
        if not current_user:
            return jsonify({'error': 'User not found'}), 404

        group = StudyGroup.query.get(group_id)
        if not group or not group.is_member(current_user.id):
            return jsonify({'error': 'Group not found or access denied'}), 403

        group_file = GroupFile.query.get(file_id)
        if not group_file or group_file.group_id != group_id or group_file.is_deleted:
            return jsonify({'error': 'File not found'}), 404

        if not os.path.exists(group_file.file_path):
            return jsonify({'error': 'File no longer exists on server'}), 404

        # Increment download count
        group_file.download_count += 1
        db.session.commit()

        return send_file(
            group_file.file_path,
            as_attachment=True,
            download_name=group_file.original_filename
        )

    except Exception as e:
        logger.error(f"Error downloading group file: {e}")
        return jsonify({'error': 'Failed to download file'}), 500

@group_study_bp.route('/api/groups/<int:group_id>/files/<int:file_id>/delete', methods=['DELETE'])
def delete_group_file(group_id, file_id):
    """Delete a file from a group (uploader or admin only)"""
    if not require_auth():
        return jsonify({'error': 'Unauthorized'}), 403

    try:
        current_user = get_current_user()
        if not current_user:
            return jsonify({'error': 'User not found'}), 404

        group = StudyGroup.query.get(group_id)
        if not group or not group.is_member(current_user.id):
            return jsonify({'error': 'Group not found or access denied'}), 403

        group_file = GroupFile.query.get(file_id)
        if not group_file or group_file.group_id != group_id:
            return jsonify({'error': 'File not found'}), 404

        # Check if user can delete (uploader or admin)
        if group_file.uploaded_by != current_user.id and not group.is_admin(current_user.id):
            return jsonify({'error': 'You can only delete files you uploaded or be a group admin'}), 403

        # Soft delete the file
        group_file.is_deleted = True
        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'File deleted successfully'
        }), 200

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting group file: {e}")
        return jsonify({'error': 'Failed to delete file'}), 500

@group_study_bp.route('/api/groups/<int:group_id>/files/<int:file_id>/info', methods=['GET'])  
def get_file_info(group_id, file_id):
    """Get detailed information about a specific file"""
    if not require_auth():
        return jsonify({'error': 'Unauthorized'}), 403

    try:
        current_user = get_current_user()
        if not current_user:
            return jsonify({'error': 'User not found'}), 404

        group = StudyGroup.query.get(group_id)
        if not group or not group.is_member(current_user.id):
            return jsonify({'error': 'Group not found or access denied'}), 403

        group_file = GroupFile.query.get(file_id)
        if not group_file or group_file.group_id != group_id or group_file.is_deleted:
            return jsonify({'error': 'File not found'}), 404

        return jsonify({
            'success': True,
            'file': group_file.to_dict()
        }), 200

    except Exception as e:
        logger.error(f"Error getting file info: {e}")
        return jsonify({'error': 'Failed to get file info'}), 500


@group_study_bp.route('/api/server-time', methods=['GET'])
def get_server_time():
    """Get server time for client synchronization"""
    try:
        server_time = get_ist_now()
        return jsonify({
            'success': True,
            'server_time': server_time.isoformat(),
            'timestamp': server_time.timestamp(),
            'timezone': 'Asia/Kolkata'
        }), 200
    except Exception as e:
        logger.error(f"Error getting server time: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to get server time'
        }), 500


# ============= ADDITIONAL UTILITY ENDPOINTS =============

@group_study_bp.route('/api/groups/<int:group_id>/stats', methods=['GET'])
def get_group_stats(group_id):
    """Get group statistics"""
    if not require_auth():
        return jsonify({'error': 'Unauthorized'}), 403

    try:
        current_user = get_current_user()
        if not current_user:
            return jsonify({'error': 'User not found'}), 404

        group = StudyGroup.query.get(group_id)
        if not group or not group.is_member(current_user.id):
            return jsonify({'error': 'Group not found or access denied'}), 403

        # Get statistics
        total_members = group.get_member_count()
        total_threads = GroupThread.query.filter_by(group_id=group_id, is_active=True).count()
        total_messages = GroupMessage.query.filter_by(group_id=group_id, is_deleted=False).count()
        total_files = GroupFile.query.filter_by(group_id=group_id, is_deleted=False).count()

        # Get recent activity (last 7 days)
        from datetime import timedelta
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent_messages = GroupMessage.query.filter(
            GroupMessage.group_id == group_id,
            GroupMessage.created_at >= week_ago,
            GroupMessage.is_deleted == False
        ).count()

        recent_files = GroupFile.query.filter(
            GroupFile.group_id == group_id,
            GroupFile.uploaded_at >= week_ago,
            GroupFile.is_deleted == False
        ).count()

        return jsonify({
            'success': True,
            'stats': {
                'total_members': total_members,
                'total_threads': total_threads,
                'total_messages': total_messages,
                'total_files': total_files,
                'recent_messages': recent_messages,
                'recent_files': recent_files,
                'group_age_days': (datetime.utcnow() - group.created_at).days
            }
        }), 200

    except Exception as e:
        logger.error(f"Error getting group stats: {e}")
        return jsonify({'error': 'Failed to get group stats'}), 500
