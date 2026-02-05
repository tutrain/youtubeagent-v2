"""
TuTrain Educator Discovery Engine - Phase 9 (CSV Deduplication)
================================================================
A Streamlit app with Deep Filtering pagination that ensures
target lead counts are met with FULLY APPROVED channels.

Phase 9 CSV Deduplication Upgrades:
- Upload Master Lead File (CSV) to exclude previously discovered channels
- Early deduplication check BEFORE expensive AI/Activity calls
- Channel ID in CSV export for future-proofing
- Target count applies to NEW UNIQUE leads only

Phase 8 Deep Filtering Features (preserved):
- AI classification + activity checks INSIDE the fetch loop
- Dynamic query rotation when primary query exhausts
- Quota protection with safety limits
- UI shows "Approved vs Scanned" progress

API Limits:
- YouTube Data API: 10,000 units/day
- Gemini API: ~15 requests/minute (free tier)
"""

# ============================================================================
# WARNING SUPPRESSION - Must be at the very top before other imports trigger warnings
# ============================================================================
import warnings
import os

# Suppress all Python warnings to keep terminal clean
warnings.filterwarnings("ignore")

# Suppress specific Google/GRPC warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

# ============================================================================
# IMPORTS
# ============================================================================
import streamlit as st
import pandas as pd
import plotly.express as px
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import google.generativeai as genai
import time
import re
from datetime import datetime, timedelta


# ============================================================================
# CONFIGURATION
# ============================================================================

# API Key - Hardcoded (works for both YouTube Data API v3 and Gemini)
GOOGLE_API_KEY = "AIzaSyDuUAfhGWq7Y1l1_UixiKI4nxEKFtDb4zw"

# Institute Keywords for heuristic classification (catches coaching centers before AI call)
INSTITUTE_KEYWORDS = [
    'Academy', 'Institute', 'Classes', 'Coaching', 'Tutorial', 'School', 
    'Education', 'Center', 'Hub', 'Campus', 'Wallah', 'Group', 'Team', 
    'System', 'Official', 'Centre', 'Learning', 'Foundation'
]

# Hard filter constraints
MIN_SUBSCRIBERS = 5000
MAX_SUBSCRIBERS = 500000
MIN_VIDEOS = 10

# Activity check constraints
INACTIVE_DAYS_THRESHOLD = 45
LOW_FREQUENCY_DAYS = 30
MIN_UPLOADS_IN_PERIOD = 2

# Deep Filtering - Quota Protection Safety Limits
MAX_SEARCH_PAGES = 10           # Max YouTube search pages to fetch
MAX_GEMINI_CALLS = 150          # Max Gemini API classification calls
QUERY_VARIANTS_ENABLED = True   # Enable automatic query rotation

# YouTube API settings
YOUTUBE_API_SERVICE = "youtube"
YOUTUBE_API_VERSION = "v3"
REGION_CODE = "IN"


# ============================================================================
# QUERY VARIANT GENERATOR (NEW - Phase 8)
# ============================================================================

def generate_query_variants(subject: str) -> list:
    """
    Generate search query variants for dynamic rotation.
    Returns list of queries to try when primary query exhausts results.
    
    Args:
        subject: Primary subject to search
        
    Returns:
        List of query variants to rotate through
    """
    base = subject.strip()
    variants = [
        f"{base} tutorial",
        f"{base} class",
        f"{base} teacher",
        f"{base} coaching",
        f"Best {base} teacher",
        f"{base} lectures",
        f"{base} online class",
        f"{base} CBSE",
        f"{base} ICSE",
        f"{base} explained",
    ]
    return variants


# ============================================================================
# CSV DEDUPLICATION (NEW - Phase 9)
# ============================================================================

def load_existing_leads(uploaded_file) -> tuple:
    """
    Load existing leads from an uploaded CSV file for deduplication.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Tuple of (existing_links_set, existing_ids_set, count, error_message)
        - existing_links_set: Set of channel links (normalized)
        - existing_ids_set: Set of channel IDs (if available)
        - count: Number of leads loaded
        - error_message: Error string if any issue, else empty string
    """
    existing_links = set()
    existing_ids = set()
    
    try:
        df = pd.read_csv(uploaded_file)
        
        if df.empty:
            return set(), set(), 0, "Uploaded file is empty"
        
        # Try to load Channel Links
        if "Channel Link" in df.columns:
            links = df["Channel Link"].dropna().astype(str).str.strip()
            # Normalize links (remove trailing slashes, lowercase)
            for link in links:
                if link and link.lower() != "nan":
                    normalized = link.lower().rstrip("/")
                    existing_links.add(normalized)
        else:
            return set(), set(), 0, "Missing 'Channel Link' column in uploaded file"
        
        # Also try to load Channel IDs if available (future-proofing)
        if "Channel ID" in df.columns:
            ids = df["Channel ID"].dropna().astype(str).str.strip()
            for cid in ids:
                if cid and cid.lower() != "nan":
                    existing_ids.add(cid)
        
        return existing_links, existing_ids, len(existing_links), ""
        
    except Exception as e:
        return set(), set(), 0, f"Error reading file: {str(e)}"


# ============================================================================
# STRICT CONTACT EXTRACTION
# ============================================================================

def extract_contacts(description: str) -> dict:
    """
    Extract contact information using STRICT regex patterns.
    Only matches properly formatted URLs/links, not random text.
    
    Args:
        description: Channel description text
        
    Returns:
        Dictionary with keys: email, instagram, telegram, facebook, twitter, website
    """
    if not description:
        description = ""
    
    contacts = {
        "email": "",
        "instagram": "",
        "telegram": "",
        "facebook": "",
        "twitter": "",
        "website": ""
    }
    
    # Email pattern - standard format
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails = re.findall(email_pattern, description, re.IGNORECASE)
    # Filter out common false positives
    valid_emails = [e.strip('.,;:!?') for e in emails if not e.endswith('.jpg') and not e.endswith('.png')]
    if valid_emails:
        contacts["email"] = ", ".join(list(set(valid_emails))[:3])
    
    # Instagram - STRICT: must have instagram.com/
    insta_pattern = r'(?:https?://)?(?:www\.)?instagram\.com/([\w\.]+)'
    insta_matches = re.findall(insta_pattern, description, re.IGNORECASE)
    if insta_matches:
        # Clean and deduplicate
        clean_insta = [u.strip('.,;:!?/') for u in insta_matches if len(u) > 1]
        if clean_insta:
            contacts["instagram"] = ", ".join([f"instagram.com/{u}" for u in list(set(clean_insta))[:2]])
    
    # Telegram - STRICT: must have t.me/ or telegram.me/
    telegram_pattern = r'(?:https?://)?(?:t\.me|telegram\.me)/([\w_]+)'
    telegram_matches = re.findall(telegram_pattern, description, re.IGNORECASE)
    if telegram_matches:
        clean_tg = [u.strip('.,;:!?/') for u in telegram_matches if len(u) > 1]
        if clean_tg:
            contacts["telegram"] = ", ".join([f"t.me/{u}" for u in list(set(clean_tg))[:2]])
    
    # Facebook - STRICT: must have facebook.com/
    fb_pattern = r'(?:https?://)?(?:www\.)?facebook\.com/([\w\.]+)'
    fb_matches = re.findall(fb_pattern, description, re.IGNORECASE)
    if fb_matches:
        clean_fb = [u.strip('.,;:!?/') for u in fb_matches if len(u) > 1]
        if clean_fb:
            contacts["facebook"] = ", ".join([f"facebook.com/{u}" for u in list(set(clean_fb))[:2]])
    
    # Twitter/X - STRICT: must have twitter.com/ or x.com/
    twitter_pattern = r'(?:https?://)?(?:www\.)?(?:twitter\.com|x\.com)/([\w_]+)'
    twitter_matches = re.findall(twitter_pattern, description, re.IGNORECASE)
    if twitter_matches:
        clean_tw = [u.strip('.,;:!?/') for u in twitter_matches if len(u) > 1 and u.lower() not in ['intent', 'share']]
        if clean_tw:
            contacts["twitter"] = ", ".join([f"twitter.com/{u}" for u in list(set(clean_tw))[:2]])
    
    # Website - STRICT: MUST start with http:// or https://
    # Only match full URLs, not bare domains
    website_pattern = r'https?://(?:www\.)?([a-zA-Z0-9][-a-zA-Z0-9]*(?:\.[a-zA-Z0-9][-a-zA-Z0-9]*)+(?:/[^\s<>\"\']*)?)'
    all_urls = re.findall(website_pattern, description, re.IGNORECASE)
    
    # Filter out social media domains
    social_domains = ['instagram.com', 't.me', 'telegram.me', 'facebook.com', 
                      'twitter.com', 'x.com', 'youtube.com', 'youtu.be', 
                      'fb.com', 'wa.me', 'whatsapp.com', 'bit.ly', 'goo.gl',
                      'tinyurl.com', 'linktr.ee', 'docs.google.com']
    
    websites = []
    for url in all_urls:
        url_clean = url.strip('.,;:!?/\'"')
        is_social = any(domain in url_clean.lower() for domain in social_domains)
        if not is_social and url_clean not in websites and len(url_clean) > 5:
            websites.append(url_clean)
    
    if websites:
        contacts["website"] = ", ".join(websites[:2])
    
    return contacts


# ============================================================================
# ACTIVITY CHECK
# ============================================================================

def check_channel_activity(youtube, uploads_playlist_id: str) -> dict:
    """
    Check if a channel is active based on recent upload history.
    """
    result = {
        "activity_status": "Unknown",
        "is_active": True,
        "reason": ""
    }
    
    if not uploads_playlist_id:
        result["activity_status"] = "No uploads"
        result["is_active"] = False
        result["reason"] = "No uploads found"
        return result
    
    try:
        playlist_response = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=uploads_playlist_id,
            maxResults=5
        ).execute()
        
        items = playlist_response.get("items", [])
        
        if not items:
            result["activity_status"] = "No videos"
            result["is_active"] = False
            result["reason"] = "No videos uploaded"
            return result
        
        # Get video publish dates
        video_dates = []
        for item in items:
            published_at = item.get("contentDetails", {}).get("videoPublishedAt")
            if published_at:
                try:
                    date_str = published_at.replace("Z", "+00:00")
                    video_date = datetime.fromisoformat(date_str)
                    video_dates.append(video_date)
                except:
                    pass
        
        if not video_dates:
            result["activity_status"] = "Unknown"
            result["is_active"] = True
            return result
        
        video_dates.sort(reverse=True)
        now = datetime.now(video_dates[0].tzinfo)
        most_recent = video_dates[0]
        days_since_last = (now - most_recent).days
        
        # Rule 1: If most recent video is older than 45 days = Inactive
        if days_since_last > INACTIVE_DAYS_THRESHOLD:
            result["activity_status"] = f"Last: {days_since_last}d ago"
            result["is_active"] = False
            result["reason"] = "Inactive"
            return result
        
        # Rule 2: Check uploads in last 30 days
        thirty_days_ago = now - timedelta(days=LOW_FREQUENCY_DAYS)
        recent_uploads = sum(1 for d in video_dates if d > thirty_days_ago)
        
        if recent_uploads < MIN_UPLOADS_IN_PERIOD:
            result["activity_status"] = f"{recent_uploads} in 30d"
            result["is_active"] = False
            result["reason"] = "Low Frequency"
            return result
        
        result["activity_status"] = f"Active ({recent_uploads} in 30d)"
        result["is_active"] = True
        return result
        
    except Exception:
        result["activity_status"] = "Check failed"
        result["is_active"] = True
        return result


# ============================================================================
# TIER SCORING
# ============================================================================

def calculate_tier(is_active: bool, contacts: dict, avg_views: int, subscribers: int) -> str:
    """
    Calculate lead tier based on activity, contacts, and engagement.
    """
    if not is_active:
        return "D"
    
    has_email = bool(contacts.get("email"))
    has_telegram = bool(contacts.get("telegram"))
    has_any_contact = any([
        contacts.get("email"),
        contacts.get("instagram"),
        contacts.get("telegram"),
        contacts.get("facebook"),
        contacts.get("twitter"),
        contacts.get("website")
    ])
    
    engagement_ratio = avg_views / subscribers if subscribers > 0 else 0
    high_engagement = engagement_ratio > 0.05 or avg_views > 10000
    
    if (has_email or has_telegram) and high_engagement:
        return "A"
    
    if has_any_contact:
        return "B"
    
    return "C"




# ============================================================================
# GEMINI CLASSIFICATION
# ============================================================================

def classify_channel(name: str, description: str, api_key: str) -> str:
    """
    Use Gemini AI to classify a channel into three categories:
    - Individual: Single teacher/educator
    - Small Institute: Small coaching center, 2-3 teachers, local academy
    - Large Brand: Big EdTech (Byju's, Unacademy), News channels, media aggregators
    
    We ACCEPT Individual and Small Institute, REJECT only Large Brand.
    
    KEYWORD HEURISTIC: First checks if channel name contains institute keywords
    to force "Small Institute" classification for coaching centers.
    """
    
    # ==== STEP A: Keyword Heuristic Check (Pre-AI) ====
    # Check if channel name contains any institute keywords (case-insensitive)
    name_lower = name.lower()
    has_institute_keyword = any(keyword.lower() in name_lower for keyword in INSTITUTE_KEYWORDS)
    
    # ==== STEP B: If keyword found, force "Small Institute" ====
    if has_institute_keyword:
        return "Small Institute"
    
    # ==== STEP C: No keyword found, proceed with Gemini AI classification ====
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        prompt = f"""Analyze this YouTube channel. Classify it into one of these three categories:

'Individual': A single teacher/educator running their own channel.
'Small Institute': A small coaching center, a group of 2-3 teachers, a local academy, or a Multi-Subject coaching institute run by a small team.
'Large Brand': A massive EdTech company (like Byju's, Unacademy, Physics Wallah), a News channel, or a generic media aggregator.

Return ONLY the category name.

Channel: {name}
Description: {description[:400]}

Category:"""
        
        response = model.generate_content(prompt)
        result = response.text.strip().lower()
        
        if "individual" in result:
            return "Individual"
        elif "small" in result or "small institute" in result:
            return "Small Institute"
        elif "large" in result or "brand" in result:
            return "Large Brand"
        # Default to Individual if unclear (benefit of doubt)
        return "Individual"
            
    except Exception as e:
        # ==== STEP D: Fallback when API fails/times out ====
        # Re-check keywords as fallback (in case we missed them)
        if has_institute_keyword:
            return "Small Institute"
        # No keyword -> return "Individual" (never return "Unknown")
        return "Individual"


# ============================================================================
# DEEP FILTERING FETCH (REFACTORED - Phase 8)
# ============================================================================

def smart_fetch_channels(
    youtube, 
    subject: str, 
    target_count: int, 
    gemini_api_key: str,
    status_container,
    existing_channel_ids: set = None,
    existing_links: set = None
) -> list:
    """
    Deep Filtering Fetch Loop: Continues fetching and filtering until we have
    enough FULLY APPROVED channels (passed ALL criteria).
    
    Phase 9 Changes:
    - CSV-based deduplication via existing_links parameter
    - Early duplicate check BEFORE expensive AI/Activity calls
    - Target count applies to NEW UNIQUE leads only
    
    Phase 8 Features (preserved):
    - AI Classification runs INSIDE the loop
    - Activity Check runs INSIDE the loop
    - Only counts channels that pass ALL filters
    - Dynamic query rotation when primary query exhausts
    - Quota protection with safety limits
    
    Args:
        youtube: YouTube API client
        subject: Search query
        target_count: Number of APPROVED channels needed
        gemini_api_key: API key for Gemini classification
        status_container: Streamlit status container for progress updates
        existing_channel_ids: Optional set of channel IDs to skip (deduplication)
        existing_links: Optional set of channel links to skip (CSV deduplication)
        
    Returns:
        List of FULLY APPROVED channel dictionaries
    """
    approved_channels = []
    seen_channel_ids = existing_channel_ids.copy() if existing_channel_ids else set()
    existing_links_normalized = existing_links if existing_links else set()
    
    # Generate query variants for rotation
    query_variants = generate_query_variants(subject) if QUERY_VARIANTS_ENABLED else [f"{subject} tutorial"]
    current_query_idx = 0
    current_query = query_variants[current_query_idx]
    
    # Tracking counters
    pages_fetched = 0
    gemini_calls = 0
    total_scanned = 0
    rejected_large_brand = 0
    rejected_inactive = 0
    rejected_duplicate_csv = 0
    rejected_basic_filter = 0
    
    next_page_token = None
    
    status_container.write(f"🚀 Starting Deep Filtering for: {subject}")
    status_container.write(f"🎯 Target: {target_count} approved leads")
    
    while len(approved_channels) < target_count:
        # Safety limit: Max search pages
        if pages_fetched >= MAX_SEARCH_PAGES:
            status_container.write(f"⚠️ Reached max search pages limit ({MAX_SEARCH_PAGES})")
            break
        
        # Safety limit: Max Gemini calls
        if gemini_calls >= MAX_GEMINI_CALLS:
            status_container.write(f"⚠️ Reached max Gemini API calls limit ({MAX_GEMINI_CALLS})")
            break
        
        pages_fetched += 1
        
        status_container.write(f"📡 Fetching page {pages_fetched} | Query: '{current_query[:30]}...'")
        
        try:
            # Search for channels
            search_params = {
                "q": current_query,
                "type": "channel",
                "part": "snippet",
                "regionCode": REGION_CODE,
                "maxResults": 50,
                "relevanceLanguage": "en"
            }
            
            if next_page_token:
                search_params["pageToken"] = next_page_token
            
            search_response = youtube.search().list(**search_params).execute()
            
            items = search_response.get("items", [])
            next_page_token = search_response.get("nextPageToken")
            
            if not items:
                status_container.write(f"⚠️ No results for query: {current_query[:30]}...")
                # Try next query variant
                current_query_idx += 1
                if current_query_idx >= len(query_variants):
                    status_container.write("📄 Exhausted all query variants")
                    break
                current_query = query_variants[current_query_idx]
                next_page_token = None
                status_container.write(f"🔄 Rotating to query: '{current_query[:30]}...'")
                continue
            
            # Get channel IDs (excluding already seen)
            channel_ids = []
            for item in items:
                ch_id = item["snippet"]["channelId"]
                if ch_id not in seen_channel_ids:
                    channel_ids.append(ch_id)
                    seen_channel_ids.add(ch_id)
            
            if not channel_ids:
                status_container.write(f"⏭️ All channels on this page already processed")
                if not next_page_token:
                    # Try next query variant
                    current_query_idx += 1
                    if current_query_idx >= len(query_variants):
                        break
                    current_query = query_variants[current_query_idx]
                    next_page_token = None
                continue
            
            total_scanned += len(channel_ids)
            
            # Fetch full channel details
            channels_response = youtube.channels().list(
                id=",".join(channel_ids),
                part="snippet,statistics,contentDetails"
            ).execute()
            
            # Process each channel with FULL filtering
            for channel in channels_response.get("items", []):
                # Check if we've reached target
                if len(approved_channels) >= target_count:
                    break
                
                # Check Gemini quota
                if gemini_calls >= MAX_GEMINI_CALLS:
                    break
                
                stats = channel.get("statistics", {})
                snippet = channel.get("snippet", {})
                content_details = channel.get("contentDetails", {})
                
                channel_id = channel["id"]
                channel_name = snippet.get("title", "Unknown")
                subscriber_count = int(stats.get("subscriberCount", 0))
                video_count = int(stats.get("videoCount", 0))
                view_count = int(stats.get("viewCount", 0))
                full_description = snippet.get("description", "")
                
                # ============ BASIC FILTERS ============
                if subscriber_count < MIN_SUBSCRIBERS or subscriber_count > MAX_SUBSCRIBERS:
                    rejected_basic_filter += 1
                    continue
                if video_count < MIN_VIDEOS:
                    rejected_basic_filter += 1
                    continue
                
                # ============ CSV DEDUPLICATION (Phase 9 - Early Check!) ============
                # Generate channel link to check against existing leads
                custom_url_check = snippet.get("customUrl", "")
                if custom_url_check:
                    link_check = f"https://www.youtube.com/{custom_url_check}".lower().rstrip("/")
                else:
                    link_check = f"https://www.youtube.com/channel/{channel_id}".lower().rstrip("/")
                
                if link_check in existing_links_normalized:
                    rejected_duplicate_csv += 1
                    status_container.write(f"  ⏭️ {channel_name[:25]}... → Already in Master CSV")
                    continue
                
                # ============ AI CLASSIFICATION (Inside Loop!) ============
                gemini_calls += 1
                classification = classify_channel(channel_name, full_description, gemini_api_key)
                
                if classification == "Large Brand":
                    rejected_large_brand += 1
                    status_container.write(f"  ❌ {channel_name[:25]}... → Large Brand")
                    time.sleep(1)  # Rate limit for Gemini
                    continue
                
                # ============ ACTIVITY CHECK (Inside Loop!) ============
                uploads_playlist_id = content_details.get("relatedPlaylists", {}).get("uploads", "")
                activity = check_channel_activity(youtube, uploads_playlist_id)
                
                if not activity["is_active"]:
                    rejected_inactive += 1
                    status_container.write(f"  ⏸️ {channel_name[:25]}... → {activity['reason']}")
                    time.sleep(0.5)
                    continue
                
                # ============ CHANNEL PASSES ALL FILTERS! ============
                avg_views = int(view_count / video_count) if video_count > 0 else 0
                contacts = extract_contacts(full_description)
                
                # Calculate tier
                tier = calculate_tier(activity["is_active"], contacts, avg_views, subscriber_count)
                
                # Generate channel link
                custom_url = snippet.get("customUrl", "")
                if custom_url:
                    channel_link = f"https://www.youtube.com/{custom_url}"
                else:
                    channel_link = f"https://www.youtube.com/channel/{channel_id}"
                
                channel_info = {
                    "channel_id": channel_id,
                    "name": channel_name,
                    "description": full_description,
                    "custom_url": custom_url,
                    "channel_link": channel_link,
                    "subscribers": subscriber_count,
                    "video_count": video_count,
                    "total_views": view_count,
                    "avg_views": avg_views,
                    "uploads_playlist_id": uploads_playlist_id,
                    "classification": classification,
                    "activity_status": activity["activity_status"],
                    "is_active": activity["is_active"],
                    "reason": "",
                    "status": "Approved",
                    "tier": tier,
                    "subject_tag": subject,
                    **contacts
                }
                
                approved_channels.append(channel_info)
                
                status_container.write(
                    f"  ✅ {channel_name[:25]}... → {classification} | {activity['activity_status']} | Tier {tier}"
                )
                
                # Rate limit delay
                time.sleep(1)
            
            # Progress update
            status_container.write(
                f"📊 **Approved: {len(approved_channels)}/{target_count}** | "
                f"Scanned: {total_scanned} | Pages: {pages_fetched}/{MAX_SEARCH_PAGES} | "
                f"Gemini: {gemini_calls}/{MAX_GEMINI_CALLS}"
            )
            
            # Check if we should rotate query
            if not next_page_token:
                current_query_idx += 1
                if current_query_idx >= len(query_variants):
                    status_container.write("📄 Exhausted all query variants")
                    break
                current_query = query_variants[current_query_idx]
                next_page_token = None
                status_container.write(f"🔄 Rotating to query: '{current_query[:30]}...'")
                
        except HttpError as e:
            status_container.write(f"⚠️ API Error: {e.reason}")
            break
        except Exception as e:
            status_container.write(f"⚠️ Error: {str(e)}")
            break
    
    # Final summary
    status_container.write("─" * 50)
    status_container.write(f"🏁 **Deep Filtering Complete**")
    status_container.write(f"   ✅ Approved: {len(approved_channels)}")
    status_container.write(f"   📊 Total Scanned: {total_scanned}")
    status_container.write(f"   ⏭️ Skipped (CSV Duplicates): {rejected_duplicate_csv}")
    status_container.write(f"   ❌ Rejected (Large Brand): {rejected_large_brand}")
    status_container.write(f"   ⏸️ Rejected (Inactive): {rejected_inactive}")
    status_container.write(f"   🔢 Rejected (Basic Filters): {rejected_basic_filter}")
    status_container.write(f"   📡 API Pages Used: {pages_fetched}")
    status_container.write(f"   🤖 Gemini Calls: {gemini_calls}")
    
    return approved_channels


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_number(num: int) -> str:
    """Format large numbers for display"""
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    return str(num)


def get_api_key():
    """Get the Google API key"""
    return GOOGLE_API_KEY


# ============================================================================
# DASHBOARD COMPONENTS
# ============================================================================

def render_kpi_metrics(df: pd.DataFrame):
    """
    Render the KPI Metrics row at the top of the dashboard.
    
    Displays:
    - Total Found: Total rows fetched
    - Approved Leads: Count of Status == 'Approved'
    - Avg Subscribers: Mean of Approved channels
    - Avg Engagement: Mean of Avg Views for Approved channels
    """
    approved_df = df[df["status"] == "Approved"]
    
    total_found = len(df)
    approved_leads = len(approved_df)
    avg_subscribers = int(approved_df["subscribers"].mean()) if len(approved_df) > 0 else 0
    avg_engagement = int(approved_df["avg_views"].mean()) if len(approved_df) > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); 
                    padding: 1.5rem; border-radius: 12px; text-align: center;
                    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);">
            <p style="color: #dbeafe; margin: 0; font-size: 0.9rem; font-weight: 500;">📊 TOTAL FOUND</p>
            <h2 style="color: #ffffff; margin: 0.5rem 0 0 0; font-size: 2.5rem; font-weight: 700;">{}</h2>
        </div>
        """.format(total_found), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                    padding: 1.5rem; border-radius: 12px; text-align: center;
                    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);">
            <p style="color: #d1fae5; margin: 0; font-size: 0.9rem; font-weight: 500;">✅ APPROVED LEADS</p>
            <h2 style="color: #ffffff; margin: 0.5rem 0 0 0; font-size: 2.5rem; font-weight: 700;">{}</h2>
        </div>
        """.format(approved_leads), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); 
                    padding: 1.5rem; border-radius: 12px; text-align: center;
                    box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3);">
            <p style="color: #ede9fe; margin: 0; font-size: 0.9rem; font-weight: 500;">👥 AVG SUBSCRIBERS</p>
            <h2 style="color: #ffffff; margin: 0.5rem 0 0 0; font-size: 2.5rem; font-weight: 700;">{}</h2>
        </div>
        """.format(format_number(avg_subscribers)), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); 
                    padding: 1.5rem; border-radius: 12px; text-align: center;
                    box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3);">
            <p style="color: #fef3c7; margin: 0; font-size: 0.9rem; font-weight: 500;">📈 AVG ENGAGEMENT</p>
            <h2 style="color: #ffffff; margin: 0.5rem 0 0 0; font-size: 2.5rem; font-weight: 700;">{}</h2>
        </div>
        """.format(format_number(avg_engagement)), unsafe_allow_html=True)


def render_quality_map(df: pd.DataFrame):
    """
    Render the Quality Map scatter plot.
    
    Shows Subscribers vs Avg Views for Approved channels,
    colored by Tier to help spot high-engagement, low-subscriber gems.
    """
    approved_df = df[df["status"] == "Approved"].copy()
    
    if len(approved_df) == 0:
        st.info("No approved channels to display in Quality Map.")
        return
    
    # Define tier colors for consistency
    tier_colors = {
        "A": "#10b981",  # Green
        "B": "#3b82f6",  # Blue  
        "C": "#f59e0b",  # Amber
        "D": "#ef4444"   # Red
    }
    
    fig = px.scatter(
        approved_df,
        x="subscribers",
        y="avg_views",
        color="tier",
        color_discrete_map=tier_colors,
        hover_data={"name": True, "classification": True, "subscribers": ":,", "avg_views": ":,"},
        labels={
            "subscribers": "Subscribers",
            "avg_views": "Avg Views per Video",
            "tier": "Tier",
            "name": "Channel",
            "classification": "Type"
        },
        title="🎯 Quality Map: Find High-Engagement Gems"
    )
    
    fig.update_traces(
        marker=dict(size=12, line=dict(width=1, color='white')),
        hovertemplate="<b>%{customdata[0]}</b><br>" +
                      "Type: %{customdata[1]}<br>" +
                      "Subscribers: %{x:,}<br>" +
                      "Avg Views: %{y:,}<br>" +
                      "<extra></extra>"
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248,250,252,1)',
        font=dict(family="Inter, sans-serif"),
        title_font=dict(size=18, color="#1e293b"),
        legend=dict(
            title="Tier",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            title_font=dict(size=12, color="#64748b"),
            tickfont=dict(size=10, color="#64748b"),
            gridcolor='rgba(226, 232, 240, 0.8)'
        ),
        yaxis=dict(
            title_font=dict(size=12, color="#64748b"),
            tickfont=dict(size=10, color="#64748b"),
            gridcolor='rgba(226, 232, 240, 0.8)'
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Inter, sans-serif"
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_pipeline_distribution(df: pd.DataFrame):
    """
    Render the Pipeline Distribution donut chart.
    
    Shows the breakdown by Status (Approved vs Rejected) and a secondary
    donut for Tier distribution among approved leads.
    """
    # Status Distribution Donut
    status_counts = df["status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]
    
    status_colors = {
        "Approved": "#10b981",
        "Rejected": "#ef4444"
    }
    
    fig = px.pie(
        status_counts,
        values="Count",
        names="Status",
        color="Status",
        color_discrete_map=status_colors,
        hole=0.5,
        title="📊 Pipeline Distribution"
    )
    
    fig.update_traces(
        textposition='outside',
        textinfo='label+percent+value',
        textfont_size=12,
        marker=dict(line=dict(color='white', width=2)),
        hovertemplate="<b>%{label}</b><br>" +
                      "Count: %{value}<br>" +
                      "Percentage: %{percent}<br>" +
                      "<extra></extra>"
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif"),
        title_font=dict(size=18, color="#1e293b"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5
        ),
        annotations=[dict(
            text='<b>Status</b>',
            x=0.5, y=0.5,
            font_size=14,
            font_color="#64748b",
            showarrow=False
        )]
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tier Distribution (for Approved only)
    approved_df = df[df["status"] == "Approved"]
    
    if len(approved_df) > 0:
        tier_counts = approved_df["tier"].value_counts().reset_index()
        tier_counts.columns = ["Tier", "Count"]
        
        tier_colors = {
            "A": "#10b981",
            "B": "#3b82f6",
            "C": "#f59e0b",
            "D": "#ef4444"
        }
        
        fig_tier = px.pie(
            tier_counts,
            values="Count",
            names="Tier",
            color="Tier",
            color_discrete_map=tier_colors,
            hole=0.5,
            title="🏅 Tier Breakdown (Approved Only)"
        )
        
        fig_tier.update_traces(
            textposition='outside',
            textinfo='label+percent+value',
            textfont_size=12,
            marker=dict(line=dict(color='white', width=2)),
            hovertemplate="<b>Tier %{label}</b><br>" +
                          "Count: %{value}<br>" +
                          "Percentage: %{percent}<br>" +
                          "<extra></extra>"
        )
        
        fig_tier.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif"),
            title_font=dict(size=18, color="#1e293b"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.1,
                xanchor="center",
                x=0.5
            ),
            annotations=[dict(
                text='<b>Tiers</b>',
                x=0.5, y=0.5,
                font_size=14,
                font_color="#64748b",
                showarrow=False
            )]
        )
        
        st.plotly_chart(fig_tier, use_container_width=True)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.set_page_config(
        page_title="TuTrain Educator Discovery",
        page_icon="🎓",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(135deg, #1e3a8a 0%, #7c3aed 100%);
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }
        .main-header h1 { color: #ffffff !important; margin: 0; font-weight: 700; }
        .main-header p { color: #e0e7ff !important; margin: 0.5rem 0 0 0; font-size: 1.1rem; }
        
        .filter-info {
            background: #f0f9ff;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #0ea5e9;
            color: #1e293b !important;
        }
        .filter-info b { color: #0f172a !important; }
        
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%) !important;
            color: #ffffff !important;
            border: none !important;
            font-weight: 600 !important;
        }
        
        .stDownloadButton > button {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
            color: #ffffff !important;
            border: none !important;
            font-weight: 600 !important;
        }
        
        [data-testid="stMetricLabel"] { color: #374151 !important; }
        [data-testid="stMetricValue"] { color: #1e3a8a !important; }
        
        .dashboard-section {
            background: #ffffff;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🎓 TuTrain Educator Discovery Engine</h1>
            <p>Phase 9: CSV Deduplication - Skip channels you already have!</p>
        </div>
    """, unsafe_allow_html=True)
    
    api_key = get_api_key()
    
    # Sidebar
    with st.sidebar:
        st.subheader("🎯 Target Settings")
        target_count = st.slider(
            "Target Lead Count",
            min_value=5,
            max_value=100,
            value=25,
            step=5,
            help="Number of APPROVED leads to find (will keep searching until reached)"
        )
        
        st.divider()
        
        st.subheader("📋 Active Filters")
        st.markdown(f"""
        <div class="filter-info">
        <b>Region:</b> India 🇮🇳<br>
        <b>Subscribers:</b> {format_number(MIN_SUBSCRIBERS)} - {format_number(MAX_SUBSCRIBERS)}<br>
        <b>Min Videos:</b> {MIN_VIDEOS}+<br>
        <b>AI Filter:</b> Individual + Small Institute<br>
        <b>Activity:</b> Active channels<br>
        <b>Max Pages:</b> {MAX_SEARCH_PAGES}<br>
        <b>Max Gemini:</b> {MAX_GEMINI_CALLS}
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        st.subheader("📊 Tier Scoring")
        st.markdown("""
        **A** 🟢 Email/Telegram + Engaged  
        **B** 🔵 Has Contact Info  
        **C** 🟡 Active, No Contact  
        **D** 🔴 Inactive
        """)
        
        st.divider()
        
        # ===== Phase 9: CSV Deduplication File Uploader =====
        st.subheader("📂 Deduplication")
        master_csv = st.file_uploader(
            "Upload Master Lead File (CSV)",
            type=["csv"],
            help="Upload a CSV from a previous run to exclude already discovered channels"
        )
        
        # Process uploaded file and store in session state
        if master_csv is not None:
            existing_links, existing_ids, lead_count, error_msg = load_existing_leads(master_csv)
            st.session_state["existing_links"] = existing_links
            st.session_state["existing_ids"] = existing_ids
            
            if error_msg:
                st.warning(f"⚠️ {error_msg}")
            elif lead_count > 0:
                st.success(f"✅ Loaded {lead_count} existing leads to exclude")
        else:
            # Clear session state if no file uploaded
            st.session_state["existing_links"] = set()
            st.session_state["existing_ids"] = set()
        
        st.divider()
        
        st.subheader("🆕 Phase 9 Features")
        st.markdown("""
        ✅ **CSV Deduplication** upload  
        ✅ **Early Skip** before AI calls  
        ✅ **Channel ID** in export  
        ✅ All Phase 8 features preserved
        """)
    
    # Main content
    subject = st.text_input(
        "🔍 Enter Subject to Search",
        placeholder="e.g., Physics Class 12, Python Programming, NEET Biology"
    )
    
    # Search button
    if st.button("🚀 Discover Educators", type="primary", use_container_width=True):
        if not subject:
            st.error("Please enter a subject to search!")
            return
        
        youtube = build(YOUTUBE_API_SERVICE, YOUTUBE_API_VERSION, developerKey=api_key)
        
        # Phase 1+2 Combined: Deep Filtering Fetch
        with st.status("🔍 Deep Filtering Search...", expanded=True) as status:
            status.write("Starting Deep Filtering loop (AI + Activity checks inside)...")
            
            # Get existing links from session state (Phase 9 deduplication)
            existing_links_from_csv = st.session_state.get("existing_links", set())
            
            # All returned channels are FULLY APPROVED (and unique from CSV)
            results = smart_fetch_channels(
                youtube, 
                subject, 
                target_count, 
                api_key,
                status,
                existing_links=existing_links_from_csv
            )
            
            if not results:
                status.update(label="❌ No approved channels found", state="error")
                st.warning("No channels found matching ALL criteria. Try a different subject.")
                return
            
            status.update(label=f"✅ Found {len(results)} approved channels", state="complete")
        

        
        # Create DataFrame for dashboard
        df = pd.DataFrame(results)
        
        # =====================================================================
        # PHASE 7: INTERACTIVE DASHBOARD
        # =====================================================================
        
        st.markdown("---")
        st.markdown("## 📊 Business Intelligence Dashboard")
        
        # KPI Metrics Row
        render_kpi_metrics(df)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Visualization Row: Quality Map & Pipeline Distribution side-by-side
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            render_quality_map(df)
        
        with chart_col2:
            render_pipeline_distribution(df)
        
        st.markdown("---")
        
        # =====================================================================
        # DATA TABLE SECTION
        # =====================================================================
        
        st.markdown("## 📋 Lead Data Table")
        
        active_count = len([r for r in results if r["is_active"]])
        rejected_count = len([r for r in results if not r["is_active"]])
        
        st.success(f"✅ Found {len(results)} approved educators! ({active_count} active, {rejected_count} inactive)")
        
        # Format display DataFrame with PHASE 6 column order
        display_df = pd.DataFrame({
            "Subject Tag": df["subject_tag"],
            "Channel Name": df["name"],
            "Channel Link": df["channel_link"],
            "Type": df["classification"],
            "Subscribers": df["subscribers"].apply(format_number),
            "Total Views": df["total_views"].apply(format_number),
            "Avg Views": df["avg_views"].apply(format_number),
            "Email": df["email"].fillna(""),
            "Instagram": df["instagram"].fillna(""),
            "Telegram": df["telegram"].fillna(""),
            "Website": df["website"].fillna(""),
            "Status": df["status"],
            "Reason": df["reason"].fillna(""),
            "Tier": df["tier"]
        })
        
        # Sort by Tier
        tier_order = {"A": 0, "B": 1, "C": 2, "D": 3}
        display_df["tier_sort"] = display_df["Tier"].map(tier_order)
        display_df = display_df.sort_values("tier_sort").drop("tier_sort", axis=1)
        
        st.dataframe(
            display_df,
            column_config={
                "Channel Link": st.column_config.LinkColumn("Channel Link", display_text="🔗 Visit")
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Tier Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tier A", len(df[df["tier"] == "A"]), help="Best leads")
        with col2:
            st.metric("Tier B", len(df[df["tier"] == "B"]))
        with col3:
            st.metric("Tier C", len(df[df["tier"] == "C"]))
        with col4:
            st.metric("Tier D", len(df[df["tier"] == "D"]), help="Inactive")
        
        # Export with PHASE 6 column order
        st.divider()
        
        # Phase 9: Added Channel ID column for future deduplication accuracy
        export_df = pd.DataFrame({
            "Subject Tag": df["subject_tag"],
            "Channel Name": df["name"],
            "Channel ID": df["channel_id"],  # Phase 9: For accurate deduplication
            "Channel Link": df["channel_link"],
            "Type": df["classification"],
            "Subscribers": df["subscribers"],
            "Total Views": df["total_views"],
            "Avg Views": df["avg_views"],
            "Email": df["email"].fillna(""),
            "Instagram": df["instagram"].fillna(""),
            "Telegram": df["telegram"].fillna(""),
            "Website": df["website"].fillna(""),
            "Status": df["status"],
            "Reason": df["reason"].fillna(""),
            "Tier": df["tier"]
        })
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download All (CSV)",
                data=export_df.to_csv(index=False),
                file_name=f"tutrain_leads_{subject.replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            active_df = export_df[export_df["Status"] == "Approved"]
            st.download_button(
                label="📥 Download Active Only",
                data=active_df.to_csv(index=False),
                file_name=f"tutrain_active_{subject.replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True
            )


if __name__ == "__main__":
    main()
