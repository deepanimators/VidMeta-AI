"""
VidMeta AI
----------
Analyze local videos with AI → Generate upload-ready metadata
for YouTube, Instagram, Facebook, TikTok, LinkedIn.
"""

import streamlit as st
import cv2
import base64
import json
import os
import shutil
import tempfile
import subprocess
import csv
import io as csv_io
from pathlib import Path
from streamlit_cookies_manager import EncryptedCookieManager

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="VidMeta AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

cookies = EncryptedCookieManager(
    prefix="vidmeta/",
    password=os.environ.get(
        "VIDMETA_COOKIE_PASSWORD",
        os.environ.get("COOKIES_PASSWORD", "vidmeta-local-dev-cookie-password"),
    ),
)
if not cookies.ready():
    st.stop()


def _cookie_text(name, default):
    value = cookies.get(name)
    return default if value in (None, "") else str(value)


def _cookie_bool(name, default):
    value = cookies.get(name)
    if value in (None, ""):
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _cookie_int(name, default):
    value = cookies.get(name)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _cookie_choice(name, options, default):
    value = _cookie_text(name, default)
    return value if value in options else default

st.markdown("""
<style>
div[data-testid="stTextArea"] textarea { font-size: 0.88rem; }
div[data-testid="stTextInput"] input   { font-size: 0.88rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎬 VidMeta AI `v1.0`")
    st.caption("Video → AI analysis → Platform metadata")
    st.divider()

    st.markdown("#### LLM Provider")
    provider_options = [
        "Ollama — Local / Free",
        "OpenRouter — Free tier",
        "OpenAI",
        "Anthropic",
        "Google Gemini",
    ]
    provider = st.selectbox(
        "provider",
        provider_options,
        index=provider_options.index(
            _cookie_choice("provider", provider_options, "Ollama — Local / Free")
        ),
        label_visibility="collapsed",
    )

    api_key    = ""
    model      = ""
    ollama_url = _cookie_text("ollama_url", "http://localhost:11434")
    api_base   = ""

    if provider == "Ollama — Local / Free":
        st.info("Requires Ollama running locally with a vision model.")
        ollama_url = st.text_input("Ollama URL", value=ollama_url)
        model      = st.text_input(
            "Model name",
            value=_cookie_text("ollama_model", "gemma4"),
            help="e.g. gemma4 · bakgemma4 · gemma4-llama3 · moondream",
        )
        st.caption("→ [Get Ollama](https://ollama.com/)  |  [Vision models](https://ollama.com/search?c=vision)")

    elif provider == "OpenRouter — Free tier":
        api_key  = st.text_input("OpenRouter API Key", value=_cookie_text("openrouter_api_key", ""), type="password")
        api_base = "https://openrouter.ai/api/v1"
        openrouter_models = [
            "meta-llama/llama-3.2-11b-vision-instruct:free",
            "qwen/qwen2.5-vl-72b-instruct:free",
            "google/gemma-3-27b-it:free",
            "mistralai/mistral-small-3.1-24b-instruct:free",
        ]
        model    = st.selectbox(
            "Model",
            openrouter_models,
            index=openrouter_models.index(_cookie_choice("openrouter_model", openrouter_models, openrouter_models[0])),
        )
        st.caption("→ [Get free key](https://openrouter.ai/)")

    elif provider == "OpenAI":
        api_key  = st.text_input("OpenAI API Key", value=_cookie_text("openai_api_key", ""), type="password")
        api_base = "https://api.openai.com/v1"
        openai_models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
        model    = st.selectbox(
            "Model",
            openai_models,
            index=openai_models.index(_cookie_choice("openai_model", openai_models, openai_models[0])),
        )
        st.caption("→ [Get OpenAI API key](https://platform.openai.com/api-keys)")
        st.caption("OpenAI accounts used here must have API billing enabled.")

    elif provider == "Anthropic":
        api_key = st.text_input("Anthropic API Key", value=_cookie_text("anthropic_api_key", ""), type="password")
        anthropic_models = [
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
        ]
        model   = st.selectbox(
            "Model",
            anthropic_models,
            index=anthropic_models.index(_cookie_choice("anthropic_model", anthropic_models, anthropic_models[0])),
        )
        st.caption("→ [Get key](https://console.anthropic.com/)")

    elif provider == "Google Gemini":
        api_key = st.text_input("Gemini API Key", value=_cookie_text("gemini_api_key", ""), type="password")
        gemini_models = [
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ]
        model   = st.selectbox(
            "Model",
            gemini_models,
            index=gemini_models.index(_cookie_choice("gemini_model", gemini_models, gemini_models[0])),
        )
        st.caption("→ [Get key](https://aistudio.google.com/apikey)")

    st.divider()

    st.markdown("#### Video Processing")
    use_whisper = st.toggle("Transcribe audio (Whisper)", value=_cookie_bool("use_whisper", True))
    whisper_model_size = "base"
    if use_whisper:
        whisper_model_size = st.select_slider(
            "Whisper accuracy",
            options=["tiny", "base", "small", "medium"],
            value=_cookie_choice("whisper_model_size", ["tiny", "base", "small", "medium"], "base"),
        )

    frame_interval = st.slider("Frame every N seconds", 2, 30, _cookie_int("frame_interval", 5))
    max_frames     = st.slider("Max frames to LLM",     3, 12,  _cookie_int("max_frames", 6))

    st.divider()

    st.markdown("#### Brand Context")
    brand_name      = st.text_input("Brand name",       value=_cookie_text("brand_name", "Condenast"))
    brand_niche     = st.text_input("Niche",            value=_cookie_text("brand_niche", "Kids fashion & clothing, India"))
    target_audience = st.text_input("Target audience",  value=_cookie_text("target_audience", "Mothers, parents, India"))
    brand_tone      = st.select_slider(
        "Tone",
        options=["Fun & playful", "Warm & friendly", "Professional", "Exciting"],
        value=_cookie_choice("brand_tone", ["Fun & playful", "Warm & friendly", "Professional", "Exciting"], "Fun & playful"),
    )

    cookies["provider"] = provider
    cookies["ollama_url"] = ollama_url
    cookies["ollama_model"] = model if provider == "Ollama — Local / Free" else _cookie_text("ollama_model", "gemma4")
    cookies["openrouter_api_key"] = api_key if provider == "OpenRouter — Free tier" else _cookie_text("openrouter_api_key", "")
    cookies["openrouter_model"] = model if provider == "OpenRouter — Free tier" else _cookie_text("openrouter_model", "meta-llama/llama-3.2-11b-vision-instruct:free")
    cookies["openai_api_key"] = api_key if provider == "OpenAI" else _cookie_text("openai_api_key", "")
    cookies["openai_model"] = model if provider == "OpenAI" else _cookie_text("openai_model", "gpt-4o")
    cookies["anthropic_api_key"] = api_key if provider == "Anthropic" else _cookie_text("anthropic_api_key", "")
    cookies["anthropic_model"] = model if provider == "Anthropic" else _cookie_text("anthropic_model", "claude-opus-4-6")
    cookies["gemini_api_key"] = api_key if provider == "Google Gemini" else _cookie_text("gemini_api_key", "")
    cookies["gemini_model"] = model if provider == "Google Gemini" else _cookie_text("gemini_model", "gemini-2.0-flash")
    cookies["use_whisper"] = str(use_whisper)
    cookies["whisper_model_size"] = whisper_model_size
    cookies["frame_interval"] = str(frame_interval)
    cookies["max_frames"] = str(max_frames)
    cookies["brand_name"] = brand_name
    cookies["brand_niche"] = brand_niche
    cookies["target_audience"] = target_audience
    cookies["brand_tone"] = brand_tone


# ─────────────────────────────────────────────
# VIDEO HELPERS
# ─────────────────────────────────────────────

def extract_frames(video_path, interval_sec, max_f):
    cap      = cv2.VideoCapture(video_path)
    fps      = cap.get(cv2.CAP_PROP_FPS) or 25
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps

    positions, t = [], 0.0
    while t < duration and len(positions) < max_f:
        positions.append(int(t * fps))
        t += interval_sec

    frames = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        if w > 800:
            frame = cv2.resize(frame, (800, int(h * 800 / w)))
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        frames.append(base64.b64encode(buf).decode())

    cap.release()
    return frames


def transcribe_audio(video_path, wmodel):
    audio_path = video_path + "_audio.wav"
    try:
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-ar", "16000", "-ac", "1", "-y", audio_path],
            capture_output=True, timeout=120,
        )
        if not os.path.exists(audio_path):
            return "[No audio track]"
        try:
            from faster_whisper import WhisperModel
            m = WhisperModel(wmodel, device="cpu", compute_type="int8")
            segs, _ = m.transcribe(audio_path, beam_size=3)
            return " ".join(s.text for s in segs).strip() or "[Silent]"
        except ImportError:
            import whisper
            m = whisper.load_model(wmodel)
            return m.transcribe(audio_path)["text"].strip() or "[Silent]"
    except FileNotFoundError:
        return "[ffmpeg not installed — audio skipped]"
    except Exception as e:
        return f"[Transcription error: {e}]"
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


def _save_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    tmp.close()

    uploaded_file.seek(0)
    with open(tmp_path, "wb") as destination:
        shutil.copyfileobj(uploaded_file, destination, length=8 * 1024 * 1024)

    return tmp_path


# ─────────────────────────────────────────────
# LLM HELPERS
# ─────────────────────────────────────────────

def _call_ollama(frames, prompt, url, mdl):
    import requests
    # Use /api/chat for vision models (Ollama's proper vision endpoint)
    messages = [
        {
            "role": "user",
            "content": prompt,
            "images": frames
        }
    ]
    r = requests.post(
        f"{url}/api/chat",
        json={"model": mdl, "messages": messages, "stream": False},
        timeout=180,
    )
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "")


def _call_openai_compat(frames, prompt, key, mdl, base):
    from openai import OpenAI
    client  = OpenAI(api_key=key, base_url=base)
    content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}} for f in frames[:6]]
    content.append({"type": "text", "text": prompt})
    resp = client.chat.completions.create(
        model=mdl, messages=[{"role": "user", "content": content}], max_tokens=2000
    )
    return resp.choices[0].message.content


def _call_anthropic(frames, prompt, key, mdl):
    import anthropic as ant
    client  = ant.Anthropic(api_key=key)
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": f}} for f in frames[:5]]
    content.append({"type": "text", "text": prompt})
    resp = client.messages.create(model=mdl, max_tokens=2000, messages=[{"role": "user", "content": content}])
    return resp.content[0].text


def _call_gemini(frames, prompt, key, mdl):
    import google.generativeai as genai
    from PIL import Image
    import io as bio
    genai.configure(api_key=key)
    parts = [Image.open(bio.BytesIO(base64.b64decode(f))) for f in frames[:5]]
    parts.append(prompt)
    return genai.GenerativeModel(mdl).generate_content(parts).text


def call_llm(frames, prompt):
    try:
        if   provider == "Ollama — Local / Free":            return _call_ollama(frames, prompt, ollama_url, model)
        elif provider in ("OpenRouter — Free tier","OpenAI"): return _call_openai_compat(frames, prompt, api_key, model, api_base)
        elif provider == "Anthropic":                        return _call_anthropic(frames, prompt, api_key, model)
        elif provider == "Google Gemini":                    return _call_gemini(frames, prompt, api_key, model)
    except Exception as e:
        raise RuntimeError(f"LLM error ({provider}): {e}") from e
    return ""


# ─────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────

ANALYSIS_PROMPT = """You are a content strategist analyzing a video for a social media brand.

Brand: {brand_name}
Niche: {brand_niche}
Target Audience: {target_audience}
Tone: {tone}

Audio Transcript:
{transcript}

Analyze the video frames and describe:
1. Main subject and core message
2. Visual elements — products, colors, setting, people, style
3. Audio/speech summary
4. Content category (tutorial, showcase, lifestyle, BTS, etc.)
5. Unique selling points visible
6. Emotional appeal and audience fit
7. Suggested social media angles

Be specific. This drives all metadata generation."""

METADATA_PROMPT = """Generate optimized social media metadata from this video analysis.

VIDEO ANALYSIS:
{analysis}

Brand: {brand_name} | Niche: {brand_niche} | Audience: {target_audience} | Tone: {tone}

Return ONLY valid JSON — no markdown, no backticks, no extra text:
{{
  "video_summary": "2-3 sentence summary",
  "content_category": "e.g. Product Showcase",
  "youtube": {{
    "title": "SEO title with keyword, under 60 chars",
    "description": "300-400 word description. Hook in first 2 lines. Timestamps if logical. Strong CTA at end.",
    "hashtags": ["tag1","tag2","tag3","tag4","tag5","tag6","tag7","tag8","tag9","tag10"],
    "keywords": ["kw1","kw2","kw3","kw4","kw5","kw6","kw7","kw8","kw9","kw10","kw11","kw12","kw13","kw14","kw15"],
    "cta": "Subscribe CTA text",
    "posting_tip": "Best posting time for YouTube India"
  }},
  "instagram": {{
    "title": "Hook line for caption opening",
    "description": "150-200 word caption. Storytelling. Line breaks. End with question or CTA.",
    "hashtags": ["#h1","#h2","#h3","#h4","#h5","#h6","#h7","#h8","#h9","#h10","#h11","#h12","#h13","#h14","#h15","#h16","#h17","#h18","#h19","#h20","#h21","#h22","#h23","#h24","#h25","#h26","#h27","#h28","#h29","#h30"],
    "keywords": ["kw1","kw2","kw3","kw4","kw5","kw6","kw7","kw8","kw9","kw10"],
    "cta": "Follow + link in bio CTA",
    "posting_tip": "Best posting time for Instagram India"
  }},
  "facebook": {{
    "title": "Facebook post heading",
    "description": "100-150 word conversational post.",
    "hashtags": ["#h1","#h2","#h3","#h4","#h5"],
    "keywords": ["kw1","kw2","kw3","kw4","kw5","kw6","kw7","kw8"],
    "cta": "Like/Share/Comment CTA",
    "posting_tip": "Best posting time for Facebook India"
  }},
  "tiktok": {{
    "title": "Punchy TikTok hook under 100 chars",
    "description": "Short fun caption under 100 words.",
    "hashtags": ["#h1","#h2","#h3","#h4","#h5","#h6","#h7","#h8","#h9","#h10","#h11","#h12","#h13","#h14","#h15"],
    "keywords": ["kw1","kw2","kw3","kw4","kw5","kw6","kw7","kw8"],
    "cta": "Follow for more CTA",
    "posting_tip": "Best posting time for TikTok India"
  }},
  "linkedin": {{
    "title": "Professional LinkedIn headline",
    "description": "200-250 word post. Brand story angle. Entrepreneur perspective.",
    "hashtags": ["#h1","#h2","#h3","#h4","#h5","#h6","#h7"],
    "keywords": ["kw1","kw2","kw3","kw4","kw5","kw6","kw7","kw8"],
    "cta": "Connect/Follow CTA",
    "posting_tip": "Best posting time for LinkedIn"
  }}
}}"""


# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────

st.markdown("# 🎬 VidMeta AI")
st.markdown("Drop a video → AI watches it → Get upload-ready metadata for every platform")

c1, c2, c3, c4, c5, _ = st.columns([1, 1, 1, 1, 1, 3])
c1.markdown("🔴 **YouTube**")
c2.markdown("🔵 **Facebook**")
c3.markdown("🟣 **Instagram**")
c4.markdown("⚫ **TikTok**")
c5.markdown("🔷 **LinkedIn**")

st.divider()

mode_single, mode_batch = st.tabs(["Single Video", "Batch — Folder"])

# ── Single video tab
video_file_path = None
uploaded_file   = None
analyze_btn     = False
batch_btn       = False
batch_folder    = None

with mode_single:
    st.markdown("#### Video Input")
    input_mode = st.radio(
        "Input mode",
        ["Upload file", "Local file path"],
        horizontal=True,
        label_visibility="collapsed",
        index=["Upload file", "Local file path"].index(_cookie_choice("input_mode", ["Upload file", "Local file path"], "Upload file")),
    )
    cookies["input_mode"] = input_mode

    if input_mode == "Upload file":
        uploaded_file = st.file_uploader(
            "Drag & drop your video here",
            type=["mp4", "mov", "avi", "mkv", "webm", "m4v"],
        )
        if uploaded_file:
            uploaded_size_mb = getattr(uploaded_file, "size", 0) / 1e6
            st.success(f"Uploaded: **{uploaded_file.name}** — {uploaded_size_mb:.1f} MB")
            if uploaded_size_mb <= 100:
                st.video(uploaded_file)
            else:
                st.info("Preview disabled for large uploads to keep memory usage lower.")
    else:
        path_input = st.text_input(
            "Full path to video",
            value=_cookie_text("single_video_path", ""),
            placeholder="/Users/Condenast/Videos/Condenast_video.mp4",
        )
        cookies["single_video_path"] = path_input
        if path_input:
            if os.path.exists(path_input):
                size_mb = os.path.getsize(path_input) / 1e6
                st.success(f"Found: **{Path(path_input).name}** — {size_mb:.1f} MB")
                st.video(path_input)
                video_file_path = path_input
            else:
                st.error("File not found at that path.")

    st.divider()
    col_btn, col_tip = st.columns([1, 4])
    with col_btn:
        analyze_btn = st.button("Analyze & Generate", type="primary", use_container_width=True)
    with col_tip:
        st.caption("Extracts frames → Transcribes audio → LLM analysis → Metadata for all 5 platforms")


# ── Batch tab
with mode_batch:
    st.markdown("#### Folder Path")
    batch_folder_input = st.text_input(
        "Folder path",
        value=_cookie_text("batch_folder_path", ""),
        placeholder="/Users/Condenast/Videos/Condenast/",
        label_visibility="collapsed",
    )
    cookies["batch_folder_path"] = batch_folder_input

    VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

    if batch_folder_input:
        bp = Path(batch_folder_input)
        if bp.exists() and bp.is_dir():
            found = sorted([f for f in bp.iterdir() if f.suffix.lower() in VIDEO_EXTS])
            if found:
                st.success(f"Found **{len(found)} video(s)** in folder")
                with st.expander("Video list"):
                    for i, f in enumerate(found, 1):
                        size_mb = f.stat().st_size / 1e6
                        st.caption(f"{i}. {f.name}  —  {size_mb:.1f} MB")
                batch_folder = bp
            else:
                st.warning("No video files found in that folder.")
        elif batch_folder_input:
            st.error("Folder not found.")

    st.divider()
    col_bb, col_bt = st.columns([1, 4])
    with col_bb:
        batch_btn = st.button("Process All Videos", type="primary", use_container_width=True)
    with col_bt:
        st.caption("Each video is analyzed in sequence. Results export as a single combined file.")


cookies.save()


# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────

if analyze_btn:
    if not uploaded_file and not video_file_path:
        st.error("Please provide a video file.")
        st.stop()
    if provider != "Ollama — Local / Free" and not api_key:
        st.error("Please enter your API key in the sidebar.")
        st.stop()
    if not model:
        st.error("Please enter or select a model name.")
        st.stop()

    if uploaded_file:
        actual_path = _save_uploaded_file(uploaded_file)
    else:
        actual_path = video_file_path

    progress = st.progress(0, text="Starting…")

    with st.status("Processing your video…", expanded=True) as status_box:

        st.write("Extracting frames…")
        try:
            frames = extract_frames(actual_path, frame_interval, max_frames)
        except Exception as e:
            st.error(f"Frame extraction failed: {e}")
            st.stop()
        progress.progress(20, text=f"{len(frames)} frames extracted")
        st.write(f"{len(frames)} frames ready")

        transcript = ""
        if use_whisper:
            st.write("Transcribing audio…")
            transcript = transcribe_audio(actual_path, whisper_model_size)
            progress.progress(45, text="Audio transcribed")
            preview = (transcript[:100] + "…") if len(transcript) > 100 else transcript
            st.write(f"Transcript: _{preview}_")
        else:
            progress.progress(45, text="Audio skipped")

        st.write(f"Analyzing with {provider}…")
        analysis_prompt = ANALYSIS_PROMPT.format(
            brand_name=brand_name, brand_niche=brand_niche,
            target_audience=target_audience, tone=brand_tone,
            transcript=transcript or "No transcript available",
        )
        try:
            analysis = call_llm(frames, analysis_prompt)
        except RuntimeError as e:
            st.error(str(e)); st.stop()
        progress.progress(70, text="Video analyzed")
        st.write("Video analysis complete")

        st.write("Generating metadata for all platforms…")
        metadata_prompt = METADATA_PROMPT.format(
            analysis=analysis, brand_name=brand_name, brand_niche=brand_niche,
            target_audience=target_audience, tone=brand_tone,
        )
        try:
            meta_raw = call_llm([], metadata_prompt)
        except RuntimeError as e:
            st.error(str(e)); st.stop()

        try:
            clean    = meta_raw.replace("```json", "").replace("```", "").strip()
            metadata = json.loads(clean)
        except json.JSONDecodeError:
            st.warning("LLM returned non-JSON — showing raw.")
            metadata = {"raw": meta_raw}

        progress.progress(100, text="Done!")
        st.write("All metadata generated!")
        status_box.update(label="Analysis complete!", state="complete")

    st.session_state["metadata"]   = metadata
    st.session_state["analysis"]   = analysis
    st.session_state["transcript"] = transcript

    if uploaded_file and os.path.exists(actual_path):
        os.remove(actual_path)


# ─────────────────────────────────────────────
# BATCH PIPELINE
# ─────────────────────────────────────────────

if batch_btn:
    if not batch_folder:
        st.error("Please enter a valid folder path.")
        st.stop()
    if provider != "Ollama — Local / Free" and not api_key:
        st.error("Please enter your API key in the sidebar.")
        st.stop()
    if not model:
        st.error("Please enter or select a model name.")
        st.stop()

    VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    video_files = sorted([f for f in batch_folder.iterdir() if f.suffix.lower() in VIDEO_EXTS])

    if not video_files:
        st.error("No video files found in that folder.")
        st.stop()

    st.divider()
    st.markdown(f"### Processing {len(video_files)} videos")

    batch_results = []
    overall = st.progress(0, text="Starting batch…")

    for idx, vfile in enumerate(video_files):
        st.markdown(f"---\n**[{idx+1}/{len(video_files)}] {vfile.name}**")

        with st.status(f"Processing {vfile.name}…", expanded=False) as vstatus:

            # Frames
            st.write("Extracting frames…")
            try:
                frames = extract_frames(str(vfile), frame_interval, max_frames)
            except Exception as e:
                st.error(f"Frame extraction failed: {e}")
                batch_results.append({"file": vfile.name, "error": str(e)})
                continue
            st.write(f"{len(frames)} frames extracted")

            # Audio
            transcript = ""
            if use_whisper:
                st.write("Transcribing audio…")
                transcript = transcribe_audio(str(vfile), whisper_model_size)
                st.write("Audio transcribed")

            # Analysis
            st.write("Analyzing video…")
            try:
                analysis = call_llm(frames, ANALYSIS_PROMPT.format(
                    brand_name=brand_name, brand_niche=brand_niche,
                    target_audience=target_audience, tone=brand_tone,
                    transcript=transcript or "No transcript available",
                ))
            except RuntimeError as e:
                st.error(str(e))
                batch_results.append({"file": vfile.name, "error": str(e)})
                continue

            # Metadata
            st.write("Generating metadata…")
            try:
                meta_raw = call_llm([], METADATA_PROMPT.format(
                    analysis=analysis, brand_name=brand_name, brand_niche=brand_niche,
                    target_audience=target_audience, tone=brand_tone,
                ))
                clean    = meta_raw.replace("```json", "").replace("```", "").strip()
                metadata = json.loads(clean)
            except Exception as e:
                st.warning(f"Metadata parse error: {e}")
                metadata = {"raw": meta_raw if 'meta_raw' in dir() else str(e)}

            batch_results.append({"file": vfile.name, "metadata": metadata})
            vstatus.update(label=f"Done — {vfile.name}", state="complete")

        overall.progress(
            int((idx + 1) / len(video_files) * 100),
            text=f"Completed {idx+1} of {len(video_files)}",
        )

    st.session_state["batch_results"] = batch_results
    st.success(f"Batch complete — {len([r for r in batch_results if 'metadata' in r])} of {len(video_files)} videos processed successfully.")


# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────

if "metadata" in st.session_state:
    md   = st.session_state["metadata"]
    anlz = st.session_state.get("analysis", "")
    trx  = st.session_state.get("transcript", "")

    st.divider()

    if md.get("video_summary"):
        st.info(md['video_summary'])
    if md.get("content_category"):
        st.caption(f"Category: `{md['content_category']}`")

    with st.expander("Raw video analysis"):
        st.write(anlz)
    if trx:
        with st.expander("Audio transcript"):
            st.write(trx)

    st.markdown("### Platform Metadata")

    PLATFORMS = {
        "youtube":   "YouTube",
        "facebook":  "Facebook",
        "instagram": "Instagram",
        "tiktok":    "TikTok",
        "linkedin":  "LinkedIn",
    }

    tabs = st.tabs(list(PLATFORMS.values()))

    for tab, (pkey, plabel) in zip(tabs, PLATFORMS.items()):
        with tab:
            pdata = md.get(pkey)
            if not pdata or not isinstance(pdata, dict):
                st.warning("No data for this platform.")
                continue

            col_l, col_r = st.columns([3, 2])

            with col_l:
                title_val = pdata.get("title", "")
                st.text_input(f"Title — {len(title_val)} chars", value=title_val, key=f"t_{pkey}")
                desc_val = pdata.get("description", "")
                st.text_area(f"Description — {len(desc_val)} chars", value=desc_val, height=220, key=f"d_{pkey}")
                if pdata.get("cta"):
                    st.text_input("Call to Action", value=pdata["cta"], key=f"cta_{pkey}")

            with col_r:
                tags = pdata.get("hashtags", [])
                tags_str = "\n".join(tags) if isinstance(tags, list) else tags
                st.text_area(
                    f"Hashtags — {len(tags) if isinstance(tags, list) else '?'}",
                    value=tags_str, height=160, key=f"h_{pkey}",
                )
                kws = pdata.get("keywords", [])
                st.text_area(
                    "SEO Keywords",
                    value=", ".join(kws) if isinstance(kws, list) else kws,
                    height=90, key=f"k_{pkey}",
                )
                if pdata.get("posting_tip"):
                    st.success(pdata['posting_tip'])

            st.caption("Download fields:")
            dc1, dc2, dc3, dc4 = st.columns(4)
            with dc1:
                st.download_button("Title",       data=pdata.get("title",""),                                       file_name=f"{pkey}_title.txt",    key=f"dl_t_{pkey}")
            with dc2:
                st.download_button("Description", data=pdata.get("description",""),                                 file_name=f"{pkey}_desc.txt",     key=f"dl_d_{pkey}")
            with dc3:
                st.download_button("Hashtags",    data="\n".join(tags) if isinstance(tags,list) else tags,          file_name=f"{pkey}_hashtags.txt", key=f"dl_h_{pkey}")
            with dc4:
                st.download_button("Keywords",    data=", ".join(kws) if isinstance(kws,list) else kws,             file_name=f"{pkey}_keywords.txt", key=f"dl_k_{pkey}")

    st.divider()
    st.markdown("#### Export All Platforms")
    e1, e2, e3 = st.columns(3)

    with e1:
        st.download_button(
            "Download JSON", use_container_width=True,
            data=json.dumps(md, indent=2, ensure_ascii=False),
            file_name="vidmeta_output.json", mime="application/json",
        )
    with e2:
        rows = []
        for pkey, plabel in PLATFORMS.items():
            pdata = md.get(pkey, {})
            if isinstance(pdata, dict):
                tags = pdata.get("hashtags", [])
                kws  = pdata.get("keywords", [])
                rows.append({
                    "Platform":    plabel,
                    "Title":       pdata.get("title", ""),
                    "Description": pdata.get("description", ""),
                    "Hashtags":    " ".join(tags) if isinstance(tags, list) else tags,
                    "Keywords":    ", ".join(kws)  if isinstance(kws,  list) else kws,
                    "CTA":         pdata.get("cta", ""),
                    "Posting Tip": pdata.get("posting_tip", ""),
                })
        buf = csv_io.StringIO()
        if rows:
            w = csv.DictWriter(buf, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)
        st.download_button(
            "Download CSV", use_container_width=True,
            data=buf.getvalue(), file_name="vidmeta_output.csv", mime="text/csv",
        )
    with e3:
        txt = []
        for pkey, plabel in PLATFORMS.items():
            pdata = md.get(pkey, {})
            if isinstance(pdata, dict):
                tags = pdata.get("hashtags", [])
                txt.append(
                    f"{'='*40}\n{plabel}\n{'='*40}\n"
                    f"TITLE:\n{pdata.get('title','')}\n\n"
                    f"DESCRIPTION:\n{pdata.get('description','')}\n\n"
                    f"HASHTAGS:\n{' '.join(tags) if isinstance(tags,list) else tags}\n\n"
                    f"KEYWORDS:\n{', '.join(pdata.get('keywords',[]))}\n\n"
                    f"CTA: {pdata.get('cta','')}\n"
                    f"POSTING TIP: {pdata.get('posting_tip','')}\n\n"
                )
        st.download_button(
            "Download TXT", use_container_width=True,
            data="\n".join(txt), file_name="vidmeta_output.txt", mime="text/plain",
        )


# ─────────────────────────────────────────────
# BATCH RESULTS
# ─────────────────────────────────────────────

if "batch_results" in st.session_state:
    results = st.session_state["batch_results"]
    ok      = [r for r in results if "metadata" in r]
    failed  = [r for r in results if "error"    in r]

    st.divider()
    st.markdown("### Batch Results")

    m1, m2, m3 = st.columns(3)
    m1.metric("Total videos",   len(results))
    m2.metric("Processed",      len(ok))
    m3.metric("Failed",         len(failed))

    if failed:
        with st.expander(f"Failed ({len(failed)})"):
            for r in failed:
                st.error(f"{r['file']} — {r['error']}")

    # Per-video expandable results
    PLATFORMS = {
        "youtube": "YouTube", "facebook": "Facebook", "instagram": "Instagram",
        "tiktok": "TikTok",   "linkedin": "LinkedIn",
    }

    for r in ok:
        with st.expander(f"{r['file']}"):
            md = r["metadata"]
            if md.get("video_summary"):
                st.info(md["video_summary"])
            tabs = st.tabs(list(PLATFORMS.values()))
            for tab, (pkey, plabel) in zip(tabs, PLATFORMS.items()):
                with tab:
                    pdata = md.get(pkey, {})
                    if not isinstance(pdata, dict):
                        st.caption("No data"); continue
                    st.text_input("Title",       value=pdata.get("title",""),       key=f"b_t_{r['file']}_{pkey}")
                    st.text_area("Description",  value=pdata.get("description",""), key=f"b_d_{r['file']}_{pkey}", height=140)
                    tags = pdata.get("hashtags", [])
                    st.text_area("Hashtags",     value="\n".join(tags) if isinstance(tags,list) else tags, key=f"b_h_{r['file']}_{pkey}", height=80)

    st.divider()
    st.markdown("#### Export All Batch Results")
    be1, be2 = st.columns(2)

    with be1:
        st.download_button(
            "Download JSON (all videos)",
            data=json.dumps(results, indent=2, ensure_ascii=False),
            file_name="vidmeta_batch.json",
            mime="application/json",
            use_container_width=True,
        )

    with be2:
        # Flat CSV — one row per platform per video
        csv_rows = []
        for r in ok:
            for pkey, plabel in PLATFORMS.items():
                pdata = r["metadata"].get(pkey, {})
                if isinstance(pdata, dict):
                    tags = pdata.get("hashtags", [])
                    kws  = pdata.get("keywords",  [])
                    csv_rows.append({
                        "File":        r["file"],
                        "Platform":    plabel,
                        "Title":       pdata.get("title", ""),
                        "Description": pdata.get("description", ""),
                        "Hashtags":    " ".join(tags) if isinstance(tags, list) else tags,
                        "Keywords":    ", ".join(kws)  if isinstance(kws,  list) else kws,
                        "CTA":         pdata.get("cta", ""),
                        "Posting Tip": pdata.get("posting_tip", ""),
                    })
        buf = csv_io.StringIO()
        if csv_rows:
            w = csv.DictWriter(buf, fieldnames=csv_rows[0].keys())
            w.writeheader(); w.writerows(csv_rows)
        st.download_button(
            "Download CSV (all videos)",
            data=buf.getvalue(),
            file_name="vidmeta_batch.csv",
            mime="text/csv",
            use_container_width=True,
        )
