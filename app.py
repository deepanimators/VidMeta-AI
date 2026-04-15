"""
Video Metadata Hub
------------------
Analyze local videos with AI → Generate platform-ready metadata
for YouTube, Instagram, Facebook, TikTok, LinkedIn.
"""

import streamlit as st
import cv2
import base64
import json
import os
import tempfile
import subprocess
import csv
import io as csv_io
from pathlib import Path

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Video Metadata Hub",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
section[data-testid="stSidebar"] { min-width: 320px; max-width: 360px; }
.platform-chip {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600; margin: 2px;
}
div[data-testid="stTextArea"] textarea { font-size: 0.88rem; }
.st-emotion-cache-1kyxreq { gap: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR — LLM + VIDEO SETTINGS
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/Video_Metadata_Hub-v1.0-blue?style=for-the-badge", use_container_width=True)

    st.header("🤖 LLM Provider")
    provider = st.selectbox(
        "Select provider",
        [
            "🏠 Ollama — Local / Free",
            "🌐 OpenRouter — Free tier",
            "🔷 OpenAI",
            "🟠 Anthropic",
            "💎 Google Gemini",
        ],
        label_visibility="collapsed",
    )

    api_key     = ""
    model       = ""
    ollama_url  = "http://localhost:11434"
    api_base    = ""

    if "Ollama" in provider:
        st.info("💡 Requires Ollama running locally with a vision model")
        ollama_url = st.text_input("Ollama base URL", value="http://localhost:11434")
        model      = st.text_input("Vision model name", value="llava",
                                   help="e.g. llava, bakllava, llava-llama3, moondream")
        st.markdown("[📥 Get Ollama](https://ollama.com/) · [Vision models](https://ollama.com/search?c=vision)")

    elif "OpenRouter" in provider:
        api_key  = st.text_input("OpenRouter API Key", type="password")
        api_base = "https://openrouter.ai/api/v1"
        model    = st.selectbox("Model", [
            "meta-llama/llama-3.2-11b-vision-instruct:free",
            "qwen/qwen2.5-vl-72b-instruct:free",
            "google/gemma-3-27b-it:free",
            "mistralai/mistral-small-3.1-24b-instruct:free",
        ])
        st.markdown("[🔑 Get free key](https://openrouter.ai/)")

    elif "OpenAI" in provider:
        api_key  = st.text_input("OpenAI API Key", type="password")
        api_base = "https://api.openai.com/v1"
        model    = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"])
        st.markdown("[🔑 Get key](https://platform.openai.com/api-keys)")

    elif "Anthropic" in provider:
        api_key = st.text_input("Anthropic API Key", type="password")
        model   = st.selectbox("Model", [
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
        ])
        st.markdown("[🔑 Get key](https://console.anthropic.com/)")

    elif "Gemini" in provider:
        api_key = st.text_input("Gemini API Key", type="password")
        model   = st.selectbox("Model", [
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ])
        st.markdown("[🔑 Get key](https://aistudio.google.com/apikey)")

    st.divider()
    st.header("🎬 Video Processing")

    use_whisper = st.toggle("Enable audio transcription (Whisper)", value=True)
    if use_whisper:
        whisper_model_size = st.selectbox(
            "Whisper model", ["tiny", "base", "small", "medium"],
            index=1, help="Bigger = more accurate, slower"
        )
    else:
        whisper_model_size = "base"

    frame_interval = st.slider("Extract 1 frame every (seconds)", 2, 30, 5)
    max_frames     = st.slider("Max frames to send to LLM", 3, 12, 6)

    st.divider()
    st.header("🏷️ Brand Context")
    brand_name      = st.text_input("Brand name",       value="Starkids India")
    brand_niche     = st.text_input("Niche",            value="Kids fashion & clothing, India")
    target_audience = st.text_input("Target audience",  value="Mothers, parents, India")
    brand_tone      = st.selectbox("Brand tone", ["Fun & playful", "Warm & friendly", "Professional", "Exciting"])


# ─────────────────────────────────────────────
# VIDEO PROCESSING FUNCTIONS
# ─────────────────────────────────────────────

def extract_frames(video_path: str, interval_sec: int, max_f: int) -> list[str]:
    """Extract frames from a video and return as base64 JPEG list."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps

    positions = []
    t = 0.0
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
            scale = 800 / w
            frame = cv2.resize(frame, (800, int(h * scale)))
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        frames.append(base64.b64encode(buf).decode())

    cap.release()
    return frames


def transcribe_audio(video_path: str, wmodel: str) -> str:
    """Transcribe audio track using Whisper (faster-whisper preferred)."""
    audio_path = video_path + "_audio.wav"
    try:
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-ar", "16000", "-ac", "1", "-y", audio_path],
            capture_output=True, timeout=120,
        )
        if not os.path.exists(audio_path):
            return "[No audio track found]"

        try:
            from faster_whisper import WhisperModel
            m = WhisperModel(wmodel, device="cpu", compute_type="int8")
            segs, _ = m.transcribe(audio_path, beam_size=3)
            return " ".join(s.text for s in segs).strip() or "[Silent audio]"
        except ImportError:
            pass

        import whisper
        m = whisper.load_model(wmodel)
        result = m.transcribe(audio_path)
        return result["text"].strip() or "[Silent audio]"

    except FileNotFoundError:
        return "[ffmpeg not installed — audio skipped]"
    except Exception as e:
        return f"[Transcription error: {e}]"
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


# ─────────────────────────────────────────────
# LLM CALL FUNCTIONS
# ─────────────────────────────────────────────

def _call_ollama(frames: list[str], prompt: str, url: str, mdl: str) -> str:
    import requests
    r = requests.post(
        f"{url}/api/generate",
        json={"model": mdl, "prompt": prompt, "images": frames, "stream": False},
        timeout=180,
    )
    r.raise_for_status()
    return r.json().get("response", "")


def _call_openai_compat(frames: list[str], prompt: str, key: str, mdl: str, base: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=base)
    content: list = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}}
        for f in frames[:6]
    ]
    content.append({"type": "text", "text": prompt})
    resp = client.chat.completions.create(
        model=mdl, messages=[{"role": "user", "content": content}], max_tokens=2000
    )
    return resp.choices[0].message.content


def _call_anthropic(frames: list[str], prompt: str, key: str, mdl: str) -> str:
    import anthropic as ant
    client = ant.Anthropic(api_key=key)
    content: list = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": f}}
        for f in frames[:5]
    ]
    content.append({"type": "text", "text": prompt})
    resp = client.messages.create(
        model=mdl, max_tokens=2000, messages=[{"role": "user", "content": content}]
    )
    return resp.content[0].text


def _call_gemini(frames: list[str], prompt: str, key: str, mdl: str) -> str:
    import google.generativeai as genai
    from PIL import Image
    import io as bio
    genai.configure(api_key=key)
    gm = genai.GenerativeModel(mdl)
    parts = []
    for f in frames[:5]:
        img = Image.open(bio.BytesIO(base64.b64decode(f)))
        parts.append(img)
    parts.append(prompt)
    return gm.generate_content(parts).text


def call_llm(frames: list[str], prompt: str) -> str:
    """Route to the correct LLM based on sidebar settings."""
    prov = provider
    try:
        if "Ollama" in prov:
            return _call_ollama(frames, prompt, ollama_url, model)
        elif "OpenRouter" in prov or "OpenAI" in prov:
            return _call_openai_compat(frames, prompt, api_key, model, api_base)
        elif "Anthropic" in prov:
            return _call_anthropic(frames, prompt, api_key, model)
        elif "Gemini" in prov:
            return _call_gemini(frames, prompt, api_key, model)
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}") from e
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

Analyze the video frames provided and give a detailed report covering:
1. Main subject and core message of the video
2. Visual elements (products shown, colors, setting, people, style, mood)
3. Audio/speech summary (what was said, key points)
4. Content category (tutorial, showcase, lifestyle, behind-the-scenes, etc.)
5. Unique selling points or highlights visible
6. Audience fit and emotional appeal
7. Suggested content angles for social media

Be specific and thorough — this analysis will be used to generate all social metadata."""


METADATA_PROMPT = """Based on the video analysis below, generate optimized social media metadata.

VIDEO ANALYSIS:
{analysis}

Brand: {brand_name} | Niche: {brand_niche} | Audience: {target_audience} | Tone: {tone}

Return ONLY valid JSON. No markdown fences. No extra text. Follow this exact structure:
{{
  "video_summary": "2-3 sentence summary of the video",
  "content_category": "e.g. Product Showcase",
  "youtube": {{
    "title": "SEO title with keyword, under 60 chars visible",
    "description": "Full 300-400 word description. First 2 lines must hook. Include timestamps if logical. End with strong CTA. Add relevant links section placeholder.",
    "hashtags": ["tag1","tag2","tag3","tag4","tag5","tag6","tag7","tag8","tag9","tag10"],
    "keywords": ["kw1","kw2","kw3","kw4","kw5","kw6","kw7","kw8","kw9","kw10","kw11","kw12","kw13","kw14","kw15"],
    "cta": "Subscribe CTA text",
    "posting_tip": "Best time + frequency for YouTube India"
  }},
  "instagram": {{
    "title": "First hook line for caption",
    "description": "150-200 word caption. Storytelling tone. Engaging. Use line breaks. End with question or CTA.",
    "hashtags": ["#h1","#h2","#h3","#h4","#h5","#h6","#h7","#h8","#h9","#h10","#h11","#h12","#h13","#h14","#h15","#h16","#h17","#h18","#h19","#h20","#h21","#h22","#h23","#h24","#h25","#h26","#h27","#h28","#h29","#h30"],
    "keywords": ["kw1","kw2","kw3","kw4","kw5","kw6","kw7","kw8","kw9","kw10"],
    "cta": "Follow CTA + link in bio text",
    "posting_tip": "Best time + days for Instagram India"
  }},
  "facebook": {{
    "title": "Facebook post heading",
    "description": "100-150 word post. Conversational. Tag suggestions where applicable.",
    "hashtags": ["#h1","#h2","#h3","#h4","#h5"],
    "keywords": ["kw1","kw2","kw3","kw4","kw5","kw6","kw7","kw8"],
    "cta": "Like/Share CTA",
    "posting_tip": "Best time for Facebook India"
  }},
  "tiktok": {{
    "title": "Punchy hook caption under 100 chars for TikTok",
    "description": "Short caption under 150 words, fun, trending, relatable",
    "hashtags": ["#h1","#h2","#h3","#h4","#h5","#h6","#h7","#h8","#h9","#h10","#h11","#h12","#h13","#h14","#h15"],
    "keywords": ["kw1","kw2","kw3","kw4","kw5","kw6","kw7","kw8"],
    "cta": "Follow for more CTA",
    "posting_tip": "Best time for TikTok India"
  }},
  "linkedin": {{
    "title": "Professional headline for LinkedIn",
    "description": "200-250 word post. Brand story angle. Entrepreneur perspective. Professional yet warm.",
    "hashtags": ["#h1","#h2","#h3","#h4","#h5","#h6","#h7"],
    "keywords": ["kw1","kw2","kw3","kw4","kw5","kw6","kw7","kw8"],
    "cta": "Connect/Follow CTA",
    "posting_tip": "Best time for LinkedIn"
  }}
}}"""


# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────

st.title("🎬 Video Metadata Hub")
st.caption("Drop a video → AI watches it → Get upload-ready metadata for every platform")

# ── Platform badges
cols = st.columns(5)
badges = [
    ("YouTube",   "#FF0000"),
    ("Facebook",  "#1877F2"),
    ("Instagram", "#E1306C"),
    ("TikTok",    "#010101"),
    ("LinkedIn",  "#0A66C2"),
]
for col, (name, color) in zip(cols, badges):
    col.markdown(
        f'<div class="platform-chip" style="background:{color};color:white">{name}</div>',
        unsafe_allow_html=True,
    )

st.divider()

# ── Video input
tab_upload, tab_path = st.tabs(["📂 Upload file", "📁 Local file path"])

video_file_path = None
uploaded_file   = None

with tab_upload:
    uploaded_file = st.file_uploader(
        "Drag & drop your video",
        type=["mp4", "mov", "avi", "mkv", "webm", "m4v"],
        help="Upload directly from your machine",
    )
    if uploaded_file:
        st.video(uploaded_file)

with tab_path:
    path_input = st.text_input(
        "Full path to video file",
        placeholder="/Users/deepak/Videos/starkids_summer_collection.mp4",
        help="Paste the complete file path. Best for large files.",
    )
    if path_input and os.path.exists(path_input):
        st.success(f"✅ File found: {Path(path_input).name}  ({os.path.getsize(path_input) / 1e6:.1f} MB)")
        st.video(path_input)
        video_file_path = path_input
    elif path_input:
        st.error("❌ File not found at that path")

# ── Analyze button
st.divider()
col_btn, col_info = st.columns([1, 3])
with col_btn:
    analyze_btn = st.button("🔍 Analyze & Generate", type="primary", use_container_width=True)
with col_info:
    st.caption("Steps: Extract frames → Transcribe audio → LLM analysis → Generate metadata for all 5 platforms")

# ─────────────────────────────────────────────
# ANALYSIS PIPELINE
# ─────────────────────────────────────────────

if analyze_btn:
    # Validate
    if not uploaded_file and not video_file_path:
        st.error("Please upload a video or provide a file path first.")
        st.stop()

    # Validate provider setup
    if "Ollama" not in provider and not api_key:
        st.error("Please enter an API key in the sidebar.")
        st.stop()
    if not model:
        st.error("Please select or enter a model name.")
        st.stop()

    # Save uploaded file to temp if needed
    if uploaded_file:
        suffix = Path(uploaded_file.name).suffix
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded_file.read())
        tmp.close()
        actual_path = tmp.name
    else:
        actual_path = video_file_path

    # ── Pipeline with status box
    progress = st.progress(0, text="Starting…")
    with st.status("🎬 Processing video…", expanded=True) as status_box:

        # STEP 1 — Extract frames
        st.write("📸 Extracting video frames…")
        try:
            frames = extract_frames(actual_path, frame_interval, max_frames)
        except Exception as e:
            st.error(f"Frame extraction failed: {e}")
            st.stop()
        progress.progress(20, text=f"Extracted {len(frames)} frames")
        st.write(f"✅ {len(frames)} frames extracted")

        # STEP 2 — Transcribe audio
        transcript = ""
        if use_whisper:
            st.write("🎙️ Transcribing audio with Whisper…")
            transcript = transcribe_audio(actual_path, whisper_model_size)
            progress.progress(45, text="Audio transcribed")
            preview = transcript[:120] + "…" if len(transcript) > 120 else transcript
            st.write(f"✅ Transcript: _{preview}_")
        else:
            st.write("⏭️ Audio transcription skipped")
            progress.progress(45)

        # STEP 3 — LLM video analysis
        st.write(f"🤖 Sending to {provider.split('—')[0].strip()}…")
        analysis_prompt = ANALYSIS_PROMPT.format(
            brand_name=brand_name,
            brand_niche=brand_niche,
            target_audience=target_audience,
            tone=brand_tone,
            transcript=transcript if transcript else "No transcript available",
        )
        try:
            analysis = call_llm(frames, analysis_prompt)
        except RuntimeError as e:
            st.error(str(e))
            st.stop()
        progress.progress(70, text="Video analyzed")
        st.write("✅ Video analysis complete")

        # STEP 4 — Generate metadata
        st.write("📋 Generating metadata for all platforms…")
        metadata_prompt = METADATA_PROMPT.format(
            analysis=analysis,
            brand_name=brand_name,
            brand_niche=brand_niche,
            target_audience=target_audience,
            tone=brand_tone,
        )
        try:
            meta_raw = call_llm([], metadata_prompt)  # text-only call
        except RuntimeError as e:
            st.error(str(e))
            st.stop()

        # Parse JSON
        try:
            clean = meta_raw.replace("```json", "").replace("```", "").strip()
            metadata = json.loads(clean)
        except json.JSONDecodeError:
            st.warning("⚠️ LLM returned non-JSON. Showing raw output.")
            metadata = {"raw": meta_raw}

        progress.progress(100, text="Done!")
        st.write("✅ Metadata generated for all 5 platforms!")
        status_box.update(label="✅ Analysis complete!", state="complete")

    # Store results
    st.session_state["metadata"]   = metadata
    st.session_state["analysis"]   = analysis
    st.session_state["transcript"] = transcript

    # Cleanup temp file
    if uploaded_file and os.path.exists(actual_path):
        os.remove(actual_path)


# ─────────────────────────────────────────────
# RESULTS DISPLAY
# ─────────────────────────────────────────────

if "metadata" in st.session_state:
    md   = st.session_state["metadata"]
    anlz = st.session_state.get("analysis", "")
    trx  = st.session_state.get("transcript", "")

    st.divider()

    # Summary cards
    if "video_summary" in md:
        st.info(f"📝 **Video Summary:** {md['video_summary']}")
    if "content_category" in md:
        st.caption(f"Category: `{md['content_category']}`")

    # Expandable: raw outputs
    with st.expander("🔍 Raw video analysis (LLM output)"):
        st.write(anlz)
    if trx:
        with st.expander("🎙️ Audio transcript"):
            st.write(trx)

    st.subheader("📦 Platform Metadata")

    PLATFORM_CONFIG = {
        "youtube":   ("🔴 YouTube",   "#FF0000"),
        "facebook":  ("🔵 Facebook",  "#1877F2"),
        "instagram": ("📸 Instagram", "#E1306C"),
        "tiktok":    ("🎵 TikTok",    "#010101"),
        "linkedin":  ("💼 LinkedIn",  "#0A66C2"),
    }

    tab_labels = [v[0] for v in PLATFORM_CONFIG.values()]
    tabs = st.tabs(tab_labels)

    for tab, (pkey, (plabel, pcolor)) in zip(tabs, PLATFORM_CONFIG.items()):
        with tab:
            pdata = md.get(pkey)
            if not pdata or not isinstance(pdata, dict):
                st.warning("No data generated for this platform.")
                continue

            col_left, col_right = st.columns([3, 2])

            with col_left:
                # Title
                title_val = pdata.get("title", "")
                st.text_input(
                    f"Title ({len(title_val)} chars)",
                    value=title_val,
                    key=f"title_{pkey}",
                )

                # Description
                desc_val = pdata.get("description", "")
                st.text_area(
                    f"Description ({len(desc_val)} chars)",
                    value=desc_val,
                    height=220,
                    key=f"desc_{pkey}",
                )

                # CTA
                if pdata.get("cta"):
                    st.text_input("Call to Action", value=pdata["cta"], key=f"cta_{pkey}")

            with col_right:
                # Hashtags
                tags = pdata.get("hashtags", [])
                tags_str = "\n".join(tags) if isinstance(tags, list) else tags
                st.text_area(
                    f"Hashtags ({len(tags) if isinstance(tags, list) else '—'})",
                    value=tags_str,
                    height=160,
                    key=f"tags_{pkey}",
                )

                # Keywords
                kws = pdata.get("keywords", [])
                kws_str = ", ".join(kws) if isinstance(kws, list) else kws
                st.text_area(
                    "SEO Keywords",
                    value=kws_str,
                    height=90,
                    key=f"kw_{pkey}",
                )

                # Posting tip
                if pdata.get("posting_tip"):
                    st.success(f"⏰ {pdata['posting_tip']}")

            # Quick-copy row
            st.caption("Quick copy:")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.download_button(
                    "⬇️ Title",
                    data=pdata.get("title", ""),
                    file_name=f"{pkey}_title.txt",
                    mime="text/plain",
                    key=f"dl_title_{pkey}",
                )
            with c2:
                st.download_button(
                    "⬇️ Description",
                    data=pdata.get("description", ""),
                    file_name=f"{pkey}_description.txt",
                    mime="text/plain",
                    key=f"dl_desc_{pkey}",
                )
            with c3:
                tags_dl = "\n".join(tags) if isinstance(tags, list) else tags
                st.download_button(
                    "⬇️ Hashtags",
                    data=tags_dl,
                    file_name=f"{pkey}_hashtags.txt",
                    mime="text/plain",
                    key=f"dl_tags_{pkey}",
                )

    # ── Export section
    st.divider()
    st.subheader("⬇️ Export all")

    ecol1, ecol2, ecol3 = st.columns(3)

    with ecol1:
        st.download_button(
            "📄 Download JSON",
            data=json.dumps(md, indent=2, ensure_ascii=False),
            file_name="video_metadata.json",
            mime="application/json",
            use_container_width=True,
        )

    with ecol2:
        # Build CSV
        csv_rows = []
        for pkey, (plabel, _) in PLATFORM_CONFIG.items():
            pdata = md.get(pkey, {})
            if isinstance(pdata, dict):
                tags = pdata.get("hashtags", [])
                kws  = pdata.get("keywords", [])
                csv_rows.append({
                    "Platform":    plabel,
                    "Title":       pdata.get("title", ""),
                    "Description": pdata.get("description", ""),
                    "Hashtags":    " ".join(tags) if isinstance(tags, list) else tags,
                    "Keywords":    ", ".join(kws)  if isinstance(kws,  list) else kws,
                    "CTA":         pdata.get("cta", ""),
                    "PostingTip":  pdata.get("posting_tip", ""),
                })
        out = csv_io.StringIO()
        if csv_rows:
            writer = csv.DictWriter(out, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        st.download_button(
            "📊 Download CSV",
            data=out.getvalue(),
            file_name="video_metadata.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with ecol3:
        # Plain text pack
        txt_parts = []
        for pkey, (plabel, _) in PLATFORM_CONFIG.items():
            pdata = md.get(pkey, {})
            if isinstance(pdata, dict):
                tags = pdata.get("hashtags", [])
                txt_parts.append(
                    f"{'='*40}\n{plabel}\n{'='*40}\n"
                    f"TITLE:\n{pdata.get('title','')}\n\n"
                    f"DESCRIPTION:\n{pdata.get('description','')}\n\n"
                    f"HASHTAGS:\n{' '.join(tags) if isinstance(tags,list) else tags}\n\n"
                    f"KEYWORDS:\n{', '.join(pdata.get('keywords',[]))}\n\n"
                    f"CTA: {pdata.get('cta','')}\n"
                    f"POSTING TIP: {pdata.get('posting_tip','')}\n\n"
                )
        st.download_button(
            "📝 Download TXT",
            data="\n".join(txt_parts),
            file_name="video_metadata.txt",
            mime="text/plain",
            use_container_width=True,
        )
