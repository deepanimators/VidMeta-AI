# 🎬 VidMeta AI Hub

Analyze local videos with AI → Generate upload-ready metadata for **YouTube, Instagram, Facebook, TikTok, LinkedIn** — all in one shot.

---

## ✅ Prerequisites

| Tool | Required | Install |
|------|----------|---------|
| Python 3.10+ | Yes | [python.org](https://python.org) |
| ffmpeg | Yes (for audio) | See below |
| Ollama | Only for local LLM | [ollama.com](https://ollama.com) |

### Install ffmpeg

**Mac:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html), add to PATH.

---

## 🚀 Setup

```bash
# 1. Clone / download this folder
cd VidMeta-AI

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## 🤖 LLM Provider Setup

### Option A — Ollama (100% Free, Local, Private)
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a vision model
ollama pull llava              # 4GB — recommended
ollama pull moondream          # 1.7GB — lightweight
ollama pull llava-llama3       # 8GB — best quality
```
In the app sidebar: select **Ollama — Local/Free**, enter `http://localhost:11434`, model name `llava`.

### Option B — OpenRouter (Free tier available)
1. Sign up at [openrouter.ai](https://openrouter.ai/)
2. Get API key (free tier available)
3. Best free vision model: `meta-llama/llama-3.2-11b-vision-instruct:free`

### Option C — OpenAI
API key from [platform.openai.com](https://platform.openai.com/api-keys). Use `gpt-4o` for best results.

### Option D — Anthropic Claude
API key from [console.anthropic.com](https://console.anthropic.com/). Use `claude-opus-4-6` for best results.

### Option E — Google Gemini
API key from [aistudio.google.com](https://aistudio.google.com/apikey). Free tier available.

---

## 🎬 How to Use

1. **Set up LLM** in the sidebar (pick provider, enter key, select model)
2. **Upload video** or paste local file path
3. Hit **Analyze & Generate**
4. Wait ~30-90 seconds (depends on video length + LLM speed)
5. Get metadata for all 5 platforms — title, description, hashtags, keywords, CTA, posting tips
6. **Export** as JSON / CSV / TXT

---

## 📁 Output Fields per Platform

| Field | YouTube | Instagram | Facebook | TikTok | LinkedIn |
|-------|---------|-----------|----------|--------|----------|
| Title | ✅ (60 chars) | ✅ hook | ✅ | ✅ (100 chars) | ✅ |
| Description | ✅ 300-400 words | ✅ 200 words | ✅ 150 words | ✅ 150 words | ✅ 250 words |
| Hashtags | ✅ 10 | ✅ 30 | ✅ 5 | ✅ 15 | ✅ 7 |
| SEO Keywords | ✅ 15 | ✅ 10 | ✅ 8 | ✅ 8 | ✅ 8 |
| CTA | ✅ | ✅ | ✅ | ✅ | ✅ |
| Posting tip | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## ⚙️ Settings Reference

| Setting | Description |
|---------|-------------|
| Whisper model | `tiny` = fast, `base` = balanced, `small/medium` = accurate |
| Frame interval | Extract 1 frame every N seconds |
| Max frames | How many frames sent to LLM (more = better analysis, slower) |
| Brand context | Used to tailor all metadata to your brand |

---

## 🔧 Troubleshooting

**`cv2` import error:**
```bash
pip install opencv-python-headless
```

**ffmpeg not found:**
Make sure ffmpeg is installed and in your PATH. Test: `ffmpeg -version`

**Ollama connection refused:**
Make sure Ollama is running: `ollama serve`

**LLM returns non-JSON:**
Try a more capable model. `gpt-4o`, `claude-opus-4-6`, or `llava-llama3` are most reliable.

**Out of memory with large videos:**
Reduce `Max frames` to 3-4, increase `Frame interval` to 10-15 seconds.
