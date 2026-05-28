import React from "react";
import ReactDOM from "react-dom/client";
import {
  AlertTriangle,
  CheckCircle2,
  Download,
  Eye,
  EyeOff,
  FolderOpen,
  HardDrive,
  History,
  Loader2,
  Play,
  Save,
  Settings,
  UploadCloud,
  Video,
  RotateCcw,
  X as XIcon
} from "lucide-react";
import "./styles.css";

type BrandContext = {
  brand_name: string;
  brand_niche: string;
  target_audience: string;
  tone: string;
  custom_instructions: string;
};

type VideoSettings = {
  use_whisper: boolean;
  whisper_model_size: string;
  frame_interval: number;
  max_frames: number;
  enable_object_detection: boolean;
  enable_ocr: boolean;
  enable_frame_captioning: boolean;
};

type ProviderSettings = {
  provider: string;
  model: string;
  api_key: string;
  api_base: string;
  ollama_url: string;
};

type StorageSettings = {
  backend: string;
  local_data_dir: string;
  import_local_files: boolean;
  s3_endpoint_url: string;
  s3_bucket: string;
  s3_region: string;
  s3_access_key_id: string;
  s3_secret_access_key: string;
};

type AppSettings = {
  app_mode: string;
  max_upload_mb: number;
  upload_retention_days: number;
  brand_context: BrandContext;
  video_settings: VideoSettings;
  provider_settings: ProviderSettings;
  storage_settings: StorageSettings;
};

type VideoMetaDetails = {
  duration_seconds?: number;
  width?: number;
  height?: number;
  fps?: number;
  aspect_ratio?: string;
  orientation?: string;
};

type JobEventDetails = {
  thumbnails?: string[];
  video_metadata?: VideoMetaDetails;
  transcript?: string;
  transcript_preview?: string;
  transcript_characters?: number;
  analysis_preview?: string;
  analysis_characters?: number;
  vision_warning?: string;
  frame_count?: number;
  [key: string]: unknown;
};

type JobEvent = {
  id: number;
  stage: string;
  progress: number;
  message: string;
  details?: JobEventDetails;
  created_at?: string;
};

type LiveExtractionData = {
  thumbnails: string[];
  videoMetadata: VideoMetaDetails;
  transcript: string;
  transcriptPreview: string;
  analysisPreview: string;
  visionWarning: string;
};

type Job = {
  id: string;
  source_type: string;
  source_path: string;
  mode: string;
  status: string;
  stage: string;
  progress: number;
  error_message?: string | null;
  created_at?: string;
  updated_at?: string;
  completed_at?: string | null;
  events?: JobEvent[];
  request?: {
    provider_settings?: ProviderSettings;
    brand_context?: BrandContext;
    video_settings?: VideoSettings;
    target_platforms?: string[];
  };
};

type JobResult = Job & {
  transcript: string;
  analysis: string;
  metadata: Record<string, unknown>;
};

const _isTauri = typeof window !== "undefined" && Boolean(
  (window as Window & { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__
);
const isDesktopApp = _isTauri;
const API_BASE = _isTauri ? "http://127.0.0.1:8000" : (import.meta.env.VITE_API_BASE ?? "");
const CUSTOM_MODEL_VALUE = "__custom__";
const VIDEO_EXTENSIONS = new Set(["mp4", "mov", "avi", "mkv", "webm", "m4v"]);

type SocialPlatform = {
  key: string;
  label: string;
  short: string;
};

type SocialPlatformGroup = {
  title: string;
  platforms: readonly SocialPlatform[];
};

const SOCIAL_PLATFORM_GROUPS: readonly SocialPlatformGroup[] = [
  {
    title: "Core video networks",
    platforms: [
      { key: "youtube", label: "YouTube", short: "YT" },
      { key: "youtube_shorts", label: "YouTube Shorts", short: "YS" },
      { key: "instagram_reels", label: "Instagram Reels", short: "IR" },
      { key: "instagram_feed", label: "Instagram Feed", short: "IF" },
      { key: "facebook", label: "Facebook", short: "FB" },
      { key: "facebook_reels", label: "Facebook Reels", short: "FR" },
      { key: "tiktok", label: "TikTok", short: "TT" },
      { key: "linkedin", label: "LinkedIn", short: "IN" }
    ]
  },
  {
    title: "Social conversation",
    platforms: [
      { key: "x", label: "X / Twitter", short: "X" },
      { key: "threads", label: "Threads", short: "TH" },
      { key: "bluesky", label: "Bluesky", short: "BS" },
      { key: "mastodon", label: "Mastodon", short: "MA" },
      { key: "reddit", label: "Reddit", short: "RD" },
      { key: "quora", label: "Quora", short: "QU" }
    ]
  },
  {
    title: "Messaging and communities",
    platforms: [
      { key: "whatsapp_channels", label: "WhatsApp Channels", short: "WA" },
      { key: "telegram_channels", label: "Telegram Channels", short: "TG" },
      { key: "discord", label: "Discord", short: "DC" },
      { key: "line_voom", label: "LINE VOOM", short: "LV" }
    ]
  },
  {
    title: "Visual discovery",
    platforms: [
      { key: "pinterest", label: "Pinterest", short: "PI" },
      { key: "snapchat_spotlight", label: "Snapchat Spotlight", short: "SC" },
      { key: "lemon8", label: "Lemon8", short: "L8" },
      { key: "tumblr", label: "Tumblr", short: "TB" }
    ]
  },
  {
    title: "Publishing and long-form",
    platforms: [
      { key: "medium", label: "Medium", short: "MD" },
      { key: "substack_notes", label: "Substack Notes", short: "SN" },
      { key: "twitch", label: "Twitch", short: "TW" },
      { key: "vimeo", label: "Vimeo", short: "VM" },
      { key: "rumble", label: "Rumble", short: "RU" },
      { key: "dailymotion", label: "Dailymotion", short: "DM" }
    ]
  },
  {
    title: "Regional high-scale networks",
    platforms: [
      { key: "wechat_channels", label: "WeChat Channels", short: "WC" },
      { key: "douyin", label: "Douyin", short: "DY" },
      { key: "kuaishou", label: "Kuaishou", short: "KS" },
      { key: "bilibili", label: "Bilibili", short: "BB" },
      { key: "weibo", label: "Weibo", short: "WB" },
      { key: "vk", label: "VK", short: "VK" },
      { key: "sharechat", label: "ShareChat", short: "SH" },
      { key: "moj", label: "Moj", short: "MJ" },
      { key: "josh", label: "Josh", short: "JO" }
    ]
  }
] as const;

const SOCIAL_PLATFORMS = SOCIAL_PLATFORM_GROUPS.flatMap((group) => group.platforms);
const PLATFORM_LABELS = Object.fromEntries(SOCIAL_PLATFORMS.map((platform) => [platform.key, platform.label]));
const PLATFORM_BY_KEY = Object.fromEntries(SOCIAL_PLATFORMS.map((platform) => [platform.key, platform])) as Record<string, SocialPlatform>;
const CORE_PLATFORM_KEYS = SOCIAL_PLATFORM_GROUPS[0].platforms.map((platform) => platform.key);

const MODEL_PRESETS: Record<string, { label: string; value: string; note?: string }[]> = {
  ollama: [
    { label: "Gemma 3 4B Vision", value: "gemma3:4b", note: "small local vision default" },
    { label: "Gemma 3 12B Vision", value: "gemma3:12b" },
    { label: "Llama 3.2 Vision 11B", value: "llama3.2-vision:11b" },
    { label: "Llama 3.2 Vision", value: "llama3.2-vision" }
  ],
  openrouter: [
    { label: "OpenRouter Auto", value: "openrouter/auto", note: "routes to an available model" },
    { label: "Gemini 2.5 Flash", value: "google/gemini-2.5-flash" },
    { label: "Claude Sonnet 4.5", value: "anthropic/claude-sonnet-4.5" },
    { label: "GPT-5.4 Mini", value: "openai/gpt-5.4-mini" }
  ],
  openai: [
    { label: "GPT-5.4 Mini", value: "gpt-5.4-mini", note: "balanced default" },
    { label: "GPT-5.4", value: "gpt-5.4" },
    { label: "GPT-5.4 Nano", value: "gpt-5.4-nano" },
    { label: "GPT-5.5", value: "gpt-5.5" }
  ],
  anthropic: [
    { label: "Claude Sonnet 4.5", value: "claude-sonnet-4-5-20250929", note: "balanced default" },
    { label: "Claude Haiku 4.5", value: "claude-haiku-4-5-20251001" },
    { label: "Claude Opus 4.1", value: "claude-opus-4-1-20250805" }
  ],
  gemini: [
    { label: "Gemini 2.5 Flash", value: "gemini-2.5-flash", note: "fast metadata default" },
    { label: "Gemini 2.5 Pro", value: "gemini-2.5-pro" },
    { label: "Gemini 2.5 Flash-Lite", value: "gemini-2.5-flash-lite" }
  ],
  nvidia: [
    { label: "Llama 4 Scout 17B Vision", value: "meta/llama-4-scout-17b-16e-instruct", note: "vision · fast default" },
    { label: "Llama 4 Maverick 17B Vision", value: "meta/llama-4-maverick-17b-128e-instruct", note: "vision · balanced" },
    { label: "Llama 3.2 90B Vision", value: "meta/llama-3.2-90b-vision-instruct", note: "vision · best quality" },
    { label: "Llama 3.2 11B Vision", value: "meta/llama-3.2-11b-vision-instruct", note: "vision · lightweight" },
    { label: "Qwen2-VL 72B", value: "qwen/qwen2-vl-72b-instruct", note: "vision · top accuracy" },
    { label: "Qwen2-VL 7B", value: "qwen/qwen2-vl-7b-instruct", note: "vision · compact" },
    { label: "Phi-4 Multimodal", value: "microsoft/phi-4-multimodal-instruct", note: "vision · Microsoft" },
    { label: "Phi-3.5 Vision", value: "microsoft/phi-3.5-vision-instruct", note: "vision · efficient" },
    { label: "Gemma 3 27B", value: "google/gemma-3-27b-instruct", note: "vision · Google" },
    { label: "Gemma 3 12B", value: "google/gemma-3-12b-instruct", note: "vision · balanced" },
    { label: "Gemma 3 4B", value: "google/gemma-3-4b-instruct", note: "vision · lightweight" },
    { label: "Llama 3.3 70B", value: "meta/llama-3.3-70b-instruct", note: "text · high quality" },
    { label: "Nemotron Nano 8B", value: "nvidia/llama-3.1-nemotron-nano-8b-v1", note: "text · ultra-fast" },
    { label: "DeepSeek R1", value: "deepseek-ai/deepseek-r1", note: "text · reasoning" }
  ],
  groq: [
    { label: "Llama 4 Scout 17B Vision", value: "meta-llama/llama-4-scout-17b-16e-instruct", note: "vision · fastest" },
    { label: "Llama 4 Maverick 17B Vision", value: "meta-llama/llama-4-maverick-17b-128e-instruct", note: "vision · balanced" },
    { label: "Llama 3.2 90B Vision", value: "llama-3.2-90b-vision-preview", note: "vision · high quality" },
    { label: "Llama 3.2 11B Vision", value: "llama-3.2-11b-vision-preview", note: "vision · lightweight" },
    { label: "LLaVA 1.5 7B", value: "llava-v1.5-7b-4096-preview", note: "vision · legacy" },
    { label: "Llama 3.3 70B Versatile", value: "llama-3.3-70b-versatile", note: "text · default" },
    { label: "Llama 3.1 8B Instant", value: "llama-3.1-8b-instant", note: "text · ultra-fast" },
    { label: "DeepSeek R1 70B", value: "deepseek-r1-distill-llama-70b", note: "text · reasoning" },
    { label: "Qwen QwQ 32B", value: "qwen-qwq-32b", note: "text · reasoning" },
    { label: "Gemma 2 9B", value: "gemma2-9b-it", note: "text · efficient" },
    { label: "Mixtral 8x7B 32K", value: "mixtral-8x7b-32768", note: "text · long context" }
  ]
};

const DEFAULT_MODEL_BY_PROVIDER = Object.fromEntries(
  Object.entries(MODEL_PRESETS).map(([provider, models]) => [provider, models[0]?.value ?? ""])
) as Record<string, string>;

async function api<T>(path: string, init?: RequestInit, timeoutMs = 30_000): Promise<T> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${API_BASE}${path}`, {
      signal: controller.signal,
      headers: init?.body instanceof FormData ? undefined : { "Content-Type": "application/json" },
      ...init
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || response.statusText);
    }
    return response.json() as Promise<T>;
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new Error("Request timed out. The service may be busy.");
    }
    throw err;
  } finally {
    window.clearTimeout(timer);
  }
}

const defaultSettings: AppSettings = {
  app_mode: isDesktopApp ? "desktop" : "local",
  max_upload_mb: 2048,
  upload_retention_days: 0,
  brand_context: {
    brand_name: "",
    brand_niche: "",
    target_audience: "",
    tone: "",
    custom_instructions: ""
  },
  video_settings: {
    use_whisper: true,
    whisper_model_size: "base",
    frame_interval: 5,
    max_frames: 20,
    enable_object_detection: true,
    enable_ocr: true,
    enable_frame_captioning: true
  },
  provider_settings: {
    provider: "ollama",
    model: "gemma3:4b",
    api_key: "",
    api_base: "",
    ollama_url: "http://localhost:11434"
  },
  storage_settings: {
    backend: "local_disk",
    local_data_dir: "",
    import_local_files: false,
    s3_endpoint_url: "",
    s3_bucket: "",
    s3_region: "",
    s3_access_key_id: "",
    s3_secret_access_key: ""
  }
};

function App() {
  const [settings, setSettings] = React.useState<AppSettings>(defaultSettings);
  const [jobs, setJobs] = React.useState<Job[]>([]);
  const [selectedJobId, setSelectedJobId] = React.useState("");
  const [result, setResult] = React.useState<JobResult | null>(null);
  const [path, setPath] = React.useState("");
  const [uploadFiles, setUploadFiles] = React.useState<File[]>([]);
  const [selectedPlatforms, setSelectedPlatforms] = React.useState<string[]>(() => [...CORE_PLATFORM_KEYS]);
  const [desktopPickerAvailable, setDesktopPickerAvailable] = React.useState(false);
  const [busy, setBusy] = React.useState(false);
  const [message, setMessage] = React.useState("");
  const [uploadProgress, setUploadProgress] = React.useState(0);
  const [liveData, setLiveData] = React.useState<Partial<LiveExtractionData> | null>(null);
  const [providerKeys, setProviderKeys] = React.useState<Record<string, string>>(() => {
    try { return JSON.parse(localStorage.getItem("vidmeta_provider_keys") ?? "{}"); } catch { return {}; }
  });
  const [showApiKey, setShowApiKey] = React.useState(false);
  const browserFilePickerRef = React.useRef<HTMLInputElement>(null);
  const browserFolderPickerRef = React.useRef<HTMLInputElement>(null);

  const selectedJob = React.useMemo(
    () => jobs.find((job) => job.id === selectedJobId) ?? jobs[0],
    [jobs, selectedJobId]
  );
  const selectedJobDetails = React.useMemo(() => {
    if (!selectedJob) return undefined;
    if (result?.id === selectedJob.id && result.events?.length) {
      return { ...selectedJob, events: result.events };
    }
    return selectedJob;
  }, [result, selectedJob]);
  const selectedResult = result?.id === selectedJob?.id ? result : null;

  React.useEffect(() => {
    localStorage.setItem("vidmeta_provider_keys", JSON.stringify(providerKeys));
  }, [providerKeys]);

  React.useEffect(() => {
    void refresh();
    void loadSettings();
    void detectDesktopPicker();
    // Adaptive polling: fast when jobs are active, slow when idle.
    let timer: number;
    function scheduleNext() {
      const hasActive = jobs.some((j) => j.status === "running" || j.status === "queued");
      timer = window.setTimeout(() => { void refresh().then(scheduleNext); }, hasActive ? 2000 : 15000);
    }
    scheduleNext();
    return () => window.clearTimeout(timer);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  React.useEffect(() => {
    browserFolderPickerRef.current?.setAttribute("webkitdirectory", "");
    browserFolderPickerRef.current?.setAttribute("directory", "");
  }, []);

  React.useEffect(() => {
    if (selectedJob?.status === "completed") {
      void loadResult(selectedJob.id);
    }
  }, [selectedJob?.id, selectedJob?.status]);

  // Clear live data when a different job is selected.
  React.useEffect(() => {
    setLiveData(null);
  }, [selectedJobId]);

  // For completed/failed jobs: fetch full event details (including thumbnails) once.
  React.useEffect(() => {
    if (!selectedJob || (selectedJob.status !== "completed" && selectedJob.status !== "failed")) return;
    api<{ events?: JobEvent[] }>(`/api/jobs/${selectedJob.id}`).then((full) => {
      const all = full.events ?? [];
      // Pick the last event for each stage that actually carries the relevant data.
      const framesEv = all.filter((e) => e.stage === "frames" && e.details?.thumbnails?.length).pop()
        ?? all.filter((e) => e.stage === "frames").pop();
      const audioEv = all.filter((e) => e.stage === "audio" && (e.details?.transcript || e.details?.transcript_preview)).pop()
        ?? all.filter((e) => e.stage === "audio").pop();
      const analysisEv = all.filter((e) => e.stage === "analysis" && e.details?.analysis_preview).pop();
      setLiveData({
        thumbnails: framesEv?.details?.thumbnails ?? [],
        videoMetadata: framesEv?.details?.video_metadata ?? {},
        transcript: audioEv?.details?.transcript ?? audioEv?.details?.transcript_preview ?? "",
        transcriptPreview: audioEv?.details?.transcript_preview ?? "",
        analysisPreview: analysisEv?.details?.analysis_preview ?? "",
        visionWarning: analysisEv?.details?.vision_warning ?? framesEv?.details?.vision_warning ?? "",
      });
    }).catch(() => {});
  }, [selectedJob?.id, selectedJob?.status]);

  // SSE stream for running/queued jobs — pushes live extraction data to UI.
  React.useEffect(() => {
    const status = selectedJob?.status;
    if (!selectedJobId || status === "completed" || status === "failed" || !status) return;
    const es = new EventSource(`${API_BASE}/api/jobs/${selectedJobId}/stream`);
    const updateSelectedJob = (patch: Partial<Job>) => {
      setJobs((current) => current.map((job) => (job.id === selectedJobId ? { ...job, ...patch } : job)));
    };
    es.onmessage = (e: MessageEvent<string>) => {
      try {
        const data = JSON.parse(e.data) as { type: string; stage?: string; details?: JobEventDetails };
        if (data.type === "status") {
          if (typeof (data as { progress?: number }).progress === "number") {
            updateSelectedJob({
              status: (data as { status?: string }).status ?? "running",
              stage: data.stage ?? status,
              progress: (data as { progress?: number }).progress ?? 0,
            });
          }
          return;
        }
        if (data.type !== "event" || !data.details) return;
        const { stage, details } = data;
        if (typeof (data as { progress?: number }).progress === "number") {
          updateSelectedJob({
            status: "running",
            stage: stage ?? status,
            progress: (data as { progress?: number }).progress ?? 0,
          });
        }
        if (stage === "frames") {
          setLiveData((prev) => ({
            ...prev,
            thumbnails: details.thumbnails ?? prev?.thumbnails ?? [],
            videoMetadata: details.video_metadata ?? prev?.videoMetadata ?? {},
          }));
        } else if (stage === "audio" && (details.transcript || details.transcript_preview)) {
          setLiveData((prev) => ({
            ...prev,
            transcript: details.transcript ?? prev?.transcript ?? details.transcript_preview ?? "",
            transcriptPreview: details.transcript_preview ?? "",
          }));
        } else if (stage === "analysis") {
          setLiveData((prev) => ({
            ...prev,
            analysisPreview: details.analysis_preview ?? prev?.analysisPreview ?? "",
            visionWarning: details.vision_warning ?? prev?.visionWarning ?? "",
          }));
        }
      } catch {
        // Ignore parse errors from malformed SSE frames.
      }
    };
    es.addEventListener("done", () => {
      es.close();
      void refresh();
    });
    es.onerror = () => es.close();
    return () => es.close();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedJobId, selectedJob?.status]);

  async function loadSettings() {
    try {
      setSettings(normalizeSettings(await api<AppSettings>("/api/settings")));
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not load settings");
    }
  }

  async function detectDesktopPicker() {
    setDesktopPickerAvailable(await canUseDesktopDialog());
  }

  async function refresh() {
    try {
      const nextJobs = await api<Job[]>("/api/jobs");
      setJobs(nextJobs);
      setSelectedJobId((current) => {
        if (current && nextJobs.some((job) => job.id === current)) {
          return current;
        }
        return nextJobs[0]?.id ?? "";
      });
    } catch {
      // Backend may still be starting.
    }
  }

  async function saveSettings() {
    setBusy(true);
    try {
      const payload = sanitizeSettingsForEnvironment(settings);
      const saved = await api<AppSettings>("/api/settings", {
        method: "PUT",
        body: JSON.stringify(payload)
      });
      setSettings(normalizeSettings(saved));
      setMessage("Settings saved");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Settings save failed");
    } finally {
      setBusy(false);
    }
  }

  async function createPathJob() {
    if (!path.trim()) {
      setMessage("Enter a local file or folder path");
      return;
    }
    if (!selectedPlatforms.length) {
      setMessage("Select at least one social platform");
      return;
    }
    setBusy(true);
    try {
      const job = await api<Job>("/api/jobs/from-path", {
        method: "POST",
        body: JSON.stringify({
          path: path.trim(),
          mode: "single",
          brand_context: settings.brand_context,
          video_settings: settings.video_settings,
          provider_settings: settings.provider_settings,
          target_platforms: selectedPlatforms
        })
      });
      setSelectedJobId(job.id);
      setMessage("Job queued");
      await refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not create job");
    } finally {
      setBusy(false);
    }
  }

  async function pickLocalSource(kind: "file" | "folder") {
    try {
      if (await canUseDesktopDialog()) {
        const dialog = await import("@tauri-apps/plugin-dialog");
        const picked = await dialog.open({
          directory: kind === "folder",
          multiple: false,
          filters: kind === "file" ? [{ name: "Video", extensions: Array.from(VIDEO_EXTENSIONS) }] : []
        });
        if (typeof picked === "string") {
          setPath(picked);
          setMessage("Local path selected");
        }
        return;
      }
      if (kind === "file") {
        browserFilePickerRef.current?.click();
      } else {
        browserFolderPickerRef.current?.click();
      }
      setMessage(
        kind === "folder"
          ? "Browser folder picker queues uploads. VidMeta AI Desktop is required for direct folder path processing."
          : "Browser file picker queues an upload. VidMeta AI Desktop is required for direct path processing."
      );
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not open picker");
    }
  }

  async function uploadAndRun() {
    if (!uploadFiles.length) {
      setMessage("Choose one or more video files");
      return;
    }
    if (!selectedPlatforms.length) {
      setMessage("Select at least one social platform");
      return;
    }
    setBusy(true);
    setUploadProgress(0);
    try {
      let lastJob: Job | null = null;
      for (let index = 0; index < uploadFiles.length; index += 1) {
        const file = uploadFiles[index];
        setMessage(`Uploading ${index + 1} of ${uploadFiles.length}: ${file.name}`);
        const job = await uploadOneFile(file, (value) => {
          const completed = index / uploadFiles.length;
          setUploadProgress(Math.round((completed + value / 100 / uploadFiles.length) * 100));
        });
        lastJob = job;
      }
      if (lastJob) setSelectedJobId(lastJob.id);
      setMessage(`${uploadFiles.length} upload job${uploadFiles.length === 1 ? "" : "s"} queued`);
      await refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  }

  async function uploadOneFile(file: File, onProgress: (progress: number) => void) {
    const created = await api<{ id: string; upload_url: string }>("/api/uploads/resumable", {
      method: "POST",
      body: JSON.stringify({
        filename: file.name,
        content_type: file.type || "application/octet-stream",
        size_bytes: file.size
      })
    });
    await uploadResumable(created.upload_url, file, onProgress);
    return api<Job>(`/api/jobs/from-upload/${created.id}`, {
      method: "POST",
      body: JSON.stringify({
        brand_context: settings.brand_context,
        video_settings: settings.video_settings,
        provider_settings: settings.provider_settings,
        target_platforms: selectedPlatforms
      })
    });
  }

  async function loadResult(jobId: string) {
    try {
      setResult(await api<JobResult>(`/api/jobs/${jobId}/result`));
    } catch {
      setResult(null);
    }
  }

  async function retryJob(jobId: string, providerSettings?: ProviderSettings) {
    try {
      const body = providerSettings ? { provider_settings: providerSettings } : {};
      const job = await api<Job>(`/api/jobs/${jobId}/retry`, {
        method: "POST",
        body: JSON.stringify(body),
      });
      setSelectedJobId(job.id);
      setMessage("Job retried");
      await refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Retry failed");
    }
  }

  async function stopJob(jobId: string) {
    if (!confirm("Stop this job? This will attempt to cancel analysis.")) return;
    try {
      await api(`/api/jobs/${jobId}/stop`, { method: "POST" });
      setMessage("Stop requested");
      await refresh();
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Stop failed");
    }
  }

  async function deleteJob(jobId: string) {
    try {
      await api(`/api/jobs/${jobId}`, { method: "DELETE" });
      if (result?.id === jobId) setResult(null);
      setSelectedJobId((current) => (current === jobId ? "" : current));
      await refresh();
      setMessage("Job deleted");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Delete failed");
    }
  }

  async function clearFailedJobs() {
    const failedIds = jobs.filter((j) => j.status === "failed").map((j) => j.id);
    if (!failedIds.length) return;
    try {
      await Promise.all(failedIds.map((id) => api(`/api/jobs/${id}`, { method: "DELETE" })));
      if (result && failedIds.includes(result.id)) setResult(null);
      setSelectedJobId((current) => (failedIds.includes(current) ? "" : current));
      await refresh();
      setMessage(`Cleared ${failedIds.length} failed job${failedIds.length === 1 ? "" : "s"}`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Clear failed");
    }
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <Video size={28} />
          <div>
            <h1>VidMeta AI</h1>
            <p>Local video intelligence service</p>
          </div>
        </div>

        {isDesktopApp && (
          <section className="panel compact">
            <h2><HardDrive size={18} /> Runtime</h2>
            <p className="mode-note">Desktop mode is locked to the local backend, local Ollama, and local disk storage.</p>
          </section>
        )}

        {/* <section className="panel compact">
          <h2><HardDrive size={18} /> Runtime</h2>
          <select aria-label="App mode" value={settings.app_mode} onChange={(event) => updateSettings("app_mode", event.target.value)}>
            <option value="local">Local web</option>
            <option value="desktop">VidMeta AI Desktop</option>
            <option value="hosted">Private hosted</option>
          </select>
          {settings.app_mode === "hosted" && (
            <p className="warning"><AlertTriangle size={15} /> Hosted mode has no auth yet. Use only on trusted private networks.</p>
          )}
        </section> */}

        <section className="panel compact">
          <h2><Settings size={18} /> Provider</h2>
          <select
            aria-label="AI provider"
            value={settings.provider_settings.provider}
            onChange={(event) => selectProvider(event.target.value)}
            disabled={isDesktopApp}
          >
            <option value="ollama">Ollama local</option>
            {!isDesktopApp && <>
              <option value="openrouter">OpenRouter</option>
              <option value="openai">OpenAI</option>
              <option value="anthropic">Anthropic</option>
              <option value="gemini">Gemini</option>
              <option value="nvidia">NVIDIA NIM</option>
              <option value="groq">Groq</option>
            </>}
          </select>
          <select
            aria-label="Model"
            value={selectedModelValue(settings.provider_settings.provider, settings.provider_settings.model)}
            onChange={(event) => selectModel(event.target.value)}
          >
            {(MODEL_PRESETS[settings.provider_settings.provider] ?? []).map((model) => (
              <option value={model.value} key={model.value}>
                {model.label}{model.note ? ` - ${model.note}` : ""}
              </option>
            ))}
            <option value={CUSTOM_MODEL_VALUE}>Other custom model</option>
          </select>
          {selectedModelValue(settings.provider_settings.provider, settings.provider_settings.model) === CUSTOM_MODEL_VALUE && (
            <input value={settings.provider_settings.model} onChange={(event) => updateProvider("model", event.target.value)} placeholder="Enter model ID" />
          )}
          {settings.provider_settings.provider === "ollama" ? (
            <input value={settings.provider_settings.ollama_url} onChange={(event) => updateProvider("ollama_url", event.target.value)} placeholder="Ollama URL" />
          ) : (
            !isDesktopApp && <>
              <div className="key-input-wrap">
                <input value={settings.provider_settings.api_key} onChange={(event) => updateApiKey(event.target.value)} placeholder="API key" type={showApiKey ? "text" : "password"} />
                <button type="button" className="key-toggle" onClick={() => setShowApiKey((v) => !v)} aria-label={showApiKey ? "Hide API key" : "Show API key"}>
                  {showApiKey ? <EyeOff size={14} /> : <Eye size={14} />}
                </button>
              </div>
              {['openrouter', 'openai'].includes(settings.provider_settings.provider) && (
                <input value={settings.provider_settings.api_base} onChange={(event) => updateProvider("api_base", event.target.value)} placeholder="API base URL (optional)" />
              )}
            </>
          )}
          {isDesktopApp && <p className="mode-note">External providers are disabled in desktop mode.</p>}
        </section>

        <section className="panel compact">
          <h2><Save size={18} /> Storage</h2>
          <select
            aria-label="Storage backend"
            value={settings.storage_settings.backend}
            onChange={(event) => updateStorage("backend", event.target.value)}
            disabled={isDesktopApp}
          >
            <option value="local_disk">Local server disk</option>
            {!isDesktopApp && <option value="s3_compatible">S3 compatible</option>}
          </select>
          {isDesktopApp && <p className="mode-note">Desktop mode keeps storage local; S3 settings are unavailable.</p>}
          <label className="check">
            <input
              type="checkbox"
              checked={settings.storage_settings.import_local_files}
              onChange={(event) => updateStorage("import_local_files", event.target.checked)}
            />
            Import local path files into library
          </label>
          {settings.storage_settings.backend === "s3_compatible" && (
            <div className="stack">
              <input value={settings.storage_settings.s3_bucket} onChange={(event) => updateStorage("s3_bucket", event.target.value)} placeholder="Bucket" />
              <input value={settings.storage_settings.s3_endpoint_url} onChange={(event) => updateStorage("s3_endpoint_url", event.target.value)} placeholder="Endpoint URL" />
              <input value={settings.storage_settings.s3_region} onChange={(event) => updateStorage("s3_region", event.target.value)} placeholder="Region" />
              <input value={settings.storage_settings.s3_access_key_id} onChange={(event) => updateStorage("s3_access_key_id", event.target.value)} placeholder="Access key" />
              <input value={settings.storage_settings.s3_secret_access_key} onChange={(event) => updateStorage("s3_secret_access_key", event.target.value)} placeholder="Secret key" type="password" />
            </div>
          )}
          <label className="inline-label">
            Upload retention (days, 0 = keep forever)
            <input
              type="number"
              min={0}
              value={settings.upload_retention_days}
              onChange={(e) => updateSettings("upload_retention_days", Number(e.target.value))}
            />
          </label>
          <button type="button" className="secondary" onClick={saveSettings} disabled={busy}><Save size={16} /> Save settings</button>
        </section>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <h2>Generate platform metadata</h2>
            <p>
              {desktopPickerAvailable
                ? "Desktop mode uses real local paths for fast no-upload processing."
                : "Web mode uses resumable uploads and only shows upload controls."}
            </p>
          </div>
          <div className="status-pill">{message || "Service ready"}</div>
        </header>

        <div className="source-grid">
          {desktopPickerAvailable ? (
            <section className="panel main-card">
              <h2><FolderOpen size={19} /> Local path or folder</h2>
              <p className="mode-note">VidMeta AI Desktop passes native file and folder paths to the local service.</p>
              <div className="path-row">
                <input value={path} onChange={(event) => setPath(event.target.value)} placeholder="/Users/you/Videos/product-demo.mp4" />
                <button type="button" className="icon-button" onClick={() => pickLocalSource("file")} title="Pick video file" aria-label="Pick video file"><Video size={18} /></button>
                <button type="button" className="icon-button" onClick={() => pickLocalSource("folder")} title="Pick video folder" aria-label="Pick video folder"><FolderOpen size={18} /></button>
              </div>
              <JobConfiguration
                settings={settings}
                selectedPlatforms={selectedPlatforms}
                onBrandChange={updateBrand}
                onVideoChange={updateVideo}
                onSelectedPlatformsChange={setSelectedPlatforms}
              />
              <button type="button" className="primary" onClick={createPathJob} disabled={busy || !selectedPlatforms.length}><Play size={17} /> Analyze local path</button>
            </section>
          ) : (
            <section className="panel main-card">
              <h2><UploadCloud size={19} /> Resumable browser upload</h2>
              <input
                ref={browserFilePickerRef}
                className="hidden-input"
                type="file"
                accept=".mp4,.mov,.avi,.mkv,.webm,.m4v,video/*"
                multiple
                aria-hidden="true"
                tabIndex={-1}
                onChange={(event) => handleBrowserFiles(event.target.files)}
              />
              <input
                ref={browserFolderPickerRef}
                className="hidden-input"
                type="file"
                accept=".mp4,.mov,.avi,.mkv,.webm,.m4v,video/*"
                multiple
                aria-hidden="true"
                tabIndex={-1}
                onChange={(event) => handleBrowserFiles(event.target.files)}
              />
              <div className="upload-picker">
                <button type="button" className="secondary action-button" onClick={() => browserFilePickerRef.current?.click()}><Video size={17} /> Select videos</button>
                <button type="button" className="secondary action-button" onClick={() => browserFolderPickerRef.current?.click()}><FolderOpen size={17} /> Select folder</button>
              </div>
              {uploadFiles.length > 0 && (
                <p className="selected-files">{uploadFiles.length} selected: {uploadFiles.slice(0, 2).map((file) => file.name).join(", ")}{uploadFiles.length > 2 ? "..." : ""}</p>
              )}
              <JobConfiguration
                settings={settings}
                selectedPlatforms={selectedPlatforms}
                onBrandChange={updateBrand}
                onVideoChange={updateVideo}
                onSelectedPlatformsChange={setSelectedPlatforms}
              />
              {uploadProgress > 0 && <progress className="progress-bar" value={uploadProgress} max={100} />}
              <button type="button" className="primary" onClick={uploadAndRun} disabled={busy || !uploadFiles.length || !selectedPlatforms.length}><UploadCloud size={17} /> Upload and analyze</button>
            </section>
          )}
        </div>

        <section className="panel">
          <div className="jobs-header">
            <h2><History size={19} /> Jobs <em>{jobs.length}</em></h2>
            {jobs.some((j) => j.status === "failed") && (
              <button type="button" className="mini-button danger" onClick={() => void clearFailedJobs()}>
                Clear failed
              </button>
            )}
          </div>
          <div className="jobs">
            {jobs.map((job) => (
              <div key={job.id} className={`job-row ${selectedJob?.id === job.id ? "selected" : ""}`}>
                <button type="button" className="job-row-select" onClick={() => setSelectedJobId(job.id)} aria-label={`Select job ${job.source_path.split("/").pop()}`}>
                  <span className={`dot ${job.status}`} />
                  <span className="job-main">
                    <strong>{job.source_path.split("/").pop()}</strong>
                    <small>{job.status} / {job.stage}</small>
                  </span>
                  <span className="job-progress">{job.progress}%</span>
                </button>
                {job.status === "failed" && (
                  <button
                    type="button"
                    className="job-retry"
                    title="Retry job"
                    aria-label="Retry job"
                    onClick={(e) => { e.stopPropagation(); void retryJob(job.id); }}
                  >
                    <RotateCcw size={13} />
                  </button>
                )}
                {job.status === "running" && (
                  <button
                    type="button"
                    className="job-stop"
                    title="Stop job"
                    aria-label="Stop job"
                    onClick={(e) => { e.stopPropagation(); void stopJob(job.id); }}
                  >
                    <XIcon size={13} />
                  </button>
                )}
                <button
                  type="button"
                  className="job-delete"
                  title="Delete job"
                  aria-label="Delete job"
                  disabled={job.status === "running"}
                  onClick={(e) => { e.stopPropagation(); void deleteJob(job.id); }}
                >
                  <XIcon size={13} />
                </button>
              </div>
            ))}
            {!jobs.length && <p className="muted">No jobs yet.</p>}
          </div>
          {selectedJobDetails && <JobProgressDetails job={selectedJobDetails} />}
          {selectedJob?.status === "failed" && (
            <RetryPanel
              job={selectedJob}
              localOnly={isDesktopApp}
              onRetry={(ps) => void retryJob(selectedJob.id, ps)}
            />
          )}
        </section>

        {liveData && <section className="panel"><LiveExtractionPanel data={liveData} /></section>}

        {selectedJob && (
          <section className="panel result-panel">
            <div className="result-header">
              <h2>{selectedJob.status === "completed" ? <CheckCircle2 size={19} /> : <Loader2 size={19} />} Result</h2>
              <div className="downloads">
                {["json", "csv", "txt"].map((format) => (
                  <a key={format} href={`${API_BASE}/api/jobs/${selectedJob.id}/exports/${format}`}><Download size={15} /> {format.toUpperCase()}</a>
                ))}
                {selectedJob.status === "running" && (
                  <button type="button" className="job-stop secondary" onClick={() => void stopJob(selectedJob.id)}>Stop</button>
                )}
              </div>
            </div>
            {selectedJob.status === "failed" && <p className="warning">{selectedJob.error_message}</p>}
            {selectedResult?.metadata ? <MetadataView metadata={selectedResult.metadata} analysis={selectedResult.analysis} transcript={selectedResult.transcript} /> : <p className="muted">Result appears here after the job completes.</p>}
          </section>
        )}
      </section>
    </main>
  );

  function updateSettings<K extends keyof AppSettings>(key: K, value: AppSettings[K]) {
    setSettings((current) => ({ ...current, [key]: value }));
  }
  function updateBrand<K extends keyof BrandContext>(key: K, value: BrandContext[K]) {
    setSettings((current) => ({ ...current, brand_context: { ...current.brand_context, [key]: value } }));
  }
  function updateVideo<K extends keyof VideoSettings>(key: K, value: VideoSettings[K]) {
    setSettings((current) => ({ ...current, video_settings: { ...current.video_settings, [key]: value } }));
  }
  function updateProvider<K extends keyof ProviderSettings>(key: K, value: ProviderSettings[K]) {
    setSettings((current) => ({ ...current, provider_settings: { ...current.provider_settings, [key]: value } }));
  }
  function updateApiKey(value: string) {
    setProviderKeys((prev) => ({ ...prev, [settings.provider_settings.provider]: value }));
    setSettings((current) => ({ ...current, provider_settings: { ...current.provider_settings, api_key: value } }));
  }
  function selectProvider(provider: string) {
    const currentProvider = settings.provider_settings.provider;
    const currentKey = settings.provider_settings.api_key;
    const merged = { ...providerKeys, [currentProvider]: currentKey };
    setProviderKeys(merged);
    setSettings((current) => ({
      ...current,
      provider_settings: {
        ...current.provider_settings,
        provider,
        api_key: merged[provider] ?? "",
        model: DEFAULT_MODEL_BY_PROVIDER[provider] ?? ""
      }
    }));
    setShowApiKey(false);
  }
  function selectModel(value: string) {
    if (value === CUSTOM_MODEL_VALUE) {
      updateProvider("model", "");
      return;
    }
    updateProvider("model", value);
  }
  function updateStorage<K extends keyof StorageSettings>(key: K, value: StorageSettings[K]) {
    setSettings((current) => ({ ...current, storage_settings: { ...current.storage_settings, [key]: value } }));
  }
  function handleBrowserFiles(fileList: FileList | null) {
    const files = Array.from(fileList ?? []).filter((file) => {
      const extension = file.name.split(".").pop()?.toLowerCase() ?? "";
      return file.type.startsWith("video/") || VIDEO_EXTENSIONS.has(extension);
    });
    setUploadFiles(files);
    setUploadProgress(0);
    setMessage(files.length ? `${files.length} video file${files.length === 1 ? "" : "s"} selected` : "No supported video files selected");
  }
}

type JobConfigurationProps = {
  settings: AppSettings;
  selectedPlatforms: string[];
  onBrandChange: <K extends keyof BrandContext>(key: K, value: BrandContext[K]) => void;
  onVideoChange: <K extends keyof VideoSettings>(key: K, value: VideoSettings[K]) => void;
  onSelectedPlatformsChange: React.Dispatch<React.SetStateAction<string[]>>;
};

function JobConfiguration({
  settings,
  selectedPlatforms,
  onBrandChange,
  onVideoChange,
  onSelectedPlatformsChange
}: JobConfigurationProps) {
  return (
    <div className="job-config">
      <div className="form-grid">
        <input value={settings.brand_context.brand_name} onChange={(event) => onBrandChange("brand_name", event.target.value)} placeholder="Brand name" />
        <input value={settings.brand_context.brand_niche} onChange={(event) => onBrandChange("brand_niche", event.target.value)} placeholder="Niche / industry" />
        <input value={settings.brand_context.target_audience} onChange={(event) => onBrandChange("target_audience", event.target.value)} placeholder="Target audience" />
        <input value={settings.brand_context.tone} onChange={(event) => onBrandChange("tone", event.target.value)} placeholder="Tone (e.g. casual, professional, energetic)" />
      </div>
      <textarea
        className="custom-instructions"
        value={settings.brand_context.custom_instructions}
        onChange={(event) => onBrandChange("custom_instructions", event.target.value)}
        placeholder="Custom instructions (optional) — e.g. always include emojis in TikTok captions, never use #ad, avoid jargon, prioritize CTA in first line"
        rows={3}
      />
      <div className="controls-row">
        <label>Frame interval <input type="number" min={1} max={120} value={settings.video_settings.frame_interval} onChange={(event) => onVideoChange("frame_interval", Number(event.target.value))} /></label>
        <label>Max frames <input type="number" min={1} max={60} value={settings.video_settings.max_frames} onChange={(event) => onVideoChange("max_frames", Number(event.target.value))} /></label>
        <label>Whisper <input type="checkbox" checked={settings.video_settings.use_whisper} onChange={(event) => onVideoChange("use_whisper", event.target.checked)} /></label>
      </div>
      <div className="enrichment-row">
        <span className="enrichment-label">Frame enrichment</span>
        <label className="check" title="Detect objects/people in each frame using YOLOv8n — requires: pip install ultralytics">
          <input type="checkbox" checked={settings.video_settings.enable_object_detection} onChange={(event) => onVideoChange("enable_object_detection", event.target.checked)} />
          Object detection
        </label>
        <label className="check" title="Extract visible text overlays, captions, product names — requires: pip install easyocr">
          <input type="checkbox" checked={settings.video_settings.enable_ocr} onChange={(event) => onVideoChange("enable_ocr", event.target.checked)} />
          OCR text
        </label>
        <label className="check" title="Generate a text caption per frame using moondream2 (~900MB download on first use) — requires: pip install moondream">
          <input type="checkbox" checked={settings.video_settings.enable_frame_captioning} onChange={(event) => onVideoChange("enable_frame_captioning", event.target.checked)} />
          Frame captions
        </label>
      </div>
      <PlatformPicker selected={selectedPlatforms} onChange={onSelectedPlatformsChange} />
    </div>
  );
}

function PlatformPicker({
  selected,
  onChange
}: {
  selected: string[];
  onChange: React.Dispatch<React.SetStateAction<string[]>>;
}) {
  const selectedSet = React.useMemo(() => new Set(selected), [selected]);
  const allPlatformKeys = React.useMemo(() => SOCIAL_PLATFORMS.map((platform) => platform.key), []);
  const [activeGroupIndex, setActiveGroupIndex] = React.useState(0);
  const activeGroup = SOCIAL_PLATFORM_GROUPS[activeGroupIndex] ?? SOCIAL_PLATFORM_GROUPS[0];
  const selectedPlatforms = selected.map((key) => PLATFORM_BY_KEY[key]).filter(Boolean);

  function setCorePlatforms() {
    onChange([...CORE_PLATFORM_KEYS]);
  }

  function setAllPlatforms() {
    onChange([...allPlatformKeys]);
  }

  function clearPlatforms() {
    onChange([]);
  }

  function togglePlatform(platformKey: string) {
    onChange((current) => {
      if (current.includes(platformKey)) {
        return current.filter((key) => key !== platformKey);
      }
      return [...current, platformKey];
    });
  }

  function selectedCountForGroup(group: SocialPlatformGroup) {
    return group.platforms.filter((platform) => selectedSet.has(platform.key)).length;
  }

  return (
    <div className="platform-picker">
      <div className="platform-picker-head">
        <div>
          <span className="eyebrow">Destinations</span>
          <strong>Choose social platforms</strong>
          <span>{selected.length} selected for generation</span>
        </div>
        <div className="platform-actions">
          <button type="button" className="mini-button" onClick={setCorePlatforms}>Core</button>
          <button type="button" className="mini-button" onClick={setAllPlatforms}>All</button>
          <button type="button" className="mini-button" onClick={clearPlatforms}>Clear</button>
        </div>
      </div>
      <div className="selected-platforms" aria-label="Selected platforms">
        {selectedPlatforms.length ? (
          selectedPlatforms.map((platform) => (
            <span className="selected-platform-chip" key={platform.key}>
              <span className="platform-mark compact" aria-hidden="true">{platform.short}</span>
              <span>{platform.label}</span>
              <button type="button" onClick={() => togglePlatform(platform.key)} title={`Remove ${platform.label}`}>
                <XIcon size={13} />
              </button>
            </span>
          ))
        ) : (
          <span className="platform-empty">Select at least one destination before generating.</span>
        )}
      </div>
      <div className="platform-tabs" role="tablist" aria-label="Platform groups">
        {SOCIAL_PLATFORM_GROUPS.map((group, index) => {
          const groupSelectedCount = selectedCountForGroup(group);
          const isActive = index === activeGroupIndex;
          return (
            <button
              type="button"
              role="tab"
              aria-selected={isActive}
              className={`platform-tab ${isActive ? "active" : ""}`}
              key={group.title}
              onClick={() => setActiveGroupIndex(index)}
            >
              <span>{group.title}</span>
              <em>{groupSelectedCount}/{group.platforms.length}</em>
            </button>
          );
        })}
      </div>
      <fieldset className="platform-group">
        <legend>{activeGroup.title}</legend>
        <div className="platform-options">
          {activeGroup.platforms.map((platform) => {
            const checked = selectedSet.has(platform.key);
            return (
              <label className={`platform-option ${checked ? "selected" : ""}`} key={platform.key}>
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={() => togglePlatform(platform.key)}
                />
                <span className="platform-mark" aria-hidden="true">{platform.short}</span>
                <span className="platform-option-name">{platform.label}</span>
                <span className="option-check" aria-hidden="true">
                  {checked ? <CheckCircle2 size={15} /> : null}
                </span>
              </label>
            );
          })}
        </div>
      </fieldset>
    </div>
  );
}

function normalizeSettings(settings: AppSettings): AppSettings {
  const provider = isDesktopApp ? "ollama" : (settings.provider_settings.provider || "ollama");
  const model =
    provider === "ollama" && settings.provider_settings.model === "gemma4"
      ? DEFAULT_MODEL_BY_PROVIDER.ollama
      : settings.provider_settings.model || DEFAULT_MODEL_BY_PROVIDER[provider] || "";
  return sanitizeSettingsForEnvironment({
    ...settings,
    app_mode: isDesktopApp ? "desktop" : (settings.app_mode || "local"),
    provider_settings: {
      ...settings.provider_settings,
      provider,
      model
    }
  });
}

function sanitizeSettingsForEnvironment(settings: AppSettings): AppSettings {
  if (!isDesktopApp) return settings;
  return {
    ...settings,
    app_mode: "desktop",
    provider_settings: {
      ...settings.provider_settings,
      provider: "ollama",
      model: settings.provider_settings.provider === "ollama" ? (settings.provider_settings.model || DEFAULT_MODEL_BY_PROVIDER.ollama) : DEFAULT_MODEL_BY_PROVIDER.ollama,
      api_key: "",
      api_base: "",
      ollama_url: settings.provider_settings.ollama_url || "http://localhost:11434"
    },
    storage_settings: {
      ...settings.storage_settings,
      backend: "local_disk",
      s3_endpoint_url: "",
      s3_bucket: "",
      s3_region: "",
      s3_access_key_id: "",
      s3_secret_access_key: ""
    }
  };
}

function selectedModelValue(provider: string, model: string) {
  const presets = MODEL_PRESETS[provider] ?? [];
  return presets.some((option) => option.value === model) ? model : CUSTOM_MODEL_VALUE;
}

async function canUseDesktopDialog() {
  try {
    const core = await import("@tauri-apps/api/core");
    return core.isTauri();
  } catch {
    return false;
  }
}

function JobProgressDetails({ job }: { job: Job }) {
  const events = job.events ?? [];
  const latest = events[events.length - 1];
  return (
    <div className="job-details">
      <div className="job-details-head">
        <div>
          <strong>{job.source_path.split("/").pop() || job.id}</strong>
          <span>{job.status} / {job.stage}</span>
        </div>
        <strong>{job.progress}%</strong>
      </div>
      <progress className="progress-bar large" value={Math.max(0, Math.min(job.progress, 100))} max={100} />
      <div className="job-stats">
        <span>Mode: {job.mode}</span>
        <span>Source: {job.source_type}</span>
        <span>Events: {events.length}</span>
        {latest?.created_at && <span>Last update: {formatDate(latest.created_at)}</span>}
      </div>
      <div className="timeline">
        {events.map((event) => (
          <div className="timeline-item" key={event.id}>
            <span className={`timeline-dot ${event.stage}`} />
            <div>
              <strong>{stageLabel(event.stage)} <em>{event.progress}%</em></strong>
              <p>{event.message || stageLabel(event.stage)}</p>
              {event.details && Object.keys(event.details).length > 0 && (
                <small>{formatDetails(event.details)}</small>
              )}
              {event.created_at && <time>{formatDate(event.created_at)}</time>}
            </div>
          </div>
        ))}
        {!events.length && <p className="muted">Detailed progress will appear after the job starts.</p>}
      </div>
    </div>
  );
}

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

function LiveExtractionPanel({ data }: { data: Partial<LiveExtractionData> }) {
  const { thumbnails = [], videoMetadata = {}, transcript = "", transcriptPreview = "", analysisPreview = "", visionWarning = "" } = data;
  const transcriptText = transcript || transcriptPreview;
  const hasContent = thumbnails.length > 0 || transcriptText || analysisPreview || Object.keys(videoMetadata).length > 0;
  if (!hasContent) return null;
  return (
    <div className="live-extraction">
      {Object.keys(videoMetadata).length > 0 && (
        <div className="live-meta-badges">
          {videoMetadata.duration_seconds != null && <span className="live-badge">{formatDuration(videoMetadata.duration_seconds)}</span>}
          {videoMetadata.width != null && <span className="live-badge">{videoMetadata.width}×{videoMetadata.height}</span>}
          {videoMetadata.aspect_ratio && <span className="live-badge">{videoMetadata.aspect_ratio}</span>}
          {videoMetadata.orientation && <span className="live-badge">{videoMetadata.orientation}</span>}
          {videoMetadata.fps != null && <span className="live-badge">{Math.round(videoMetadata.fps)} fps</span>}
        </div>
      )}
      {visionWarning && (
        <div className="live-warning">
          <AlertTriangle size={14} />
          <span>{visionWarning}</span>
        </div>
      )}
      {thumbnails.length > 0 && (
        <div className="live-section">
          <div className="live-section-head">
            <Video size={13} />
            <strong>Extracted Frames</strong>
            <em>{thumbnails.length} frames</em>
          </div>
          <div className="live-frames">
            {thumbnails.map((thumb, i) => (
              <img
                key={i}
                src={`data:image/jpeg;base64,${thumb}`}
                alt={`Frame ${i + 1}`}
                className="live-frame-thumb"
                title={`Frame ${i + 1}`}
              />
            ))}
          </div>
        </div>
      )}
      {transcriptText && (
        <div className="live-section">
          <div className="live-section-head">
            <History size={13} />
            <strong>Transcript</strong>
            <div style={{ display: 'flex', gap: 6 }}>
              <button className="mini-button" onClick={() => navigator.clipboard.writeText(transcriptText)}>Copy</button>
            </div>
          </div>
          <div className="live-text">{transcriptText}</div>
        </div>
      )}
      {analysisPreview && (
        <div className="live-section">
          <div className="live-section-head">
            <CheckCircle2 size={13} />
            <strong>Visual Analysis</strong>
            <div style={{ display: 'flex', gap: 6 }}>
              <button className="mini-button" onClick={() => navigator.clipboard.writeText(analysisPreview)}>Copy</button>
            </div>
          </div>
          <div className="live-text">{analysisPreview}</div>
        </div>
      )}
    </div>
  );
}

function RetryPanel({ job, onRetry, localOnly = false }: { job: Job; onRetry: (ps?: ProviderSettings) => void; localOnly?: boolean }) {
  const orig = job.request?.provider_settings;
  const [provider, setProvider] = React.useState(localOnly ? "ollama" : (orig?.provider ?? "ollama"));
  const [model, setModel] = React.useState(orig?.model ?? DEFAULT_MODEL_BY_PROVIDER.ollama);
  const [apiKey, setApiKey] = React.useState(orig?.api_key ?? "");
  const [apiBase, setApiBase] = React.useState(orig?.api_base ?? "");
  const [ollamaUrl, setOllamaUrl] = React.useState(orig?.ollama_url ?? "http://localhost:11434");
  const [showKey, setShowKey] = React.useState(false);

  function handleProviderChange(next: string) {
    if (localOnly) return;
    const saved: Record<string, string> = (() => {
      try { return JSON.parse(localStorage.getItem("vidmeta_provider_keys") ?? "{}"); } catch { return {}; }
    })();
    saved[provider] = apiKey;
    localStorage.setItem("vidmeta_provider_keys", JSON.stringify(saved));
    setProvider(next);
    setModel(DEFAULT_MODEL_BY_PROVIDER[next] ?? "");
    setApiKey(saved[next] ?? "");
    setShowKey(false);
  }

  const modelValue = selectedModelValue(provider, model);

  function buildSettings(): ProviderSettings {
    if (localOnly) {
      return { provider: "ollama", model, api_key: "", api_base: "", ollama_url: ollamaUrl };
    }
    return { provider, model, api_key: apiKey, api_base: apiBase, ollama_url: ollamaUrl };
  }

  return (
    <div className="retry-panel">
      <div className="retry-panel-head">
        <RotateCcw size={15} />
        <strong>Retry with different settings</strong>
      </div>
      <div className="retry-fields">
        {localOnly ? (
          <p className="mode-note">Desktop retries are locked to local Ollama.</p>
        ) : (
          <select aria-label="Retry provider" value={provider} onChange={(e) => handleProviderChange(e.target.value)}>
            <option value="ollama">Ollama local</option>
            <option value="openrouter">OpenRouter</option>
            <option value="openai">OpenAI</option>
            <option value="anthropic">Anthropic</option>
            <option value="gemini">Gemini</option>
            <option value="nvidia">NVIDIA NIM</option>
            <option value="groq">Groq</option>
          </select>
        )}
        <select aria-label="Retry model" value={modelValue} onChange={(e) => { if (e.target.value === CUSTOM_MODEL_VALUE) { setModel(""); } else { setModel(e.target.value); } }}>
          {(MODEL_PRESETS[provider] ?? []).map((m) => (
            <option value={m.value} key={m.value}>{m.label}{m.note ? ` - ${m.note}` : ""}</option>
          ))}
          <option value={CUSTOM_MODEL_VALUE}>Other custom model</option>
        </select>
        {modelValue === CUSTOM_MODEL_VALUE && (
          <input value={model} onChange={(e) => setModel(e.target.value)} placeholder="Enter model ID" />
        )}
        {provider === "ollama" ? (
          <input value={ollamaUrl} onChange={(e) => setOllamaUrl(e.target.value)} placeholder="Ollama URL" />
        ) : (
          !localOnly && <>
            <div className="key-input-wrap">
              <input value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder="API key" type={showKey ? "text" : "password"} />
              <button type="button" className="key-toggle" onClick={() => setShowKey((v) => !v)} aria-label={showKey ? "Hide API key" : "Show API key"}>
                {showKey ? <EyeOff size={14} /> : <Eye size={14} />}
              </button>
            </div>
            {["openrouter", "openai"].includes(provider) && (
              <input value={apiBase} onChange={(e) => setApiBase(e.target.value)} placeholder="API base URL (optional)" />
            )}
          </>
        )}
      </div>
      <div className="retry-actions">
        <button type="button" className="secondary" onClick={() => onRetry()}>
          <RotateCcw size={14} /> Retry same settings
        </button>
        <button type="button" className="primary" onClick={() => onRetry(buildSettings())}>
          <RotateCcw size={14} /> Retry with above settings
        </button>
      </div>
    </div>
  );
}

function MetadataView({ metadata, analysis, transcript }: { metadata: Record<string, unknown>; analysis: string; transcript: string }) {
  const batchResults = Array.isArray(metadata.batch_results) ? metadata.batch_results : null;
  const platforms = platformEntries(metadata);
  return (
    <div className="metadata">
      <div className="summary">
        <strong>{String(metadata.video_summary ?? "Generated metadata")}</strong>
        <span>{String(metadata.content_category ?? "")}{!batchResults ? ` / ${platforms.length} platform profiles` : ""}</span>
      </div>
      {batchResults && (
        <div className="batch-list">
          {batchResults.map((item, index) => {
            const row = item as { file?: string; metadata?: Record<string, unknown> };
            const rowPlatforms = row.metadata ? platformEntries(row.metadata) : [];
            return (
              <article className="platform-card" key={`${row.file}-${index}`}>
                <h3>{row.file ?? `Video ${index + 1}`}</h3>
                <p className="muted">{String(row.metadata?.video_summary ?? "Metadata generated")}</p>
                <small>{rowPlatforms.length} platform profiles generated</small>
              </article>
            );
          })}
        </div>
      )}
      {!batchResults && <div className="platform-grid">
        {platforms.map(({ key, label, data }) => {
          return (
            <article className="platform-card" key={key}>
              <div className="platform-card-head">
                <h3>{label}</h3>
                <PlatformDownloadButtons platformKey={key} label={label} data={data} />
              </div>
              <label>Title<textarea readOnly value={String(data.title ?? "")} /></label>
              <label>Description<textarea readOnly value={String(data.description ?? "")} /></label>
              <label>Hashtags<textarea readOnly value={Array.isArray(data.hashtags) ? data.hashtags.join(" ") : String(data.hashtags ?? "")} /></label>
              <label>Keywords<textarea readOnly value={Array.isArray(data.keywords) ? data.keywords.join(", ") : String(data.keywords ?? "")} /></label>
              <label>CTA<textarea readOnly value={String(data.cta ?? "")} /></label>
              <label>Posting tip<textarea readOnly value={String(data.posting_tip ?? "")} /></label>
            </article>
          );
        })}
      </div>}
      <details>
        <summary>Analysis and transcript</summary>
        <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
          <button className="mini-button" onClick={() => navigator.clipboard.writeText(analysis)}>Copy Analysis</button>
          <button className="mini-button" onClick={() => navigator.clipboard.writeText(transcript)}>Copy Transcript</button>
        </div>
        <pre>{analysis}</pre>
        <pre>{transcript}</pre>
      </details>
    </div>
  );
}

function PlatformDownloadButtons({
  platformKey,
  label,
  data
}: {
  platformKey: string;
  label: string;
  data: Record<string, unknown>;
}) {
  return (
    <div className="platform-downloads" aria-label={`${label} downloads`}>
      {(["json", "csv", "txt"] as const).map((format) => (
        <button
          type="button"
          key={format}
          onClick={() => downloadPlatform(platformKey, label, data, format)}
          title={`Download ${label} ${format.toUpperCase()}`}
          aria-label={`Download ${label} ${format.toUpperCase()}`}
        >
          <Download size={13} />
          {format.toUpperCase()}
        </button>
      ))}
    </div>
  );
}

function platformEntries(metadata: Record<string, unknown>) {
  const source = isRecord(metadata.platforms) ? metadata.platforms : metadata;
  const entries: { key: string; label: string; data: Record<string, unknown> }[] = [];
  const seen = new Set<string>();
  for (const platform of SOCIAL_PLATFORMS) {
    const value = source[platform.key];
    if (isRecord(value)) {
      entries.push({ key: platform.key, label: platform.label, data: value });
      seen.add(platform.key);
    }
  }
  for (const [key, value] of Object.entries(source)) {
    if (!seen.has(key) && isRecord(value) && looksLikePlatformMetadata(value)) {
      entries.push({ key, label: platformLabel(key), data: value });
    }
  }
  return entries;
}

function looksLikePlatformMetadata(value: Record<string, unknown>) {
  return ["title", "description", "hashtags", "keywords", "cta", "posting_tip"].some((key) => key in value);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function platformLabel(key: string) {
  return PLATFORM_LABELS[key] ?? titleize(key);
}

function downloadPlatform(
  platformKey: string,
  label: string,
  data: Record<string, unknown>,
  format: "json" | "csv" | "txt"
) {
  const filename = `vidmeta-${slugify(label || platformKey)}.${format}`;
  const content = platformExportContent(platformKey, label, data, format);
  const type = format === "json" ? "application/json" : format === "csv" ? "text/csv" : "text/plain";
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function platformExportContent(
  platformKey: string,
  label: string,
  data: Record<string, unknown>,
  format: "json" | "csv" | "txt"
) {
  if (format === "json") {
    return JSON.stringify({ platform: platformKey, label, ...data }, null, 2);
  }
  if (format === "csv") {
    const row = [
      label,
      data.title ?? "",
      data.description ?? "",
      Array.isArray(data.hashtags) ? data.hashtags.join(" ") : data.hashtags ?? "",
      Array.isArray(data.keywords) ? data.keywords.join(", ") : data.keywords ?? "",
      data.cta ?? "",
      data.posting_tip ?? ""
    ];
    return `Platform,Title,Description,Hashtags,Keywords,CTA,Posting Tip\n${row.map(csvCell).join(",")}\n`;
  }
  const hashtags = Array.isArray(data.hashtags) ? data.hashtags.join(" ") : String(data.hashtags ?? "");
  const keywords = Array.isArray(data.keywords) ? data.keywords.join(", ") : String(data.keywords ?? "");
  return [
    label,
    "",
    `TITLE:\n${String(data.title ?? "")}`,
    `DESCRIPTION:\n${String(data.description ?? "")}`,
    `HASHTAGS:\n${hashtags}`,
    `KEYWORDS:\n${keywords}`,
    `CTA:\n${String(data.cta ?? "")}`,
    `POSTING TIP:\n${String(data.posting_tip ?? "")}`
  ].join("\n\n");
}

function csvCell(value: unknown) {
  return `"${String(value).replace(/"/g, '""')}"`;
}

function slugify(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "platform";
}

function stageLabel(stage: string) {
  return titleize(stage.replace(/^batch_/, "Batch "));
}

function titleize(value: string) {
  return value.replace(/_/g, " ").replace(/\b\w/g, (char: string) => char.toUpperCase());
}

const DETAIL_SKIP = new Set(["thumbnails", "video_metadata", "transcript", "transcript_preview", "analysis_preview"]);

function formatDetails(details: Record<string, unknown>) {
  return Object.entries(details)
    .filter(([key]) => !DETAIL_SKIP.has(key))
    .map(([key, value]) => `${stageLabel(key)}: ${String(value)}`)
    .join(" / ");
}

function formatDate(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

async function uploadResumable(url: string, file: File, onProgress: (progress: number) => void) {
  const chunkSize = 8 * 1024 * 1024;

  // HEAD to check current offset — enables actual resume of interrupted uploads.
  let offset = 0;
  const headResponse = await fetch(url, {
    method: "HEAD",
    headers: { "Tus-Resumable": "1.0.0" }
  });
  if (headResponse.ok) {
    const serverOffset = Number(headResponse.headers.get("Upload-Offset") ?? "0");
    if (!Number.isNaN(serverOffset) && serverOffset > 0 && serverOffset < file.size) {
      offset = serverOffset;
    }
  }

  while (offset < file.size) {
    const chunk = file.slice(offset, offset + chunkSize);
    const response = await fetch(url, {
      method: "PATCH",
      headers: {
        "Content-Type": "application/offset+octet-stream",
        "Tus-Resumable": "1.0.0",
        "Upload-Offset": String(offset)
      },
      body: chunk
    });
    if (!response.ok) {
      throw new Error(await response.text());
    }
    offset = Number(response.headers.get("Upload-Offset") ?? offset + chunk.size);
    onProgress(Math.round((offset / file.size) * 100));
  }
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
