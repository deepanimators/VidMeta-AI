import React from "react";
import ReactDOM from "react-dom/client";
import {
  AlertTriangle,
  CheckCircle2,
  Download,
  FolderOpen,
  HardDrive,
  History,
  Loader2,
  Play,
  Save,
  Settings,
  UploadCloud,
  Video
} from "lucide-react";
import "./styles.css";

type BrandContext = {
  brand_name: string;
  brand_niche: string;
  target_audience: string;
  tone: string;
};

type VideoSettings = {
  use_whisper: boolean;
  whisper_model_size: string;
  frame_interval: number;
  max_frames: number;
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
  brand_context: BrandContext;
  video_settings: VideoSettings;
  provider_settings: ProviderSettings;
  storage_settings: StorageSettings;
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
};

type JobResult = Job & {
  transcript: string;
  analysis: string;
  metadata: Record<string, unknown>;
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: init?.body instanceof FormData ? undefined : { "Content-Type": "application/json" },
    ...init
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || response.statusText);
  }
  return response.json() as Promise<T>;
}

const defaultSettings: AppSettings = {
  app_mode: "local",
  max_upload_mb: 2048,
  brand_context: {
    brand_name: "Condenast",
    brand_niche: "Kids fashion & clothing, India",
    target_audience: "Mothers, parents, India",
    tone: "Fun & playful"
  },
  video_settings: {
    use_whisper: true,
    whisper_model_size: "base",
    frame_interval: 5,
    max_frames: 6
  },
  provider_settings: {
    provider: "ollama",
    model: "gemma4",
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
  const [uploadFile, setUploadFile] = React.useState<File | null>(null);
  const [busy, setBusy] = React.useState(false);
  const [message, setMessage] = React.useState("");
  const [uploadProgress, setUploadProgress] = React.useState(0);

  const selectedJob = React.useMemo(
    () => jobs.find((job) => job.id === selectedJobId) ?? jobs[0],
    [jobs, selectedJobId]
  );

  React.useEffect(() => {
    void refresh();
    void loadSettings();
    const timer = window.setInterval(refresh, 2500);
    return () => window.clearInterval(timer);
  }, []);

  React.useEffect(() => {
    if (selectedJob?.status === "completed") {
      void loadResult(selectedJob.id);
    }
  }, [selectedJob?.id, selectedJob?.status]);

  async function loadSettings() {
    try {
      setSettings(await api<AppSettings>("/api/settings"));
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not load settings");
    }
  }

  async function refresh() {
    try {
      const nextJobs = await api<Job[]>("/api/jobs");
      setJobs(nextJobs);
      if (!selectedJobId && nextJobs.length) {
        setSelectedJobId(nextJobs[0].id);
      }
    } catch {
      // Backend may still be starting.
    }
  }

  async function saveSettings() {
    setBusy(true);
    try {
      const saved = await api<AppSettings>("/api/settings", {
        method: "PUT",
        body: JSON.stringify(settings)
      });
      setSettings(saved);
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
    setBusy(true);
    try {
      const job = await api<Job>("/api/jobs/from-path", {
        method: "POST",
        body: JSON.stringify({
          path: path.trim(),
          mode: "single",
          brand_context: settings.brand_context,
          video_settings: settings.video_settings,
          provider_settings: settings.provider_settings
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

  async function pickWithTauri(kind: "file" | "folder") {
    try {
      const dialog = await import("@tauri-apps/plugin-dialog");
      const picked = await dialog.open({
        directory: kind === "folder",
        multiple: false,
        filters: kind === "file" ? [{ name: "Video", extensions: ["mp4", "mov", "avi", "mkv", "webm", "m4v"] }] : []
      });
      if (typeof picked === "string") setPath(picked);
    } catch {
      setMessage("Native picker is available in the Tauri desktop app.");
    }
  }

  async function uploadAndRun() {
    if (!uploadFile) {
      setMessage("Choose a video file");
      return;
    }
    setBusy(true);
    setUploadProgress(0);
    try {
      const created = await api<{ id: string; upload_url: string }>("/api/uploads/resumable", {
        method: "POST",
        body: JSON.stringify({
          filename: uploadFile.name,
          content_type: uploadFile.type || "application/octet-stream",
          size_bytes: uploadFile.size
        })
      });
      await uploadResumable(created.upload_url, uploadFile, (value) => setUploadProgress(value));
      const job = await api<Job>(`/api/jobs/from-upload/${created.id}`, {
        method: "POST",
        body: JSON.stringify({
          brand_context: settings.brand_context,
          video_settings: settings.video_settings,
          provider_settings: settings.provider_settings
        })
      });
      setSelectedJobId(job.id);
      setMessage("Upload complete and job queued");
      await refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  }

  async function loadResult(jobId: string) {
    try {
      setResult(await api<JobResult>(`/api/jobs/${jobId}/result`));
    } catch {
      setResult(null);
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

        <section className="panel compact">
          <h2><HardDrive size={18} /> Runtime</h2>
          <select value={settings.app_mode} onChange={(event) => updateSettings("app_mode", event.target.value)}>
            <option value="local">Local web</option>
            <option value="desktop">Tauri desktop</option>
            <option value="hosted">Private hosted</option>
          </select>
          {settings.app_mode === "hosted" && (
            <p className="warning"><AlertTriangle size={15} /> Hosted mode has no auth yet. Use only on trusted private networks.</p>
          )}
        </section>

        <section className="panel compact">
          <h2><Settings size={18} /> Provider</h2>
          <select
            value={settings.provider_settings.provider}
            onChange={(event) => updateProvider("provider", event.target.value)}
          >
            <option value="ollama">Ollama local</option>
            <option value="openrouter">OpenRouter</option>
            <option value="openai">OpenAI</option>
            <option value="anthropic">Anthropic</option>
            <option value="gemini">Gemini</option>
          </select>
          <input value={settings.provider_settings.model} onChange={(event) => updateProvider("model", event.target.value)} placeholder="Model" />
          {settings.provider_settings.provider === "ollama" ? (
            <input value={settings.provider_settings.ollama_url} onChange={(event) => updateProvider("ollama_url", event.target.value)} placeholder="Ollama URL" />
          ) : (
            <input value={settings.provider_settings.api_key} onChange={(event) => updateProvider("api_key", event.target.value)} placeholder="API key" type="password" />
          )}
        </section>

        <section className="panel compact">
          <h2><Save size={18} /> Storage</h2>
          <select
            value={settings.storage_settings.backend}
            onChange={(event) => updateStorage("backend", event.target.value)}
          >
            <option value="local_disk">Local server disk</option>
            <option value="s3_compatible">S3 compatible</option>
          </select>
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
          <button className="secondary" onClick={saveSettings} disabled={busy}><Save size={16} /> Save settings</button>
        </section>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <h2>Generate platform metadata</h2>
            <p>Use local paths in desktop/local mode for videos that should never pass through browser upload.</p>
          </div>
          <div className="status-pill">{message || "Service ready"}</div>
        </header>

        <div className="grid">
          <section className="panel main-card">
            <h2><FolderOpen size={19} /> Local path or folder</h2>
            <div className="path-row">
              <input value={path} onChange={(event) => setPath(event.target.value)} placeholder="/Users/you/Videos/product-demo.mp4" />
              <button className="icon-button" onClick={() => pickWithTauri("file")} title="Pick file in desktop app"><Video size={18} /></button>
              <button className="icon-button" onClick={() => pickWithTauri("folder")} title="Pick folder in desktop app"><FolderOpen size={18} /></button>
            </div>
            <div className="form-grid">
              <input value={settings.brand_context.brand_name} onChange={(event) => updateBrand("brand_name", event.target.value)} placeholder="Brand" />
              <input value={settings.brand_context.brand_niche} onChange={(event) => updateBrand("brand_niche", event.target.value)} placeholder="Niche" />
              <input value={settings.brand_context.target_audience} onChange={(event) => updateBrand("target_audience", event.target.value)} placeholder="Audience" />
              <input value={settings.brand_context.tone} onChange={(event) => updateBrand("tone", event.target.value)} placeholder="Tone" />
            </div>
            <div className="controls-row">
              <label>Frame interval <input type="number" min={1} max={120} value={settings.video_settings.frame_interval} onChange={(event) => updateVideo("frame_interval", Number(event.target.value))} /></label>
              <label>Max frames <input type="number" min={1} max={60} value={settings.video_settings.max_frames} onChange={(event) => updateVideo("max_frames", Number(event.target.value))} /></label>
              <label>Whisper <input type="checkbox" checked={settings.video_settings.use_whisper} onChange={(event) => updateVideo("use_whisper", event.target.checked)} /></label>
            </div>
            <button className="primary" onClick={createPathJob} disabled={busy}><Play size={17} /> Analyze local path</button>
          </section>

          <section className="panel main-card">
            <h2><UploadCloud size={19} /> Resumable browser upload</h2>
            <input className="file-input" type="file" accept=".mp4,.mov,.avi,.mkv,.webm,.m4v,video/*" onChange={(event) => setUploadFile(event.target.files?.[0] ?? null)} />
            <p className="muted">Use this when the browser and backend are on different machines. For local desktop files, path mode is faster and avoids upload limits.</p>
            {uploadProgress > 0 && <div className="progress"><span style={{ width: `${uploadProgress}%` }} /></div>}
            <button className="primary" onClick={uploadAndRun} disabled={busy || !uploadFile}><UploadCloud size={17} /> Upload and analyze</button>
          </section>
        </div>

        <section className="panel">
          <h2><History size={19} /> Jobs</h2>
          <div className="jobs">
            {jobs.map((job) => (
              <button key={job.id} className={`job-row ${selectedJob?.id === job.id ? "selected" : ""}`} onClick={() => setSelectedJobId(job.id)}>
                <span className={`dot ${job.status}`} />
                <span className="job-main">
                  <strong>{job.source_path.split("/").pop()}</strong>
                  <small>{job.status} / {job.stage}</small>
                </span>
                <span className="job-progress">{job.progress}%</span>
              </button>
            ))}
            {!jobs.length && <p className="muted">No jobs yet.</p>}
          </div>
        </section>

        {selectedJob && (
          <section className="panel result-panel">
            <div className="result-header">
              <h2>{selectedJob.status === "completed" ? <CheckCircle2 size={19} /> : <Loader2 size={19} />} Result</h2>
              <div className="downloads">
                {["json", "csv", "txt"].map((format) => (
                  <a key={format} href={`${API_BASE}/api/jobs/${selectedJob.id}/exports/${format}`}><Download size={15} /> {format.toUpperCase()}</a>
                ))}
              </div>
            </div>
            {selectedJob.status === "failed" && <p className="warning">{selectedJob.error_message}</p>}
            {result?.metadata ? <MetadataView metadata={result.metadata} analysis={result.analysis} transcript={result.transcript} /> : <p className="muted">Result appears here after the job completes.</p>}
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
  function updateStorage<K extends keyof StorageSettings>(key: K, value: StorageSettings[K]) {
    setSettings((current) => ({ ...current, storage_settings: { ...current.storage_settings, [key]: value } }));
  }
}

function MetadataView({ metadata, analysis, transcript }: { metadata: Record<string, unknown>; analysis: string; transcript: string }) {
  const platforms = ["youtube", "instagram", "facebook", "tiktok", "linkedin"];
  const batchResults = Array.isArray(metadata.batch_results) ? metadata.batch_results : null;
  return (
    <div className="metadata">
      <div className="summary">
        <strong>{String(metadata.video_summary ?? "Generated metadata")}</strong>
        <span>{String(metadata.content_category ?? "")}</span>
      </div>
      {batchResults && (
        <div className="batch-list">
          {batchResults.map((item, index) => {
            const row = item as { file?: string; metadata?: Record<string, unknown> };
            return (
              <article className="platform-card" key={`${row.file}-${index}`}>
                <h3>{row.file ?? `Video ${index + 1}`}</h3>
                <p className="muted">{String(row.metadata?.video_summary ?? "Metadata generated")}</p>
              </article>
            );
          })}
        </div>
      )}
      {!batchResults && <div className="platform-grid">
        {platforms.map((platform) => {
          const data = (metadata[platform] ?? {}) as Record<string, unknown>;
          return (
            <article className="platform-card" key={platform}>
              <h3>{platform}</h3>
              <label>Title<textarea readOnly value={String(data.title ?? "")} /></label>
              <label>Description<textarea readOnly value={String(data.description ?? "")} /></label>
              <label>Hashtags<textarea readOnly value={Array.isArray(data.hashtags) ? data.hashtags.join(" ") : String(data.hashtags ?? "")} /></label>
            </article>
          );
        })}
      </div>}
      <details>
        <summary>Analysis and transcript</summary>
        <pre>{analysis}</pre>
        <pre>{transcript}</pre>
      </details>
    </div>
  );
}

async function uploadResumable(url: string, file: File, onProgress: (progress: number) => void) {
  const chunkSize = 8 * 1024 * 1024;
  let offset = 0;
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
