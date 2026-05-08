CREATE TABLE videos (
    id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    file_name TEXT NOT NULL,
    file_size_bytes INTEGER,
    duration_seconds REAL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE analysis_runs (
    id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    frame_interval_seconds INTEGER NOT NULL,
    max_frames INTEGER NOT NULL,
    whisper_enabled INTEGER NOT NULL DEFAULT 0,
    whisper_model TEXT,
    status TEXT NOT NULL,
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT
);

CREATE TABLE transcripts (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE metadata_outputs (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
    platform TEXT NOT NULL,
    title TEXT,
    description TEXT,
    hashtags_json TEXT,
    keywords_json TEXT,
    cta TEXT,
    posting_tip TEXT,
    raw_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_analysis_runs_video_id ON analysis_runs(video_id);
CREATE INDEX idx_metadata_outputs_run_platform ON metadata_outputs(run_id, platform);
