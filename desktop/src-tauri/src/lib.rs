use std::sync::Mutex;
use tauri::{Manager, RunEvent};
use tauri_plugin_shell::process::CommandChild;
use tauri_plugin_shell::ShellExt;

pub struct SidecarHandle(Mutex<Option<CommandChild>>);

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            match app.shell().sidecar("vidmeta-server") {
                Ok(cmd) => match cmd.spawn() {
                    Ok((mut rx, child)) => {
                        app.manage(SidecarHandle(Mutex::new(Some(child))));
                        tauri::async_runtime::spawn(async move {
                            use tauri_plugin_shell::process::CommandEvent;
                            while let Some(event) = rx.recv().await {
                                match event {
                                    CommandEvent::Stderr(line) => {
                                        eprintln!("[backend] {}", String::from_utf8_lossy(&line));
                                    }
                                    CommandEvent::Terminated(_) => break,
                                    _ => {}
                                }
                            }
                        });
                    }
                    Err(e) => eprintln!("[vidmeta] Failed to spawn sidecar: {e}"),
                },
                Err(e) => eprintln!("[vidmeta] Sidecar not found (dev mode?): {e}"),
            }
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building VidMeta AI")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                if let Some(handle) = app.try_state::<SidecarHandle>() {
                    if let Ok(mut guard) = handle.0.lock() {
                        if let Some(child) = guard.take() {
                            let _ = child.kill();
                        }
                    }
                }
            }
        });
}
