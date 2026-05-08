# VidMeta AI Desktop

This is the Tauri shell for the React dashboard. It gives the app native file and folder pickers so local videos can be processed by path instead of uploaded through the browser.

Current v1 desktop flow:

1. Start the FastAPI service:

   ```bash
   vidmeta serve
   ```

2. Start the web frontend or Tauri dev shell:

   ```bash
   cd web && npm run dev
   cd ../desktop && npm run dev
   ```

The backend is treated as a required local service for this implementation. Packaging it as a bundled sidecar is the next desktop packaging step.

## Installer Builds

Use platform-specific build scripts for installable artifacts:

```bash
npm run build:mac
npm run build:windows
npm run build:linux
```

- Windows builds create NSIS/MSI installers. NSIS is configured to offer current-user or all-users installation.
- macOS builds create an app bundle and DMG.
- Linux builds create Debian, RPM, and AppImage packages.

Windows trusted-install instructions are in `../docs/WINDOWS_TRUSTED_INSTALLATION.md`.
