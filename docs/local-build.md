Local build result:

- VidMeta AI_2.0.0_aarch64.dmg — 161 MB ✓
- Sidecar (vidmeta-server Python backend) bundled inside .app/Contents/MacOS/ ✓
- Sidecar health check at localhost:8000 passes ✓

  ---
  GitHub secrets needed:

  ┌──────────────┬───────────────┬─────────────────────────────────────────────────────────────────┐
  │    Secret    │   Required    │                             Purpose                             │
  ├──────────────┼───────────────┼─────────────────────────────────────────────────────────────────┤  
  │ GITHUB_TOKEN │ Auto-provided │ Create draft releases — GitHub injects this automatically, no   │
  │              │               │ setup needed                                                    │  
  ├──────────────┼───────────────┼─────────────────────────────────────────────────────────────────┤  
  │ Nothing else │ —             │ Basic unsigned builds need zero secrets                         │
  └──────────────┴───────────────┴─────────────────────────────────────────────────────────────────┘  

  ---
  Optional secrets for better UX (code signing):

  Without signing, users get security warnings on first run:

  ┌──────────────────────────────┬──────────┬──────────────────────────────────────────────────────┐
  │            Secret            │ Platform │                    What it's for                     │  
  ├──────────────────────────────┼──────────┼──────────────────────────────────────────────────────┤  
  │ APPLE_CERTIFICATE            │ macOS    │ Base64 .p12 dev cert — skips "unidentified           │
  │                              │          │ developer" Gatekeeper popup                          │  
  ├──────────────────────────────┼──────────┼──────────────────────────────────────────────────────┤  
  │ APPLE_CERTIFICATE_PASSWORD   │ macOS    │ Password for the .p12                                │
  ├──────────────────────────────┼──────────┼──────────────────────────────────────────────────────┤  
  │ APPLE_ID                     │ macOS    │ Apple ID email — for notarization                    │
  ├──────────────────────────────┼──────────┼──────────────────────────────────────────────────────┤  
  │ APPLE_PASSWORD               │ macOS    │ App-specific password — for notarization             │
  ├──────────────────────────────┼──────────┼──────────────────────────────────────────────────────┤  
  │ APPLE_TEAM_ID                │ macOS    │ 10-character team ID                                 │
  ├──────────────────────────────┼──────────┼──────────────────────────────────────────────────────┤  
  │ WINDOWS_CERTIFICATE          │ Windows  │ Base64 .pfx cert — removes SmartScreen warning       │
  ├──────────────────────────────┼──────────┼──────────────────────────────────────────────────────┤  
  │ WINDOWS_CERTIFICATE_PASSWORD │ Windows  │ Password for the .pfx                                │
  └──────────────────────────────┴──────────┴──────────────────────────────────────────────────────┘  

  ---
  To trigger a release:
  git tag v2.0.0
  git push origin v2.0.0

  GitHub Actions fires, builds Mac arm64 + Mac x86_64 + Windows, and creates a draft release with all
  installers. You review the draft and publish it.  
