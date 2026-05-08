# Windows Trusted Installation

This page is for Windows users who receive a private VidMeta AI installer and a publisher certificate from the project maintainer.

## Important Key Rule

Users should receive only a public certificate file, usually `.cer`.

Do not give users the private signing key or `.pfx` file. The private key is only for the maintainer or CI system that signs the installer.

## What Users Need

- `VidMeta-AI-Setup.exe` or `VidMeta-AI.msi`
- Publisher certificate, for example `VidMeta-AI-Publisher.cer`
- Optional private root certificate, for example `VidMeta-AI-Root.cer`, only if the publisher certificate chains to an internal CA

## Current User Trust Install

Use this when the user installs only for their Windows account.

Open PowerShell, go to the folder containing the certificate, then run:

```powershell
Import-Certificate -FilePath .\VidMeta-AI-Publisher.cer -CertStoreLocation Cert:\CurrentUser\TrustedPublisher
```

If a private root certificate is also provided:

```powershell
Import-Certificate -FilePath .\VidMeta-AI-Root.cer -CertStoreLocation Cert:\CurrentUser\Root
```

Then run the VidMeta AI installer.

## All Users Trust Install

Use this when an administrator wants the certificate trusted for every Windows account on the machine.

Open PowerShell as Administrator:

```powershell
Import-Certificate -FilePath .\VidMeta-AI-Publisher.cer -CertStoreLocation Cert:\LocalMachine\TrustedPublisher
```

If a private root certificate is also provided:

```powershell
Import-Certificate -FilePath .\VidMeta-AI-Root.cer -CertStoreLocation Cert:\LocalMachine\Root
```

Then run the VidMeta AI installer.

## Verify The Installer Signature

Before installing, users can verify that the installer is signed:

```powershell
Get-AuthenticodeSignature .\VidMeta-AI-Setup.exe | Format-List
```

Expected result:

- `Status` should be `Valid`.
- `SignerCertificate.Subject` should match the VidMeta AI publisher identity.

For MSI builds:

```powershell
Get-AuthenticodeSignature .\VidMeta-AI.msi | Format-List
```

## Verify The Trusted Certificate

Current user store:

```powershell
Get-ChildItem Cert:\CurrentUser\TrustedPublisher | Where-Object Subject -like "*VidMeta*"
```

All users store:

```powershell
Get-ChildItem Cert:\LocalMachine\TrustedPublisher | Where-Object Subject -like "*VidMeta*"
```

## Maintainer Signing Notes

The maintainer signs the Windows installer with a real code signing certificate. Tauri supports Windows installer signing through the `bundle.windows` signing settings, including `certificateThumbprint`, `digestAlgorithm`, and `timestampUrl`.

Private build secrets belong only in the maintainer machine or CI secrets:

- `WINDOWS_CERTIFICATE`
- `WINDOWS_CERTIFICATE_PASSWORD`
- certificate thumbprint
- timestamp URL

## Build Installer Artifacts

Run these commands from `desktop/` on the target operating system:

```bash
npm run build:windows
```

Windows output:

- NSIS setup executable: `src-tauri/target/release/bundle/nsis/`
- MSI installer: `src-tauri/target/release/bundle/msi/`

The NSIS installer is configured with `installMode: "both"`, so Windows can offer current-user or all-users installation depending on privileges.

## Public Release Recommendation

For public GitHub releases, use a public CA-issued code signing certificate. Installing a private certificate is acceptable for internal testing, but it is not the best public distribution experience.
