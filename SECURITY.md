# Security Policy

## Supported Versions

Only the latest release is actively maintained with security fixes.

| Version | Supported |
|---------|-----------|
| 1.4.x (latest) | ✅ |
| < 1.4.0 | ❌ |

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

To report a vulnerability privately:

1. Go to the [Security tab](https://github.com/GitDonkeyHubbed/qwen3vl-captioner/security) of this repository.
2. Click **"Report a vulnerability"** to open a private advisory draft.
3. Describe the issue, steps to reproduce, and any potential impact.

You can expect an acknowledgment within **72 hours** and a fix or workaround within **14 days** for confirmed issues.

## Scope

This is a local desktop application: it does not run a server or expose ports. It does, however, fetch and parse data it did not produce, and reports about any of it are in scope:

- **Model downloads** — GGUF/safetensors files and vision encoders pulled from HuggingFace over HTTPS, written to disk and then loaded by llama.cpp / mlx-vlm.
- **Installer fetches** — `setup.bat` / `setup.sh` download the uv installer and prebuilt llama-cpp-python wheels from pinned URLs.
- **User media** — images the user imports are decoded by Qt and Pillow, and `.txt` caption sidecars are read from disk.
- **The dependency chain** — Python packages, scanned regularly and patched promptly.

Path traversal in downloaded filenames, unsafe deserialization, decoder crashes on crafted images, and TLS/verification weaknesses in any of the above are all things we want to hear about.
