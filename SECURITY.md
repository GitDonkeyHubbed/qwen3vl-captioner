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

This is a local desktop application — it does not run a server, expose ports, or process untrusted remote input by design. The most relevant security surface is the dependency chain (Python packages). Dependency vulnerabilities are scanned regularly and patched promptly.
