# Mimic 3 — Hyper-Optimized & Asynchronous Edition

[![Fork Status](https://shields.io)](#-what-makes-this-fork-better)
[![License: AGPL v3](https://shields.io)](LICENSE)

> [!NOTE]
> This is an actively maintained, hyper-optimized fork of Mycroft Mimic 3. While the original upstream repository is no longer maintained by Mycroft AI, this fork rewrites the core execution architecture for high-performance automation, massive batch processing, and thread-safe pipeline execution.

---

##  What Makes This Fork Better?

The original Mimic 3 CLI processed text strictly line-by-line, causing significant latency and blocking bottlenecks. This fork introduces a production-ready asynchronous framework:

*   **Ultra-Fast Asynchronous Engine:** Utilizes an `asyncio` pipeline paired with a `ThreadPoolExecutor` and concurrency-controlled semaphores to synthesize large batches of text simultaneously.
*   **Smart I/O Performance:** Offloads heavy disk saving operations (`.wav` generation) to background worker threads, completely eliminating terminal and file-system write lag.
*   **Fail-Safe Architecture:** Hardened with a bulletproof context manager lifecycle (`__aenter__` / `__aexit__`) to clean up background thread pools instantly on cancellation (`Ctrl+C`), preventing zombie processes.
*   **Advanced Automation Features:** Added dynamic CSV data mappings (`id|voice|text`), multi-player interactive playback loops, and intelligent paragraph grouping via the `--process-on-blank-line` flag.

---

## 🛠️ Quickstart

### 1. Installation & Setup
Install system speech dependencies and clone this fork to a virtual environment:

```sh
# Install core system voice libraries
sudo apt-get install libespeak-ng1

# Create and enter your virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip3 install --upgrade pip

# Install package dependencies (including new async requirements)
pip3 install -e .
pip3 install aiohttp tqdm
```

### 2. High-Performance CLI Usage

#### Bulk Batch Synthesis (Fastest)
Pass large text documents or lines directly via standard input. Your engine automatically distributes workers concurrently across your processor cores:
```sh
cat long_story.txt | mimic3 --output-dir ./output_wavs --output-naming text
```

#### Process Paragraph-by-Paragraph
Assemble split line buffers seamlessly by telling the engine to execute breaks only on structural blank lines:
```sh
cat transcript.txt | mimic3 --process-on-blank-line --output-dir ./processed_audio
```

#### Flexible Dynamic CSV Pipelines
Feed structured datasets directly to map individual lines to unique voice modules and file IDs instantly:
```sh
# Format: id|voice|text
cat database.csv | mimic3 --csv --csv-voice --output-naming id --output-dir ./export
```

#### List Available System Voices
Quickly audit your underlying engine profiles without messy terminal dictionary dumps:
```sh
mimic3 --voices
```

---

##  Docker Deployment

The companion container system has been completely hardened with automated initialization handlers and host-bridge access points.

```sh
# Set up persistent cache volumes
mkdir -p "\${HOME}/.local/share/mycroft/mimic3"
chmod a+rwx "\${HOME}/.local/share/mycroft/mimic3"

# Run the hyper-optimized container runner script
./docker/mimic3 "Hello world from an optimized container." | aplay
```

> [!TIP]
> The included `./docker/mimic3` script is pre-patched with `--init` to guarantee immediate thread termination on `Ctrl+C`, alongside `--network host` for frictionless remote API streaming.

---

##  Web Server & Remote Engine

For repeated, extreme-throughput automation loops, pair your client with a localized server container instance:

```sh
# Spin up the background speech node
docker run -d -p 59125:59125 v "\${HOME}/.local/share/mycroft/mimic3:/home/mimic3/.local/share/mycroft/mimic3" mycroftai/mimic3
```

Point your newly optimized asynchronous client directly at the network endpoint using the remote flag to leverage immediate network-level concurrency:
```sh
cat massive_text_dump.txt | mimic3 --remote http://localhost:59125 --output-dir ./streamed_audio
```

---

##  License
This program is distributed as free software under the terms of the **GNU Affero General Public License (AGPLv3)**. See the `LICENSE` file for deep copyleft compliance details.
