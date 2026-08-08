#!/usr/bin/env python3
"""
Mimic3 TTS CLI — Hyper-Optimized Edition
Copyright (C) 2024 CAT Industries
License: MIT
"""

import asyncio
import argparse
import csv
import io
import logging
import os
import shlex
import shutil
import signal
import string
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import AsyncIterator, Optional, List, Dict, Any, Union
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

import aiohttp
import numpy as np
from tqdm import tqdm

# Local imports (assumes mimic3_tts is installed)
from mimic3_tts import Mimic3Settings, Mimic3TextToSpeechSystem, Voice, AudioResult, MarkResult

# -----------------------------------------------------------------------------
# Constants & Configuration
# -----------------------------------------------------------------------------

DEFAULT_PLAY_PROGRAMS = ["paplay", "play -q", "aplay -q"]
CHUNK_SIZE = 4096  # PCM chunk size for streaming
MAX_CONCURRENT_REQUESTS = 10  # For remote TTS
RETRY_COUNT = 3
RETRY_DELAY = 0.5

# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------

class OutputNaming(str, Enum):
    TEXT = "text"
    TIME = "time"
    ID = "id"

class StdinFormat(str, Enum):
    AUTO = "auto"
    LINES = "lines"
    DOCUMENT = "document"

@dataclass
class SynthesisJob:
    text: str
    voice: Optional[str] = None
    speaker: Optional[str] = None
    line_id: str = ""
    is_ssml: bool = False
    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_w: float = 0.8

@dataclass
class SynthesisResult:
    audio_bytes: bytes
    sample_rate_hz: int
    sample_width_bytes: int
    num_channels: int
    text: str
    line_id: str
    marks: List[str] = field(default_factory=list)

# -----------------------------------------------------------------------------
# Core TTS Engine (Async + Batched)
# -----------------------------------------------------------------------------

class OptimizedTTS:
    """Async TTS engine with batched synthesis and native audio output"""
    
    def __init__(
        self,
        voice: Optional[str] = None,
        speaker: Optional[str] = None,
        voices_dir: Optional[List[Path]] = None,
        use_cuda: bool = False,
        deterministic: bool = False,
        remote_url: Optional[str] = None,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
        max_concurrent: int = MAX_CONCURRENT_REQUESTS,
    ):
        self.voice = voice
        self.speaker = speaker
        self.use_cuda = use_cuda
        self.deterministic = deterministic
        self.remote_url = remote_url
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w = noise_w
        self.max_concurrent = max_concurrent
        
        self._local_tts = None
        self._session = None
        self._executor = None  # Initialized in __aenter__ to prevent leaks
        
        if not remote_url:
            # Initialize local TTS
            settings = Mimic3Settings(
                length_scale=length_scale,
                noise_scale=noise_scale,
                noise_w=noise_w,
                voices_directories=[str(d) for d in (voices_dir or [])],
                use_cuda=use_cuda,
                use_deterministic_compute=deterministic,
            )
            self._local_tts = Mimic3TextToSpeechSystem(settings)
            if voice:
                self._local_tts.voice = voice
            if speaker:
                self._local_tts.speaker = speaker
    
    async def __aenter__(self):
        self._executor = ThreadPoolExecutor(max_workers=8)
        if self.remote_url:
            self._session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
        if self._local_tts:
            self._local_tts.shutdown()
        if self._executor:
            # wait=False ensures we don't hang indefinitely if a thread crashed
            self._executor.shutdown(wait=False)
    
    async def synthesize_batch(self, jobs: List[SynthesisJob]) -> AsyncIterator[SynthesisResult]:
        """Synthesize multiple jobs with concurrent execution"""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_job(job: SynthesisJob):
            async with semaphore:
                return await self._synthesize_single(job)
        
        # Use asyncio.gather with a semaphore for concurrency control
        tasks = [process_job(job) for job in jobs]
        for coro in asyncio.as_completed(tasks):
            yield await coro
    
    async def _synthesize_single(self, job: SynthesisJob) -> SynthesisResult:
        """Synthesize a single text string"""
        if self.remote_url:
            return await self._synthesize_remote(job)
        else:
            return await self._synthesize_local(job)
    
    async def _synthesize_local(self, job: SynthesisJob) -> SynthesisResult:
        """Local synthesis using mimic3_tts"""
        loop = asyncio.get_event_loop()
        
        def sync_synthesize():
            assert self._local_tts is not None
            
            # Set voice/speaker if provided
            if job.voice:
                self._local_tts.voice = job.voice
            if job.speaker:
                self._local_tts.speaker = job.speaker
            
            results = []
            marks = []
            
            if job.is_ssml:
                from mimic3_tts import SSMLSpeaker
                results = SSMLSpeaker(self._local_tts).speak(job.text)
            else:
                self._local_tts.begin_utterance()
                self._local_tts.speak_text(job.text)
                results = self._local_tts.end_utterance()
            
            # Extract marks and combine audio
            combined_audio = b""
            sample_rate = 22050
            sample_width = 2
            num_channels = 1
            
            for result in results:
                if isinstance(result, AudioResult):
                    combined_audio += result.audio_bytes
                    sample_rate = result.sample_rate_hz
                    sample_width = result.sample_width_bytes
                    num_channels = result.num_channels
                elif isinstance(result, MarkResult):
                    marks.append(result.name)
            
            # Restore default voice/speaker
            if job.voice and self.voice:
                self._local_tts.voice = self.voice
            if job.speaker and self.speaker:
                self._local_tts.speaker = self.speaker
            
            return SynthesisResult(
                audio_bytes=combined_audio,
                sample_rate_hz=sample_rate,
                sample_width_bytes=sample_width,
                num_channels=num_channels,
                text=job.text,
                line_id=job.line_id,
                marks=marks,
            )
        
        return await loop.run_in_executor(self._executor, sync_synthesize)
    
    async def _synthesize_remote(self, job: SynthesisJob) -> SynthesisResult:
        """Remote synthesis via HTTP API with retries"""
        assert self._session is not None
        
        url = f"{self.remote_url}/api/tts"
        params = {}
        
        if job.voice:
            params["voice"] = job.voice
        elif self.voice:
            params["voice"] = self.voice
            if self.speaker:
                params["voice"] = f"{self.voice}#{self.speaker}"
        
        if self.length_scale:
            params["lengthScale"] = str(self.length_scale)
        if self.noise_scale:
            params["noiseScale"] = str(self.noise_scale)
        if self.noise_w:
            params["noiseW"] = str(self.noise_w)
        
        headers = {
            "Content-Type": "application/ssml+xml" if job.is_ssml else "text/plain"
        }
        
        for attempt in range(RETRY_COUNT):
            try:
                async with self._session.post(url, params=params, headers=headers, data=job.text) as resp:
                    resp.raise_for_status()
                    wav_bytes = await resp.read()
                    break
            except Exception as e:
                if attempt == RETRY_COUNT - 1:
                    raise
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
        
        # Parse WAV bytes into AudioResult
        with io.BytesIO(wav_bytes) as wav_io:
            with wave.open(wav_io, "rb") as wav_file:
                return SynthesisResult(
                    audio_bytes=wav_file.readframes(wav_file.getnframes()),
                    sample_rate_hz=wav_file.getframerate(),
                    sample_width_bytes=wav_file.getsampwidth(),
                    num_channels=wav_file.getnchannels(),
                    text=job.text,
                    line_id=job.line_id,
                    marks=[],
                )

# -----------------------------------------------------------------------------
# Audio Output Utilities
# -----------------------------------------------------------------------------

async def play_audio(result: SynthesisResult, play_programs: List[str]):
    """Play audio bytes using system audio player"""
    with tempfile.NamedTemporaryFile(mode="wb+", suffix=".wav") as wav_file:
        # Write WAV header + PCM data
        with io.BytesIO() as wav_io:
            with wave.open(wav_io, "wb") as wf:
                wf.setframerate(result.sample_rate_hz)
                wf.setsampwidth(result.sample_width_bytes)
                wf.setnchannels(result.num_channels)
                wf.writeframes(result.audio_bytes)
            wav_file.write(wav_io.getvalue())
            wav_file.flush()
        
        for play_program in reversed(play_programs):
            play_cmd = shlex.split(play_program)
            if not shutil.which(play_cmd[0]):
                continue
            play_cmd.append(wav_file.name)
            subprocess.check_output(play_cmd)
            break

def write_wav(output_path: Path, result: SynthesisResult):
    """Write WAV file to disk"""
    with output_path.open("wb") as f:
        with wave.open(f, "wb") as wf:
            wf.setframerate(result.sample_rate_hz)
            wf.setsampwidth(result.sample_width_bytes)
            wf.setnchannels(result.num_channels)
            wf.writeframes(result.audio_bytes)

# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------

async def main_async(args: argparse.Namespace):
    """Async main entry point"""
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("mimic3_cli")
    
    # Build job list
    jobs = []
    
    if args.text:
        texts = args.text
    else:
        stdin_format = StdinFormat.LINES
        if args.stdin_format == StdinFormat.AUTO and args.ssml:
            stdin_format = StdinFormat.DOCUMENT
        
        if stdin_format == StdinFormat.DOCUMENT:
            texts = [sys.stdin.read()]
        else:
            texts = sys.stdin
    
    if args.process_on_blank_line:
        def process_on_blank_line(lines):
            text = ""
            for line in lines:
                line = line.strip()
                if not line:
                    if text:
                        yield text
                    text = ""
                    continue
                text += " " + line
        texts = process_on_blank_line(texts)
    
    voice = args.voice
    speaker = args.speaker
    if voice and "#" in voice and not speaker:
        voice, speaker = voice.split("#", maxsplit=1)
    
    for idx, line in enumerate(texts):
        line = line.strip()
        if not line:
            continue
        
        job = SynthesisJob(
            text=line,
            voice=voice,
            speaker=speaker,
            is_ssml=args.ssml,
            length_scale=args.length_scale or 1.0,
            noise_scale=args.noise_scale or 0.667,
            noise_w=args.noise_w or 0.8,
        )
        
        if args.output_naming == OutputNaming.ID:
            # Parse ID|text format
            with io.StringIO(line) as line_io:
                reader = csv.reader(line_io, delimiter=args.csv_delimiter)
                row = next(reader)
                job.line_id = row[0]
                job.text = row[-1]
                if args.csv_voice:
                    job.voice = row[1] if len(row) > 2 else None
        
        jobs.append(job)
    
    logger.info(f"Processing {len(jobs)} jobs...")
    
    async with OptimizedTTS(
        voice=voice,
        speaker=speaker,
        voices_dir=args.voices_dir or [],
        use_cuda=args.cuda,
        deterministic=args.deterministic,
        remote_url=args.remote,
        length_scale=args.length_scale or 1.0,
        noise_scale=args.noise_scale or 0.667,
        noise_w=args.noise_w or 0.8,
    ) as tts:
        
        # Progress bar
        pbar = tqdm(total=len(jobs), desc="Synthesizing", unit="text")
        
        # Collect all audio for combined output
        all_audio = b""
        sample_rate = 22050
        sample_width = 2
        num_channels = 1
        
        # Process results
        async for result in tts.synthesize_batch(jobs):
            pbar.update(1)
            
            # Handle marks
            if result.marks and args.mark_file:
                with open(args.mark_file, "a") as f:
                    for mark in result.marks:
                        print(mark, file=f)
            
            # Interactive playback
            if args.interactive:
                play_programs = args.play_program or DEFAULT_PLAY_PROGRAMS
                await play_audio(result, play_programs)
            
            # Write to output directory
            if args.output_dir:
                output_dir = Path(args.output_dir)
                if args.output_naming == OutputNaming.TEXT:
                    file_name = result.text.strip().replace(" ", "_")
                    file_name = file_name.translate(str.maketrans("", "", string.punctuation.replace("_", "")))
                elif args.output_naming == OutputNaming.TIME:
                    file_name = str(time.time())
                else:
                    file_name = result.line_id or str(time.time())
                output_path = output_dir / f"{file_name}.wav"
                write_wav(output_path, result)
                logger.debug(f"Wrote {output_path}")
            
            # Accumulate for combined output
            all_audio += result.audio_bytes
            sample_rate = result.sample_rate_hz
            sample_width = result.sample_width_bytes
            num_channels = result.num_channels
        
        pbar.close()
        
        # Write combined audio to stdout
        if all_audio and (args.stdout or not sys.stdout.isatty()):
            with wave.open(sys.stdout.buffer, "wb") as wf:
                wf.setframerate(sample_rate)
                wf.setsampwidth(sample_width)
                wf.setnchannels(num_channels)
                wf.writeframes(all_audio)
            sys.stdout.buffer.flush()

def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="Mimic3 TTS CLI — CAT Edition")
    parser.add_argument("text", nargs="*", help="Text to synthesize (or stdin)")
    parser.add_argument("--remote", help="Remote HTTP server URL (e.g., http://localhost:59125)")
    parser.add_argument("--stdin-format", choices=["auto", "lines", "document"], default="auto")
    parser.add_argument("--voice", "-v", help="Voice name")
    parser.add_argument("--speaker", "-s", help="Speaker name or ID")
    parser.add_argument("--voices-dir", action="append", help="Voice directory path")
    parser.add_argument("--voices", action="store_true", help="List available voices")
    parser.add_argument("--output-dir", help="Directory to save WAV files")
    parser.add_argument("--output-naming", choices=["text", "time", "id"], default="text")
    parser.add_argument("--csv", action="store_true", help="Input is CSV: id|text")
    parser.add_argument("--csv-delimiter", default="|", help="CSV delimiter")
    parser.add_argument("--csv-voice", action="store_true", help="CSV: id|voice|text")
    parser.add_argument("--mark-file", help="File to write SSML mark names")
    parser.add_argument("--interactive", action="store_true", help="Play audio immediately")
    parser.add_argument("--play-program", action="append", default=None)
    parser.add_argument("--noise-scale", type=float, help="Noise scale [0-1]")
    parser.add_argument("--length-scale", type=float, help="Length scale (1.0 = normal)")
    parser.add_argument("--noise-w", type=float, help="Cadence variation [0-1]")
    parser.add_argument("--ssml", action="store_true", help="Input text is SSML")
    parser.add_argument("--stdout", action="store_true", help="Force audio to stdout")
    parser.add_argument("--preload-voice", action="append", help="Preload voice at startup")
    parser.add_argument("--process-on-blank-line", action="store_true", help="Process on blank line")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA for ONNX")
    parser.add_argument("--deterministic", action="store_true", help="Deterministic output")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--version", action="store_true", help="Print version")
    parser.add_argument("--debug", action="store_true", help="Debug logging")
    
    args = parser.parse_args()
    
    if args.version:
        from mimic3_tts import __version__
        print(__version__)
        sys.exit(0)
        
    if args.voices:
        # Load settings and list voices, then exit
        settings = Mimic3Settings(
            voices_directories=[str(d) for d in (args.voices_dir or [])],
            use_cuda=args.cuda
        )
        tts = Mimic3TextToSpeechSystem(settings)
        voices = getattr(tts, 'voices', [])
        if not voices and hasattr(tts, 'get_voices'):
            voices = tts.get_voices()
            
        for v in voices:
            print(v.name if hasattr(v, 'name') else v)
        sys.exit(0)
    
    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Run async main
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        sys.exit(130)

if __name__ == "__main__":
    main()
