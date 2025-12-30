# src/fatingest/parse.py

import asyncio
import base64
import clevercsv
import filetype
import gzip
import httpx
import io
import logging
import markdown as md
import math
import os
import polars as pl
import tarfile
import zipfile

from .blobstorage import BlobStorage
from bs4 import BeautifulSoup
from markdownify import MarkdownConverter
from typing import BinaryIO


MARKDOWN_CACHE_BUCKET = "markdown-cache"

logger = logging.getLogger()
storage = BlobStorage() # CALL ENSURE BUCKET !!!


async def parse(file_bytes: bytes | BinaryIO, file_name: str, use_cache: bool = True) -> list[dict]:
    """
    Initiates parsing from file bytes and name, leveraging cache unless specified otherwise.
    Supports archives, audio/video, tabular, images, documents and text formats.
    Returns a list of any file(s) that was parsed into markdown in the format:
        [
            {
                'file_name': 'document.pdf',
                'content_type': 'application/pdf',
                'markdown': '...',
                'file_key': '/ab/cd/efgh...',
            }
            ...
        ]
    """
    # Normalize bytes
    file_bytes = file_bytes.read() if isinstance(file_bytes, BinaryIO) else file_bytes
    # Get sha256-based CAS key as unique identifier and test for cache
    file_key = await storage.key(file_bytes)
    if use_cache:
        cached = await parse_from_cache(file_key)
        if cached:
            cached |= {'file_key': file_key}
            return [cached]
    # Test for archive and recursively unarchive
    unarchived = await parse_from_archive(file_bytes, file_name)
    if unarchived:
        return [
            parsed
            for file in unarchived
            for parsed in await parse(file['file_bytes'], file['file_name'], use_cache)
        ]
    # Parse non-cached, non-archive file
    parsed = await parse_to_markdown(file_bytes, file_name)
    parsed |= {'file_key': file_key}
    await parse_to_cache(parsed)
    return [parsed]


async def parse_from_archive(file_bytes: bytes, file_name: str) -> list[dict]:
    """
    Will unarchive zip, tar and gzip bytes or return empty.
    Returns a list of any files that was unarchived in the format:
        [{'file_bytes': '...', 'file_name': 'document.pdf'}]
    """
    file_ext = filetype.guess_extension(file_bytes)
    extracted = []
    if file_ext == 'zip':
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
                for name in z.namelist():
                    if not z.getinfo(name).is_dir():  # Files only
                        extracted.append({
                            'file_bytes': z.read(name),
                            'file_name': name,
                        })
        except Exception as e:
            logger.warning(f"Failed unarchiving {file_name}: {e}")
            pass
    elif file_ext == 'tar':
        try:
            with tarfile.open(fileobj=io.BytesIO(file_bytes)) as t:
                for member in t:
                    if member.isfile():  # Files only
                        f = t.extractfile(member)
                        if f:
                            extracted.append({
                                'file_bytes': f.read(),
                                'file_name': member.name,
                            })
        except Exception as e:
            logger.warning(f"Failed unarchiving {file_name}: {e}")
            pass
    elif file_ext == 'gz':
        try:
            extracted = [{
                'file_bytes': gzip.decompress(file_bytes),
                'file_name': file_name,
            }]
        except Exception as e:
            logger.warning(f"Failed unarchiving {file_name}: {e}")
            pass
    return extracted


async def parse_from_cache(file_key: str) -> dict | None:
    """
    Retrieves cached markdown from given file key.
    """
    cache = await storage.get(MARKDOWN_CACHE_BUCKET, file_key)
    if cache:
        return {
            'file_name': cache['file_name'],
            'content_type': cache['content_type'],
            'markdown': cache['file_bytes'].decode('utf-8')
        }
    return None


async def parse_to_cache(parsed: dict):
    """
    Caches parsed markdown and returns file keys.
    """
    await storage.put(
        MARKDOWN_CACHE_BUCKET,
        parsed['markdown'].encode('utf-8'),
        parsed['file_name'],
        parsed['content_type'],
        parsed.get('file_key')
    )


async def parse_to_markdown(file_bytes: bytes, file_name: str) -> dict:
    """
    Parses given file and returns markdown.
    Supports audio/video, tabular, images, documents and text formats.
    Returns a dict in the format:
        {'file_name': 'document.pdf', 'content_type': 'application/pdf', 'markdown': '...'}
    """
    file_type = filetype.guess(file_bytes)

    # Detection-based
    if file_type:

        # Audio/video
        if file_type.mime[:6] in ('audio/', 'video/'):
            markdown = await audio_to_markdown(file_bytes, file_name, file_type.mime)
            return {'file_name': file_name, 'content_type': file_type.mime, 'markdown': markdown}

        # Image
        if file_type.mime[:6] == 'image/':
            markdown = await image_to_markdown(file_bytes, file_name)
            return {'file_name': file_name, 'content_type': file_type.mime, 'markdown': markdown}

        # Spreadsheet
        if file_type.extension in ('xls', 'xlsx', 'ods'):
            markdown = await spreadsheet_to_markdown(file_bytes, file_name, file_type.extension)
            return {'file_name': file_name, 'content_type': file_type.mime, 'markdown': markdown}

        # Document
        if file_type.extension in ('pdf', 'doc', 'docx', 'ppt', 'pptx', 'odp'):
            markdown = await document_to_markdown(file_bytes, file_name, file_type.extension)
            return {'file_name': file_name, 'content_type': file_type.mime, 'markdown': markdown}
    
    # Duck-based
    else:

        # Parquet
        markdown = await parquet_to_markdown(file_bytes)
        if markdown:
            return {'file_name': file_name, 'content_type': 'application/vnd.apache.parquet', 'markdown': markdown}

        # Ensure normalized text encoding and EOL before testing for various text-based formats
        file_bytes = await normalize_text(file_bytes)

        # JSON
        markdown = await json_to_markdown(file_bytes)
        if markdown:
            return {'file_name': file_name, 'content_type': 'application/json', 'markdown': markdown}
        
        # CSV
        markdown = await csv_to_markdown(file_bytes)
        if markdown:
            return {'file_name': file_name, 'content_type': 'text/csv', 'markdown': markdown}

        # Text/markdown/HTML
        markdown = await text_to_markdown(file_bytes, file_name, 'html')
        if markdown:
            return {'file_name': file_name, 'content_type': 'text/plain', 'markdown': markdown}

    # Empty
    return {'file_name': file_name, 'content_type': 'unknown/unsupported', 'markdown': ''}


async def text_to_markdown(file_bytes: bytes, file_name: str, file_ext: str) -> str:
    """
    Attemps to open file and normalize any html or markdown mix to clean markdown.
    """
    try: # Text-based?
        text = file_bytes.decode('utf-8')
        text = md.markdown(text)
        markdown = await document_to_markdown(text.encode('utf-8'), file_name, file_ext)
    except:
        return ""
    return markdown


async def spreadsheet_to_markdown(file_bytes: bytes, file_name: str, file_ext: str) -> str:
    """
    Attempts to open file as Parquet in Polars and returns markdown.
    """
    try: # Spreadsheet?
        if file_ext == 'ods':
            df = pl.read_ods(io.BytesIO(file_bytes))
        else:
            df = pl.read_excel(io.BytesIO(file_bytes))
    except Exception as e:
        logger.warning(f"Failed converting spreadsheet to markdown {file_name}: {e}")
        return ""
    return await df_to_markdown(df)


async def parquet_to_markdown(file_bytes: bytes) -> str:
    """
    Attempts to open file as Parquet in Polars and returns markdown.
    """
    try: # Parquet?
        df = pl.read_parquet(io.BytesIO(file_bytes))
    except:
        return ""
    return await df_to_markdown(df)


async def json_to_markdown(file_bytes: bytes) -> str:
    """
    Attempts to open file as JSON in Polars and returns markdown.
    """
    try: # Regular JSON?
        df = pl.read_json(io.BytesIO(file_bytes))
    except:
        try: # NDJSON?
            df = pl.read_ndjson(io.BytesIO(file_bytes))
        except:
            return ""
    return await df_to_markdown(df)


async def csv_to_markdown(file_bytes: bytes) -> str:
    """
    Attempts to detect and open file as CSV in polars and returns markdown.
    """
    try:
        # Decode and detect dialect (delimiter, quotechar, escapechar)
        text = file_bytes.decode('utf-8', errors='replace')
        sniffer = clevercsv.Sniffer()
        dialect = sniffer.sniff(text, verbose=False)
        if dialect is None or dialect.delimiter is None:
            return ""
        # Detect if file has header
        has_header = sniffer.has_header(text)
        # Map CleverCSV dialect to Polars parameters ('' -> None)
        quote_char = dialect.quotechar if dialect.quotechar != '' else None
        # Read CSV with Polars using detected dialect
        df = pl.read_csv(
            io.BytesIO(file_bytes),
            separator=dialect.delimiter,
            quote_char=quote_char,
            has_header=has_header,
            infer_schema_length=10000,   # Good balance of speed/accuracy
            ignore_errors=True,          # Skip malformed rows
            truncate_ragged_lines=True,  # Handle uneven row lengths
        )
        return await df_to_markdown(df)
    except Exception:
        return ""


async def df_to_markdown(df: pl.DataFrame) -> str:
    """
    Returns df flattened and exported to markdown.
    """
    # Flatten structs and stringify lists (drop primitive lists of len > 4)
    while any(isinstance(dtype, pl.Struct) or isinstance(dtype, pl.List) for dtype in df.dtypes):
        for col in df.columns:
            # Flatten structs
            if isinstance(df[col].dtype, pl.Struct):
                unnested = df[col].struct.unnest()
                unnested = unnested.rename({c: f"{col}.{c}" for c in unnested.columns})
                df = df.drop(col).hstack(unnested)
            elif isinstance(df[col].dtype, pl.List):
                # Drop primitive lists of len > 4
                prim_type = df[col].dtype.inner.is_numeric() or df[col].dtype.inner == pl.Boolean # type: ignore
                max_len_4 = (df[col].list.len().max() or 0) > 4 # type: ignore
                if prim_type and max_len_4:
                    df = df.drop(col)
                # Stringify remaining lists
                else:
                    df = df.with_columns(pl.col(col).list.join(", "))
    return df.to_pandas().to_markdown(index=False)


async def audio_to_markdown(file_bytes: bytes, file_name: str, content_type: str) -> str:
    """
    Transcribes audio/video using Whisper and returns markdown table.
    """
    try:
        endpoint = os.environ.get("WHISPER_URL", "http://whisper:8000").rstrip("/")
        
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                f"{endpoint}/v1/audio/transcriptions",
                files={'file': (file_name, file_bytes, content_type)},
                data={
                    'response_format': 'verbose_json',
                    'hallucination_silence_threshold': '2.0',
                    #'vad_filter': 'true', # causes hallucinations :(
                    #'language': 'da', # too restrictive, auto-detect is better
                },
            )
            response.raise_for_status()
            result = response.json()
        
        segments = result.get('segments', [])
        if not segments:
            logger.warning(f"No segments returned from Whisper for {file_name}")
            return ""
        
        # Merge segments adaptively to minute boundaries
        merged = merge_to_adaptive_minutes(segments, min_duration=30.0)
        
        # Build markdown table
        rows = [
            f"| {i} | {format_timestamp(seg['start'])} - {format_timestamp(seg['end'])} | {seg['text'].strip()} |"
            for i, seg in enumerate(merged, 1)
        ]
        
        header = "| IDX | Tid | Tekst |\n|-|-|-|"
        return header + "\n" + "\n".join(rows)
        
    except httpx.HTTPError as e:
        logger.warning(f"HTTP error transcribing {file_name}: {e}")
        return ""
    except Exception as e:
        logger.warning(f"Failed transcribing audio {file_name}: {e}")
        return ""


def merge_to_adaptive_minutes(segments: list[dict], min_duration: float = 30.0, interval: float = 60.0) -> list[dict]:
    """
    Merges Whisper segments adaptively:
    - Always merge until exceeding next whole minute from buffer start
    - After flush, next buffer must be min_duration AND exceed the next whole minute after that
    """
    
    merged = []
    buffer_segments = []
    buffer_start = 0.0
    
    for seg in segments:
        buffer_segments.append(seg)
        buffer_end = seg['end']
        buffer_duration = buffer_end - buffer_start
        
        # Next whole minute from buffer start
        next_minute_from_start = math.ceil(buffer_start / interval) * interval
        
        # Have we exceeded that minute?
        if buffer_end > next_minute_from_start:
            # Do we also have minimum duration?
            if buffer_duration >= min_duration:
                # What's the next whole minute after (start + min_duration)?
                min_end = buffer_start + min_duration
                next_minute_after_min = math.ceil(min_end / interval) * interval
                
                # Have we exceeded that minute too?
                if buffer_end >= next_minute_after_min:
                    # Flush buffer
                    merged_seg = {
                        'start': buffer_start,
                        'end': buffer_end,
                        'text': ' '.join(s['text'].strip() for s in buffer_segments)
                    }
                    merged.append(merged_seg)
                    
                    # Reset for next buffer
                    buffer_start = buffer_end
                    buffer_segments = []
    
    # Flush remaining
    if buffer_segments:
        buffer_end = buffer_segments[-1]['end']
        merged_seg = {
            'start': buffer_start,
            'end': buffer_end,
            'text': ' '.join(s['text'].strip() for s in buffer_segments)
        }
        merged.append(merged_seg)
    
    return merged


def format_timestamp(seconds: float) -> str:
    """
    Formats seconds to HH:MM:SS timestamp.
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
 

async def normalize_text(file_bytes: bytes) -> bytes:
    """
    If given text bytes, will encourage utf-8 and \n EOL.
    Returns bytes.
    """
    for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
        try:
            text = file_bytes.decode(encoding)
            text = text.replace('\r\n', '\n').replace('\r', '\n')
            return text.encode('utf-8')
        except:
            continue
    return file_bytes


async def normalize_audio(file_bytes: bytes, file_name: str) -> bytes:
    """
    Accepts (almost) any audio or video format and returns WAV16 bytes.
    """
    cmd = [
        'ffmpeg',
        '-hide_banner',
        '-nostdin',
        '-i', 'pipe:0',                     # Read input from stdin
        '-vn',                              # Discard video
        '-acodec', 'pcm_s16le',             # Audio Codec: PCM 16-bit (WAV standard)
        '-ar', '16000',                     # Audio Rate: 16 kHz (Whisper standard)
        '-channel_layout', 'mono',          # Mono layout explicited
        '-ac', '1',                         # Audio Channels: 1 (Mono)
        '-sample_fmt', 's16',               # Sample format explicited
        '-af', 'aresample=resampler=soxr',  # Soxr resampler
        '-f', 'wav',                        # Format: WAV
        'pipe:1'                            # Write to stdout
    ]
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate(input=file_bytes)
        if process.returncode != 0:
            error_msg = stderr.decode('utf-8', errors='ignore')
            logger.warning(f"ffmpeg failed for {file_name} (code {process.returncode}): {error_msg}")
            return b""
        return stdout
    except Exception as e:
        logger.warning(f"Failed normalizing audio {file_name}: {e}")
        return b""


async def normalize_image(file_bytes: bytes, file_name: str) -> bytes:
    """
    Accepts (almost) any image format and returns PNG bytes.
    """
    cmd = [
        'ffmpeg',
        '-hide_banner',
        '-nostdin',
        '-i', 'pipe:0',                     # Read from stdin
        '-map', '0:v:0',                    # Get first stream (video)
        '-frames:v', '1',                   # Get 1 frame only (for GIF/video)
        '-f', 'image2pipe',                 # Send image-bytes to pipe
        '-vcodec', 'png',                   # Output as PNG
        'pipe:1'                            # Write to stdout
    ]
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate(input=file_bytes)
        if process.returncode != 0:
            error_msg = stderr.decode('utf-8', errors='ignore')
            logger.warning(f"ffmpeg failed for {file_name} (code {process.returncode}): {error_msg}")
            return b""
        return stdout
    except Exception as e:
        logger.warning(f"Failed normalizing image {file_name}: {e}")
        return b""


async def document_to_markdown(file_bytes: bytes, file_name: str, file_ext: str) -> str:
    """
    Attempts to parse document through PDF > PNG > VLM and returns markdown.
    """
    return ""


async def document_to_pdf(file_bytes: bytes, file_ext: str) -> bytes | None:
    """
    Attempts to convert document to pdf using Gotenberg, returning bytes or None.
    """
    return None


async def image_to_markdown(file_bytes: bytes, file_name: str) -> str:
    """
    Attempts to perform OCR including image description(s) using VLM and returns markdown.
    """
    file_bytes = await normalize_image(file_bytes, file_name)
    return ""