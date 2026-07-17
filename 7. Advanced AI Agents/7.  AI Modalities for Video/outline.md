```text
START — a video file enters the pipeline
  │
  ▼
[V1] Inspect the video
  ├── resolution
  ├── FPS
  ├── frame count
  └── duration
  │
  ▼
[V2] Sample representative frames
  ├── sending every frame is expensive and redundant
  ├── resize frames to control cost
  ├── preserve timestamps
  └── convert BGR → RGB
  │
  ▼
[V2] Measure frame-to-frame change
  ├── small difference → visually similar frames
  └── large difference → motion, event change or shot boundary
  │
  ▼
[V3] Prepare frames for the vision model
  ├── encode each frame as JPEG/Base64
  ├── attach its timestamp
  └── send multiple ordered frames in one request
  │
  ▼
[V4] Video understanding
  ├── summarize what happens
  ├── identify key moments
  ├── extract visible text
  └── return either prose or strict structured JSON
  │
  ▼
[V5] Understand the cost trade-off
  ├── more frames → better temporal coverage
  ├── higher resolution → better visual detail
  └── both increase visual tokens and processing cost
  │
  ▼
[V6] Extract and transcribe audio
  ├── separate audio using ffmpeg
  ├── transcribe with timestamps
  └── produce segments:
        {start, end, text}
  │
  ▼
[V7] Fuse audio and video
  ├── transcript explains what was said
  ├── keyframes explain what was shown
  └── the model generates a timestamp-grounded summary
  │
  ▼
[V8] Build searchable video chunks
  │
  ├── visual route:
  │     frame → caption → {timestamp, text}
  │
  └── audio route:
        transcript segment → {timestamp, text}
  │
  ▼
[V8] Embed every chunk
  ├── convert captions or transcript segments into vectors
  └── keep timestamps attached as metadata
  │
  ▼
════════════════════ QUERY TIME ════════════════════
  │
  ▼
User asks a question about the video
  │
  ▼
Embed the question using the same embedding model
  │
  ▼
Compare the query vector with all video-chunk vectors
  │
  ▼
Return the most relevant:
  ├── timestamps
  ├── captions or transcript segments
  └── similarity scores
  │
  ▼
OPTIONAL — complete the RAG pipeline
  ├── send retrieved moments to an LLM
  └── generate an answer grounded in those moments
  │
  ▼
END RESULT

One video can now support:

  • visual summarization
  • timestamped event detection
  • audio–visual understanding
  • semantic search across moments
  • a foundation for video RAG


CENTRAL IDEA

Do not send the entire video blindly.

Convert it into a compact representation made of:

  sampled frames
      +
  timestamps
      +
  transcript
      +
  searchable text chunks
```
