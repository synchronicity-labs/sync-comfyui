import time, json, requests
from pathlib import Path
from os.path import getsize
from sync import Sync
from sync.common import Audio, Video, GenerationOptions

# ─────────────── INPUT NODE ──────────────────────────────────────────────
class SyncLipsyncInputNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path":        ("STRING", {"default": ""}),
                "audio_path":        ("STRING", {"default": ""}),
                "video_url":         ("STRING", {"default": ""}),
                "audio_url":         ("STRING", {"default": ""}),
                "tts_voice_id":      ("STRING", {"default": ""}),
                "tts_script":        ("STRING", {"default": ""}),
                "api_key":           ("STRING", {"default": ""}),
                "output_video_name": ("STRING", {"default": ""}),
                "webhook_url":       ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES, RETURN_NAMES = ("SYNC_INPUT",), ("sync_input",)
    FUNCTION, CATEGORY        = "provide", "Sync.so/Lipsync"

    def provide(
        self,
        video_path, audio_path, video_url, audio_url,
        tts_voice_id, tts_script,
        api_key, output_video_name, webhook_url,
    ):
        return (
            {
                "video_path":  video_path,
                "audio_path":  audio_path,
                "video_url":   video_url,
                "audio_url":   audio_url,
                "tts_voice_id": tts_voice_id,
                "tts_script":  tts_script,
                "api_key":     api_key,
                "output_video_name": output_video_name,
                "webhook_url": webhook_url,
                "poll_interval": 5.0,
            },
        )


# ─────────────── GENERATE NODE ──────────────────────────────────────
class SyncLipsyncMainNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sync_input":     ("SYNC_INPUT", {}),
                "model":          (["lipsync-2", "lipsync-1.9.0-beta"],),
                "segment_secs":   ("STRING", {"default": ""}),
                "segment_frames": ("STRING", {"default": ""}),
                "sync_mode":      (
                    ["loop", "bounce", "cut_off", "silence", "remap"],
                    {"default": "cut_off"},
                ),
                "temperature":    ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "active_speaker": ("BOOLEAN", {"default": False}),
                "occlusion_detection": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES, RETURN_NAMES = ("STRING",), ("output_path",)
    FUNCTION, CATEGORY         = "lipsync_generate", "Sync.so/Lipsync"

    def lipsync_generate(
        self,
        sync_input, model, segment_secs, segment_frames,
        sync_mode, temperature, active_speaker, occlusion_detection,
    ):
        video_path   = sync_input["video_path"]
        audio_path   = sync_input["audio_path"]
        video_url    = sync_input["video_url"]
        audio_url    = sync_input["audio_url"]
        tts_voice_id = sync_input["tts_voice_id"]
        tts_script   = sync_input["tts_script"]
        api_key      = sync_input["api_key"]
        poll_iv      = sync_input["poll_interval"]
        out_name     = sync_input["output_video_name"]
        webhook_url  = sync_input["webhook_url"]

        MAX_BYTES = 20 * 1024 * 1024
        headers   = {"x-api-key": api_key, "x-sync-source": "comfyui"}
        print(" lipsync_generate called")

        try:
            job_id = None

            # ──────────────── TTS MODE ────────────────
            if tts_voice_id and tts_script:
                if not video_url:
                    raise ValueError("TTS mode requires video_url.")

                print(" Using TTS input instead of audio")

                payload = {
                    "model": model,
                    "input": [
                        {
                            "type": "text",
                            "provider": {
                                "name":    "elevenlabs",
                                "voiceId": tts_voice_id,
                                "script":  tts_script,
                            },
                        },
                        {
                            "type": "video",
                            "url":  video_url,
                        },
                    ],
                    "options": {
                        "sync_mode":      sync_mode,
                        "temperature":    temperature,
                        "active_speaker": active_speaker,
                    },
                }

                if segment_secs:
                    payload["options"]["segments_secs"] = segment_secs
                if segment_frames:
                    payload["options"]["segments_frames"] = segment_frames
                if occlusion_detection:
                    payload["options"]["active_speaker_detection"] = {
                        "occlusion_detection_enabled": True
                    }
                if webhook_url:
                    payload["webhookUrl"] = webhook_url
                if out_name:
                    payload["outputFileName"] = out_name

                print(" Sending JSON POST request...")
                res = requests.post("https://api.sync.so/v2/generate", headers=headers, json=payload)
                print(f" Response code: {res.status_code}")
                res.raise_for_status()
                job_id = res.json()["id"]
                print(f" Job ID: {job_id}")

            # ───────────── FILE or URL MODE ─────────────
            else:
                if (video_path and Path(video_path).exists() and getsize(video_path) <= MAX_BYTES) or \
                   (audio_path and Path(audio_path).exists() and getsize(audio_path) <= MAX_BYTES):
                    print(" Using file upload (v2)")

                    input_block = [{"type": "video"}, {"type": "audio"}]
                    if segment_secs:
                        input_block[0]["segments_secs"] = json.loads(segment_secs)
                    if segment_frames:
                        input_block[0]["segments_frames"] = json.loads(segment_frames)

                    fields = [
                        ("model", model),
                        ("sync_mode", sync_mode),
                        ("temperature", str(temperature)),
                        ("active_speaker", str(active_speaker).lower()),
                        ("input", json.dumps(input_block))
                    ]

                    if webhook_url:
                        fields.append(("webhookUrl", webhook_url))
                    if occlusion_detection:
                        fields.append(("active_speaker_detection", json.dumps({"occlusion_detection_enabled": True})))

                    files = {}
                    if video_path and Path(video_path).exists():
                        files["video"] = open(video_path, "rb")
                        print(f" Opening video file: {video_path}")
                    elif video_url:
                        fields.append(("video_url", video_url))
                        print(f" Using video URL: {video_url}")

                    if audio_path and Path(audio_path).exists():
                        files["audio"] = open(audio_path, "rb")
                        print(f" Opening audio file: {audio_path}")
                    elif audio_url:
                        fields.append(("audio_url", audio_url))
                        print(f" Using audio URL: {audio_url}")

                    print(" Sending POST request...")
                    res = requests.post("https://api.sync.so/v2/generate", headers=headers, data=fields, files=files or None)
                    print(f" Response code: {res.status_code}")
                    res.raise_for_status()
                    job_id = res.json()["id"]
                    print(f" Job ID: {job_id}")
                else:
                    print(" Using SDK fallback")
                    client = Sync(base_url="https://api.sync.so", api_key=api_key).generations
                    video_kwargs = {}
                    if segment_secs:
                        video_kwargs["segments_secs"] = eval(segment_secs)
                    if segment_frames:
                        video_kwargs["segments_frames"] = eval(segment_frames)

                    response = client.create(
                        input=[Video(url=video_url, **video_kwargs),
                               Audio(url=audio_url)],
                        model=model,
                        options=GenerationOptions(
                            sync_mode=sync_mode,
                            temperature=temperature,
                            active_speaker=active_speaker,
                        ),
                    )
                    job_id = response.id
                    print(f" Job ID: {job_id}")

            # ──────── POLLING ────────
            timestamp = int(time.time())
            Path("output").mkdir(exist_ok=True)
            json_path = Path("output") / f"sync_job_{timestamp}.json"
            print(f" Polling job: {job_id}")
            status = None

            while status not in {"COMPLETED", "FAILED"}:
                print(f" Waiting {poll_iv}s...")
                time.sleep(poll_iv)
                poll = requests.get(f"https://api.sync.so/v2/generate/{job_id}", headers=headers)
                poll.raise_for_status()
                status = poll.json()["status"]
                print(f" Job status: {status}")

            with open(json_path, "w") as f:
                json.dump({"job_id": job_id, "final_status": status}, f, indent=2)

            if status != "COMPLETED":
                print(" Job failed")
                return ("",)

            output_url = poll.json().get("outputUrl") or (poll.json().get("result") or {}).get("outputUrl")
            segment_output_url = poll.json().get("segmentOutputUrl")

            base = Path(out_name).stem if out_name else f"sync_output_{timestamp}"
            full_output_path = Path("output") / f"{base}.mp4"
            if output_url:
                print(f" Downloading full video from: {output_url}")
                r = requests.get(output_url)
                r.raise_for_status()
                full_output_path.write_bytes(r.content)
                print(f" Full video saved to: {full_output_path}")
            else:
                full_output_path = ""

            if segment_output_url:
                segment_output_path = Path("output") / f"{base}_segment.mp4"
                print(f" Downloading segment video from: {segment_output_url}")
                r = requests.get(segment_output_url)
                r.raise_for_status()
                segment_output_path.write_bytes(r.content)
                print(f" Segment video saved to: {segment_output_path}")

            return (str(full_output_path),)

        except Exception as e:
            print(" Exception:", e)
            return ("",)


# ─────────────── OUTPUT NODE ────────────────────────────────
class SyncLipsyncOutputNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"output_path": ("STRING", {})}}

    RETURN_TYPES, RETURN_NAMES = ("STRING",), ("output_path",)
    FUNCTION, CATEGORY = "passthrough", "Sync.so/Lipsync"
    OUTPUT_NODE = True

    def passthrough(self, output_path):
        if not output_path or not Path(output_path).exists():
            return {"ui": {"texts": [" No video output was generated."]}, "result": (output_path,)}
        return {
            "ui": {"videos": [{"filename": Path(output_path).name, "subfolder": "", "type": "output"}]},
            "result": (output_path,)
        }

# ────────────── REGISTER ────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "SyncLipsyncInputNode":  SyncLipsyncInputNode,
    "SyncLipsyncMainNode":   SyncLipsyncMainNode,
    "SyncLipsyncOutputNode": SyncLipsyncOutputNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SyncLipsyncInputNode":  "sync.so lipsync – input",
    "SyncLipsyncMainNode":   "sync.so lipsync – generate",
    "SyncLipsyncOutputNode": "sync.so lipsync – output",
}

print("sync.so lipsync hybrid node loaded.")
