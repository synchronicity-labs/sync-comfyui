# import time, json, requests, os
# from pathlib import Path
# from os.path import getsize
# from sync import Sync
# from sync.common import Audio, Video, GenerationOptions

# # ─────────────── API KEY NODE ──────────────────────────────────────────────
# class SyncApiKeyNode:
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "api_key": ("STRING", {"default": ""}),
#             }
#         }

#     RETURN_TYPES = ("SYNC_API_KEY",)
#     RETURN_NAMES = ("api_key",)
#     FUNCTION = "provide_api_key"
#     CATEGORY = "Sync.so/Lipsync"

#     def provide_api_key(self, api_key):
#         return ({"api_key": api_key},)


# # ─────────────── UNIFIED VIDEO INPUT NODE ──────────────────────────────────────────────
# class SyncVideoInputNode:
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {},
#             "optional": {
#                 "video": ("*",),  # This accepts video frames from VHS LoadVideo node
#                 "video_path": ("STRING", {"default": ""}),
#                 "video_url": ("STRING", {"default": ""}),
#             }
#         }

#     RETURN_TYPES = ("SYNC_VIDEO",)
#     RETURN_NAMES = ("video",)
#     FUNCTION = "provide_video"
#     CATEGORY = "Sync.so/Lipsync"

#     def provide_video(self, video=None, video_path="", video_url=""):
        
#         if video is not None:
#             return self._process_loaded_video(video)
        
#         if video_path and video_path != "":
#             if os.path.exists(video_path):
#                 print(f" Using manual video path: {video_path}")
#                 return ({"video_path": video_path, "type": "path"},)
#             else:
#                 print(f" Manual video path not found: {video_path}")
        
#         if video_url and video_url != "":
#             print(f" Using video URL: {video_url}")
#             return ({"video_url": video_url, "type": "url"},)
        
#         print(" No valid video input provided")
#         return ({"video_path": "", "type": "path"},)

#     def _process_loaded_video(self, video):
#         fps = 30.0
        
#         try:
#             try:
#                 import cv2
#             except ImportError:
#                 error_msg = "OpenCV (cv2) is not installed. Please run: pip install opencv-python-headless"
#                 print(f" Error: {error_msg}")
#                 raise RuntimeError(error_msg)
            
#             import numpy as np
#             import torch
            
#             print(f" Processing video input with type: {type(video)}")
            
#             if video is None:
#                 raise ValueError("Video input is None")
            
#             if hasattr(video, 'save_to'):
#                 print(" Detected VideoFromFile object, saving to temp file...")
                
#                 temp_dir = os.path.join(os.path.dirname(__file__), "..", "..", "temp")
#                 os.makedirs(temp_dir, exist_ok=True)
                
#                 timestamp = int(time.time())
#                 temp_path = os.path.join(temp_dir, f"loaded_video_{timestamp}.mp4")
                
#                 video.save_to(temp_path)
                
#                 if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
#                     print(f" Video saved from VideoFromFile to: {temp_path}")
#                     return ({"video_path": temp_path, "type": "path"},)
#                 else:
#                     raise ValueError(f"Failed to save VideoFromFile to {temp_path}")
            
#             video_data = None
            
#             if isinstance(video, torch.Tensor):
#                 video_data = video.cpu().numpy()
#                 print(f" Tensor input shape: {video.shape}")
            
#             elif isinstance(video, np.ndarray):
#                 video_data = video
#                 print(f" NumPy array input shape: {video.shape}")
            
#             elif isinstance(video, list) and len(video) > 0:
#                 if isinstance(video[0], torch.Tensor):
#                     video_data = torch.stack(video).cpu().numpy()
#                 elif isinstance(video[0], np.ndarray):
#                     video_data = np.array(video)
#                 else:
#                     video_data = np.array(video)
#                 print(f" List converted to array shape: {video_data.shape}")
            
#             elif hasattr(video, 'frames'):
#                 frames = video.frames
#                 if isinstance(frames, torch.Tensor):
#                     video_data = frames.cpu().numpy()
#                 elif isinstance(frames, np.ndarray):
#                     video_data = frames
#                 else:
#                     video_data = np.array(frames)
#                 print(f" Frames attribute shape: {video_data.shape}")
            
#             elif hasattr(video, '__array__'):
#                 try:
#                     video_data = np.array(video)
#                     print(f" __array__ conversion shape: {video_data.shape}")
#                 except:
#                     raise ValueError(f"Failed to convert video using __array__ method")
            
#             else:
#                 try:
#                     video_data = np.array(video)
#                     if video_data.shape == ():  
#                         raise ValueError(f"Video conversion resulted in empty scalar. Video type: {type(video)}, available attributes: {dir(video)}")
#                     print(f" Direct conversion shape: {video_data.shape}")
#                 except:
#                     raise ValueError(f"Cannot convert video input of type {type(video)} to processable format. Available attributes: {[attr for attr in dir(video) if not attr.startswith('_')]}")
            
#             if video_data is None or video_data.size == 0:
#                 raise ValueError("Video data is empty after conversion")
            
#             temp_dir = os.path.join(os.path.dirname(__file__), "..", "..", "temp")
#             os.makedirs(temp_dir, exist_ok=True)
            
#             timestamp = int(time.time())
#             temp_path = os.path.join(temp_dir, f"loaded_video_{timestamp}.mp4")
            
#             if len(video_data.shape) == 4:  
#                 frames, height, width, channels = video_data.shape
#                 print(f" Video info: {frames} frames, {height}x{width}, {channels} channels")
#             elif len(video_data.shape) == 3:  
#                 height, width, channels = video_data.shape
#                 frames = 1
#                 video_data = np.expand_dims(video_data, axis=0)  
#                 print(f" Single frame video: {height}x{width}, {channels} channels")
#             else:
#                 raise ValueError(f"Unsupported video tensor shape: {video_data.shape}")
            
#             if height <= 0 or width <= 0 or frames <= 0:
#                 raise ValueError(f"Invalid video dimensions: {frames}x{height}x{width}")
            
#             if video_data.max() <= 1.0:
#                 video_data = (video_data * 255).astype(np.uint8)
#             else:
#                 video_data = video_data.astype(np.uint8)
            
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
            
#             if not out.isOpened():
#                 fourcc = cv2.VideoWriter_fourcc(*'XVID')
#                 temp_path = temp_path.replace('.mp4', '.avi')
#                 out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
                
#                 if not out.isOpened():
#                     raise RuntimeError("Failed to open video writer with any codec")
            
#             for i in range(frames):
#                 frame = video_data[i]
                
#                 if channels == 3:
#                     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#                 elif channels == 4:
#                     frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
#                 elif channels == 1:
#                     frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                
#                 out.write(frame)
            
#             out.release()
            
#             if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
#                 raise RuntimeError(f"Failed to create valid video file at {temp_path}")
            
#             print(f" Video saved from LoadVideo to: {temp_path}")
#             return ({"video_path": temp_path, "type": "path"},)
            
#         except Exception as e:
#             print(f" Error processing loaded video: {e}")
#             import traceback
#             traceback.print_exc()
#             return ({"video_path": "", "type": "path"},)


# # ─────────────── UNIFIED AUDIO INPUT NODE WITH TTS ──────────────────────────────────────────────
# class SyncAudioInputNode:
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {},
#             "optional": {
#                 "audio": ("AUDIO",),  
#                 "audio_path": ("STRING", {"default": ""}),
#                 "audio_url": ("STRING", {"default": ""}),
#                 "tts_voice_id": ("STRING", {"default": ""}),
#                 "tts_script": ("STRING", {"default": "", "multiline": True}),
#             }
#         }

#     RETURN_TYPES = ("SYNC_AUDIO",)
#     RETURN_NAMES = ("audio",)
#     FUNCTION = "provide_audio"
#     CATEGORY = "Sync.so/Lipsync"

#     def provide_audio(self, audio=None, audio_path="", audio_url="", tts_voice_id="", tts_script=""):
        
#         if tts_voice_id and tts_script:
#             print(f" Using TTS input: voice_id={tts_voice_id}, script length={len(tts_script)}")
#             return ({
#                 "type": "tts",
#                 "tts_voice_id": tts_voice_id,
#                 "tts_script": tts_script,
#                 "audio_path": "",  
#             },)
        
#         if audio is not None:
#             return self._process_loaded_audio(audio)
        
#         if audio_path and audio_path != "":
#             if os.path.exists(audio_path):
#                 print(f" Using manual audio path: {audio_path}")
#                 return ({"audio_path": audio_path, "type": "path"},)
#             else:
#                 print(f" Manual audio path not found: {audio_path}")
        
#         if audio_url and audio_url != "":
#             print(f" Using audio URL: {audio_url}")
#             return ({"audio_url": audio_url, "type": "url"},)
        
#         print(" No valid audio input provided")
#         return ({"audio_path": "", "type": "path"},)

#     def _process_loaded_audio(self, audio):
#         try:
#             try:
#                 import librosa
#             except ImportError:
#                 try:
#                     import soundfile as sf
#                     import numpy as np
#                     print(" Using soundfile as fallback for audio processing")
#                 except ImportError:
#                     error_msg = "Audio processing libraries not found. Please install: pip install librosa soundfile"
#                     print(f" Error: {error_msg}")
#                     raise RuntimeError(error_msg)
            
#             import numpy as np
#             import torch
            
#             print(f" Processing audio input with type: {type(audio)}")
            
#             if audio is None:
#                 raise ValueError("Audio input is None")
            
#             audio_data = None
#             sample_rate = 44100  
            
#             if isinstance(audio, dict):
#                 if 'waveform' in audio:
#                     audio_data = audio['waveform']
#                     if 'sample_rate' in audio:
#                         sample_rate = audio['sample_rate']
#                     print(f" Dictionary audio format detected, sample_rate: {sample_rate}")
#                 elif 'audio' in audio:
#                     audio_data = audio['audio']
#                     if 'sample_rate' in audio:
#                         sample_rate = audio['sample_rate']
#                     print(f" Dictionary audio format (alt) detected, sample_rate: {sample_rate}")
#                 else:
#                     raise ValueError(f"Dictionary audio format not recognized. Keys: {audio.keys()}")
            
#             elif hasattr(audio, 'save_to'):
#                 print(" Detected AudioFromFile object, saving to temp file...")
                
#                 temp_dir = os.path.join(os.path.dirname(__file__), "..", "..", "temp")
#                 os.makedirs(temp_dir, exist_ok=True)
                
#                 timestamp = int(time.time())
#                 temp_path = os.path.join(temp_dir, f"loaded_audio_{timestamp}.wav")
                
#                 audio.save_to(temp_path)
                
#                 if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
#                     print(f" Audio saved from AudioFromFile to: {temp_path}")
#                     return ({"audio_path": temp_path, "type": "path"},)
#                 else:
#                     raise ValueError(f"Failed to save AudioFromFile to {temp_path}")
            
#             elif isinstance(audio, torch.Tensor):
#                 audio_data = audio.cpu().numpy()
#                 print(f" Tensor input shape: {audio.shape}")
            
#             elif isinstance(audio, np.ndarray):
#                 audio_data = audio
#                 print(f" NumPy array input shape: {audio.shape}")
            
#             elif isinstance(audio, tuple) and len(audio) == 2:
#                 audio_data, sample_rate = audio
#                 if isinstance(audio_data, torch.Tensor):
#                     audio_data = audio_data.cpu().numpy()
#                 print(f" Tuple format detected: {type(audio_data)}, sample_rate: {sample_rate}")
            
#             elif hasattr(audio, 'audio'):
#                 audio_data = audio.audio
#                 if hasattr(audio, 'sample_rate'):
#                     sample_rate = audio.sample_rate
#                 if isinstance(audio_data, torch.Tensor):
#                     audio_data = audio_data.cpu().numpy()
#                 print(f" Audio attribute detected, sample_rate: {sample_rate}")
            
#             elif hasattr(audio, '__array__'):
#                 try:
#                     audio_data = np.array(audio)
#                     print(f" __array__ conversion shape: {audio_data.shape}")
#                 except:
#                     raise ValueError(f"Failed to convert audio using __array__ method")
            
#             else:
#                 try:
#                     audio_data = np.array(audio)
#                     if audio_data.shape == ():  
#                         raise ValueError(f"Audio conversion resulted in empty scalar. Audio type: {type(audio)}, available attributes: {dir(audio)}")
#                     print(f" Direct conversion shape: {audio_data.shape}")
#                 except:
#                     raise ValueError(f"Cannot convert audio input of type {type(audio)} to processable format. Available attributes: {[attr for attr in dir(audio) if not attr.startswith('_')]}")
            
#             if audio_data is None or audio_data.size == 0:
#                 raise ValueError("Audio data is empty after conversion")
            
#             if isinstance(audio_data, torch.Tensor):
#                 audio_data = audio_data.cpu().numpy()
            
#             if len(audio_data.shape) == 3:  
#                 if audio_data.shape[0] == 1:  
#                     audio_data = audio_data[0]
#                 else:
#                     raise ValueError(f"Batch size > 1 not supported: {audio_data.shape}")
            
#             if len(audio_data.shape) == 2:  
#                 if audio_data.shape[0] == 1:  
#                     audio_data = audio_data[0]
#                 elif audio_data.shape[0] == 2:  
#                     audio_data = np.mean(audio_data, axis=0)
#                     print(" Converted stereo to mono by averaging channels")
#                 else:
#                     audio_data = audio_data[0]
#                     print(f" Using first channel from {audio_data.shape[0]} channels")
            
#             if len(audio_data.shape) == 1: 
#                 pass
#             else:
#                 raise ValueError(f"Unsupported audio tensor shape: {audio_data.shape}")
            
#             print(f" Final audio shape: {audio_data.shape}, sample_rate: {sample_rate}")
            
#             if audio_data.max() > 1.0 or audio_data.min() < -1.0:
#                 if audio_data.max() > 1.0:
#                     audio_data = audio_data.astype(np.float32) / 32767.0
#                     print(" Normalized audio from int16 range to float32")
            
#             temp_dir = os.path.join(os.path.dirname(__file__), "..", "..", "temp")
#             os.makedirs(temp_dir, exist_ok=True)
            
#             timestamp = int(time.time())
#             temp_path = os.path.join(temp_dir, f"loaded_audio_{timestamp}.wav")
            
#             # Save audio file
#             try:
#                 import librosa
#                 librosa.output.write_wav if hasattr(librosa.output, 'write_wav') else None
#                 # For newer librosa versions, use soundfile
#                 import soundfile as sf
#                 sf.write(temp_path, audio_data, sample_rate)
#             except:
#                 # Fallback to soundfile if librosa doesn't work
#                 try:
#                     import soundfile as sf
#                     sf.write(temp_path, audio_data, sample_rate)
#                 except Exception as sf_error:
#                     raise RuntimeError(f"Failed to save audio file: {sf_error}")
            
#             if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
#                 raise RuntimeError(f"Failed to create valid audio file at {temp_path}")
            
#             print(f" Audio saved from LoadAudio to: {temp_path}")
#             return ({"audio_path": temp_path, "type": "path"},)
            
#         except Exception as e:
#             print(f" Error processing loaded audio: {e}")
#             import traceback
#             traceback.print_exc()
#             return ({"audio_path": "", "type": "path"},)


# # ─────────────── SIMPLIFIED GENERATE NODE ──────────────────────────────────────────
# class SyncLipsyncMainNode:
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "api_key": ("SYNC_API_KEY", {"forceInput": True}),
#                 "video": ("SYNC_VIDEO", {"forceInput": True}),
#                 "audio": ("SYNC_AUDIO", {"forceInput": True}),  # Now handles both audio and TTS
#                 "model": (["lipsync-2-pro", "lipsync-2", "lipsync-1.9.0-beta"],),
#                 "segment_secs": ("STRING", {"default": ""}),
#                 "segment_frames": ("STRING", {"default": ""}),
#                 "sync_mode": (
#                     ["loop", "bounce", "cut_off", "silence", "remap"],
#                     {"default": "cut_off"},
#                 ),
#                 "temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
#                 "active_speaker": ("BOOLEAN", {"default": False}),
#                 "occlusion_detection": ("BOOLEAN", {"default": False}),
#             }
#         }

#     RETURN_TYPES = ("STRING",)
#     RETURN_NAMES = ("output_path",)
#     FUNCTION = "lipsync_generate"
#     CATEGORY = "Sync.so/Lipsync"

#     @classmethod
#     def VALIDATE_INPUTS(cls, input_types):
#         """Validate that required inputs are connected"""
#         errors = []
        
#         if "video" not in input_types:
#             errors.append("video input is required")
        
#         if "audio" not in input_types:
#             errors.append("audio input is required")
        
#         return True if not errors else " | ".join(errors)

#     def lipsync_generate(
#         self,
#         api_key, video, audio, model, segment_secs, segment_frames,
#         sync_mode, temperature, active_speaker, occlusion_detection,
#     ):
#         api_key_str = api_key["api_key"]
        
#         video_path_str = ""
#         video_url_str = ""
#         if video["type"] == "path":
#             video_path_str = video["video_path"]
#         elif video["type"] == "url":
#             video_url_str = video["video_url"]
        
#         audio_path_str = ""
#         audio_url_str = ""
#         tts_voice_id = ""
#         tts_script = ""
#         is_tts_mode = False
        
#         if audio["type"] == "tts":
#             # TTS mode
#             is_tts_mode = True
#             tts_voice_id = audio.get("tts_voice_id", "")
#             tts_script = audio.get("tts_script", "")
#             print(f" TTS mode detected: voice_id={tts_voice_id}, script_length={len(tts_script)}")
#         elif audio["type"] == "path":
#             audio_path_str = audio.get("audio_path", "")
#             print(f" Audio file mode: {audio_path_str}")
#         elif audio["type"] == "url":
#             audio_url_str = audio.get("audio_url", "")
#             print(f" Audio URL mode: {audio_url_str}")

#         MAX_BYTES = 20 * 1024 * 1024
#         headers = {"x-api-key": api_key_str, "x-sync-source": "comfyui"}
#         poll_iv = 5.0
#         print(" lipsync_generate called")

#         try:
#             job_id = None

#             # ──────────────── TTS MODE ────────────────
#             if is_tts_mode and tts_voice_id and tts_script:
#                 print(" Using TTS input instead of audio")
                
#                 if video_url_str:
#                     # Pure JSON request for video URL + TTS
#                     payload = {
#                         "model": model,
#                         "input": [
#                             {
#                                 "type": "text",
#                                 "provider": {
#                                     "name": "elevenlabs",
#                                     "voiceId": tts_voice_id,
#                                     "script": tts_script,
#                                 },
#                             },
#                             {
#                                 "type": "video",
#                                 "url": video_url_str
#                             }
#                         ],
#                         "options": {
#                             "sync_mode": sync_mode,
#                             "temperature": temperature,
#                             "active_speaker": active_speaker,
#                         },
#                     }

#                     if segment_secs:
#                         payload["options"]["segments_secs"] = segment_secs
#                     if segment_frames:
#                         payload["options"]["segments_frames"] = segment_frames
#                     if occlusion_detection:
#                         payload["options"]["active_speaker_detection"] = {
#                             "occlusion_detection_enabled": True
#                         }

#                     print(" Sending TTS request with video URL...")
#                     print(f" Payload: {json.dumps(payload, indent=2)}")

#                     # Add Content-Type for JSON requests
#                     tts_headers = headers.copy()
#                     tts_headers["Content-Type"] = "application/json"
                    
#                     res = requests.post("https://api.sync.so/v2/generate", headers=tts_headers, json=payload)
                    
#                 elif video_path_str and Path(video_path_str).exists():
#                     # Multipart form data request for video file + TTS
#                     print(" Sending TTS request with video file upload...")
                    
#                     input_array = [
#                         {
#                             "type": "text",
#                             "provider": {
#                                 "name": "elevenlabs",
#                                 "voiceId": tts_voice_id,
#                                 "script": tts_script,  # Use "script" as in working standalone example
#                             },
#                         },
#                         {
#                             "type": "video"  # File will be attached separately
#                         }
#                     ]
                    
#                     data = {
#                         "model": model,
#                         "input": json.dumps(input_array),
#                     }
                    
#                     options = {
#                         "sync_mode": sync_mode,
#                         "temperature": temperature,
#                         "active_speaker": active_speaker,
#                     }
                    
#                     if segment_secs:
#                         options["segments_secs"] = segment_secs
#                     if segment_frames:
#                         options["segments_frames"] = segment_frames
#                     if occlusion_detection:
#                         options["active_speaker_detection"] = {
#                             "occlusion_detection_enabled": True
#                         }
                    
#                     data["options"] = json.dumps(options)
                    
#                     # Open and attach video file
#                     files = {"video": open(video_path_str, "rb")}
                    
#                     print(f" Form data: {data}")
#                     print(f" Video file: {video_path_str}")
                    
#                     file_headers = headers.copy()
                    
#                     res = requests.post("https://api.sync.so/v2/generate", headers=file_headers, data=data, files=files)
#                     files["video"].close()
                    
#                 else:
#                     raise ValueError("TTS mode requires either a video URL or video file.")

#                 print(f" Response code: {res.status_code}")
#                 print(f" Response headers: {dict(res.headers)}")

#                 # Add detailed error logging
#                 if res.status_code != 200:
#                     print(f" Error response content: {res.text}")
#                     try:
#                         error_json = res.json()
#                         print(f" Error details: {json.dumps(error_json, indent=2)}")
#                     except:
#                         print(" Could not parse error response as JSON")

#                 res.raise_for_status()
#                 job_id = res.json()["id"]
#                 print(f" Job ID: {job_id}")

#             # ───────────── FILE or URL MODE ─────────────
#             else:
#                 if (video_path_str and Path(video_path_str).exists() and getsize(video_path_str) <= MAX_BYTES) or \
#                    (audio_path_str and Path(audio_path_str).exists() and getsize(audio_path_str) <= MAX_BYTES):
#                     print(" Using file upload (v2)")

#                     input_block = [{"type": "video"}, {"type": "audio"}]
#                     if segment_secs:
#                         try:
#                             input_block[0]["segments_secs"] = json.loads(segment_secs)
#                         except:
#                             print(f" Warning: Could not parse segment_secs: {segment_secs}")
#                     if segment_frames:
#                         try:
#                             input_block[0]["segments_frames"] = json.loads(segment_frames)
#                         except:
#                             print(f" Warning: Could not parse segment_frames: {segment_frames}")

#                     fields = [
#                         ("model", model),
#                         ("sync_mode", sync_mode),
#                         ("temperature", str(temperature)),
#                         ("active_speaker", str(active_speaker).lower()),
#                         ("input", json.dumps(input_block))
#                     ]

#                     if occlusion_detection:
#                         fields.append(("active_speaker_detection", json.dumps({"occlusion_detection_enabled": True})))

#                     files = {}
#                     if video_path_str and Path(video_path_str).exists():
#                         files["video"] = open(video_path_str, "rb")
#                         print(f" Opening video file: {video_path_str}")
#                     elif video_url_str:
#                         fields.append(("video_url", video_url_str))
#                         print(f" Using video URL: {video_url_str}")

#                     if audio_path_str and Path(audio_path_str).exists():
#                         files["audio"] = open(audio_path_str, "rb")
#                         print(f" Opening audio file: {audio_path_str}")
#                     elif audio_url_str:
#                         fields.append(("audio_url", audio_url_str))
#                         print(f" Using audio URL: {audio_url_str}")

#                     print(" Sending POST request...")
#                     res = requests.post("https://api.sync.so/v2/generate", headers=headers, data=fields, files=files or None)
#                     print(f" Response code: {res.status_code}")
#                     res.raise_for_status()
#                     job_id = res.json()["id"]
#                     print(f" Job ID: {job_id}")
                    
#                     for file_handle in files.values():
#                         file_handle.close()
                        
#                 else:
#                     print(" Using SDK fallback")
#                     client = Sync(base_url="https://api.sync.so", api_key=api_key_str).generations
#                     video_kwargs = {}
#                     if segment_secs:
#                         try:
#                             video_kwargs["segments_secs"] = eval(segment_secs)
#                         except:
#                             print(f" Warning: Could not parse segment_secs: {segment_secs}")
#                     if segment_frames:
#                         try:
#                             video_kwargs["segments_frames"] = eval(segment_frames)
#                         except:
#                             print(f" Warning: Could not parse segment_frames: {segment_frames}")

#                     response = client.create(
#                         input=[Video(url=video_url_str, **video_kwargs),
#                                Audio(url=audio_url_str)],
#                         model=model,
#                         options=GenerationOptions(
#                             sync_mode=sync_mode,
#                             temperature=temperature,
#                             active_speaker=active_speaker,
#                         ),
#                     )
#                     job_id = response.id
#                     print(f" Job ID: {job_id}")

#             # ──────── POLLING ────────
#             timestamp = int(time.time())
#             Path("output").mkdir(exist_ok=True)
#             json_path = Path("output") / f"sync_job_{timestamp}.json"
#             print(f" Polling job: {job_id}")
#             status = None

#             while status not in {"COMPLETED", "FAILED"}:
#                 print(f" Waiting {poll_iv}s...")
#                 time.sleep(poll_iv)
#                 poll = requests.get(f"https://api.sync.so/v2/generate/{job_id}", headers=headers)
#                 poll.raise_for_status()
#                 status = poll.json()["status"]
#                 print(f" Job status: {status}")

#             with open(json_path, "w") as f:
#                 json.dump({"job_id": job_id, "final_status": status}, f, indent=2)

#             if status != "COMPLETED":
#                 print(" Job failed")
#                 return ("",)

#             output_url = poll.json().get("outputUrl") or (poll.json().get("result") or {}).get("outputUrl")
#             segment_output_url = poll.json().get("segmentOutputUrl")

#             base = f"sync_output_{timestamp}"
#             full_output_path = Path("output") / f"{base}.mp4"
#             if output_url:
#                 print(f" Downloading full video from: {output_url}")
#                 r = requests.get(output_url)
#                 r.raise_for_status()
#                 full_output_path.write_bytes(r.content)
#                 print(f" Full video saved to: {full_output_path}")
#             else:
#                 full_output_path = ""

#             if segment_output_url:
#                 segment_output_path = Path("output") / f"{base}_segment.mp4"
#                 print(f" Downloading segment video from: {segment_output_url}")
#                 r = requests.get(segment_output_url)
#                 r.raise_for_status()
#                 segment_output_path.write_bytes(r.content)
#                 print(f" Segment video saved to: {segment_output_path}")

#             return (str(full_output_path),)

#         except Exception as e:
#             print(" Exception:", e)
#             import traceback
#             traceback.print_exc()
#             return ("",)


# # ─────────────── OUTPUT NODE ────────────────────────────────
# class SyncLipsyncOutputNode:
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "output_path": ("STRING", {}),
#             },
#             "optional": {
#                 "custom_video_path": ("STRING", {"default": ""}),
#                 "custom_video_name": ("STRING", {"default": ""}),
#             }
#         }

#     RETURN_TYPES = ("STRING", "IMAGE")
#     RETURN_NAMES = ("output_path", "output_video")
#     FUNCTION = "passthrough"
#     CATEGORY = "Sync.so/Lipsync"
#     OUTPUT_NODE = True

#     def passthrough(self, output_path, custom_video_path="", custom_video_name=""):
#         try:
#             import torch
#             import numpy as np
#         except ImportError as e:
#             print(f" Error importing required libraries: {e}")
#             return {
#                 "ui": {"texts": [" Required libraries (torch/numpy) not available."]}, 
#                 "result": (output_path, None)
#             }
        
#         video_frames = None
#         final_output_path = output_path
        
#         if custom_video_path and custom_video_name and output_path and Path(output_path).exists():
#             try:
#                 import shutil
                
#                 custom_dir = Path(custom_video_path)
#                 custom_dir.mkdir(parents=True, exist_ok=True)
                
#                 # Add .mp4 extension if not present
#                 if not custom_video_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
#                     custom_video_name += '.mp4'
                
#                 custom_full_path = custom_dir / custom_video_name
                
#                 shutil.copy2(output_path, custom_full_path)
#                 print(f" Video copied to custom location: {custom_full_path}")
                
#                 final_output_path = str(custom_full_path)
                
#             except Exception as e:
#                 print(f" Error copying video to custom location: {e}")
#                 # Continue with original path if copying fails
#                 final_output_path = output_path
        
#         video_to_load = final_output_path if final_output_path and Path(final_output_path).exists() else output_path
        
#         if video_to_load and Path(video_to_load).exists():
#             try:
#                 import cv2
                
#                 print(f" Loading video frames from: {video_to_load}")
                
#                 cap = cv2.VideoCapture(str(video_to_load))
                
#                 if not cap.isOpened():
#                     print(f" Warning: Could not open video file: {video_to_load}")
#                     video_frames = torch.zeros((1, 64, 64, 3), dtype=torch.float32)  
#                 else:
#                     frames = []
#                     frame_count = 0
                    
#                     while True:
#                         ret, frame = cap.read()
#                         if not ret:
#                             break
                        
#                         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
#                         frame = frame.astype(np.float32) / 255.0
                        
#                         frames.append(frame)
#                         frame_count += 1
                    
#                     cap.release()
                    
#                     if frames:
#                         video_frames = torch.from_numpy(np.array(frames))
#                         print(f" Loaded {frame_count} frames with shape: {video_frames.shape}")
#                     else:
#                         print(" Warning: No frames loaded from video")
#                         video_frames = torch.zeros((1, 64, 64, 3), dtype=torch.float32)  
                        
#             except ImportError as e:
#                 print(f" Error importing cv2: {e}")
#                 video_frames = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
#             except Exception as e:
#                 print(f" Error loading video frames: {e}")
#                 import traceback
#                 traceback.print_exc()
#                 video_frames = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
#         else:
#             print(" ")
#             video_frames = torch.zeros((1, 64, 64, 3), dtype=torch.float32)  
        
#         if not final_output_path or not Path(final_output_path).exists():
#             return {
#                 "ui": {"texts": [" No video output was generated."]}, 
#                 "result": (final_output_path, video_frames)
#             }
        
#         video_filename = Path(final_output_path).name
#         video_subfolder = "" if str(Path(final_output_path).parent) == "output" else str(Path(final_output_path).parent)
        
#         return {
#             "ui": {"videos": [{"filename": video_filename, "subfolder": video_subfolder, "type": "output"}]},
#             "result": (final_output_path, video_frames)
#         }

# # ────────────── REGISTER (UPDATED) ────────────────────────────────────
# NODE_CLASS_MAPPINGS = {
#     "SyncApiKeyNode": SyncApiKeyNode,
#     "SyncVideoInputNode": SyncVideoInputNode,
#     "SyncAudioInputNode": SyncAudioInputNode, 
#     "SyncLipsyncMainNode": SyncLipsyncMainNode,
#     "SyncLipsyncOutputNode": SyncLipsyncOutputNode,
# }

# NODE_DISPLAY_NAME_MAPPINGS = {
#     "SyncApiKeyNode": "sync.so lipsync – api key",
#     "SyncVideoInputNode": "sync.so lipsync – video input",
#     "SyncAudioInputNode": "sync.so lipsync – audio/tts input",  
#     "SyncLipsyncMainNode": "sync.so lipsync – generate",
#     "SyncLipsyncOutputNode": "sync.so lipsync – output",
# }

# print("sync.so lipsync node loaded.")

import time, json, requests, os
from pathlib import Path
from os.path import getsize
from sync import Sync
from sync.common import Audio, Video, GenerationOptions

# ─────────────── API KEY NODE ──────────────────────────────────────────────
class SyncApiKeyNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("SYNC_API_KEY",)
    RETURN_NAMES = ("api_key",)
    FUNCTION = "provide_api_key"
    CATEGORY = "Sync.so/Lipsync"
    
    # Add instructions support
    DESCRIPTION = """
    Sync.so API Key
    
    Enter your Sync.so API key to authenticate requests. You can get your API key from:
    - Visit sync.so dashboard
    - Navigate to API settings
    - Copy your API key
    
    Key is required for all lipsync operations.
    """

    def provide_api_key(self, api_key):
        return ({"api_key": api_key},)


# ─────────────── UNIFIED VIDEO INPUT NODE ──────────────────────────────────────────────
class SyncVideoInputNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "video": ("*",),  # This accepts video frames from VHS LoadVideo node
                "video_path": ("STRING", {"default": ""}),
                "video_url": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("SYNC_VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "provide_video"
    CATEGORY = "Sync.so/Lipsync"
    
    # Add instructions support
    DESCRIPTION = """
    Video Input
    
    Provide a video input in one of three ways:
    
    1. Direct Video Connection: Connect a video output from other nodes (e.g., LoadVideo)
    2. Local File Path: Enter the full path to a video file on your system
    3. Video URL: Enter a direct URL to a video file
    
    Supported formats: MP4
    File size limit: 20MB for file uploads
    
    """

    def provide_video(self, video=None, video_path="", video_url=""):
        
        if video is not None:
            return self._process_loaded_video(video)
        
        if video_path and video_path != "":
            if os.path.exists(video_path):
                print(f" Using manual video path: {video_path}")
                return ({"video_path": video_path, "type": "path"},)
            else:
                print(f" Manual video path not found: {video_path}")
        
        if video_url and video_url != "":
            print(f" Using video URL: {video_url}")
            return ({"video_url": video_url, "type": "url"},)
        
        print(" No valid video input provided")
        return ({"video_path": "", "type": "path"},)

    def _process_loaded_video(self, video):
        fps = 30.0
        
        try:
            try:
                import cv2
            except ImportError:
                error_msg = "OpenCV (cv2) is not installed. Please run: pip install opencv-python-headless"
                print(f" Error: {error_msg}")
                raise RuntimeError(error_msg)
            
            import numpy as np
            import torch
            
            print(f" Processing video input with type: {type(video)}")
            
            if video is None:
                raise ValueError("Video input is None")
            
            if hasattr(video, 'save_to'):
                print(" Detected VideoFromFile object, saving to temp file...")
                
                temp_dir = os.path.join(os.path.dirname(__file__), "..", "..", "temp")
                os.makedirs(temp_dir, exist_ok=True)
                
                timestamp = int(time.time())
                temp_path = os.path.join(temp_dir, f"loaded_video_{timestamp}.mp4")
                
                video.save_to(temp_path)
                
                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                    print(f" Video saved from VideoFromFile to: {temp_path}")
                    return ({"video_path": temp_path, "type": "path"},)
                else:
                    raise ValueError(f"Failed to save VideoFromFile to {temp_path}")
            
            video_data = None
            
            if isinstance(video, torch.Tensor):
                video_data = video.cpu().numpy()
                print(f" Tensor input shape: {video.shape}")
            
            elif isinstance(video, np.ndarray):
                video_data = video
                print(f" NumPy array input shape: {video.shape}")
            
            elif isinstance(video, list) and len(video) > 0:
                if isinstance(video[0], torch.Tensor):
                    video_data = torch.stack(video).cpu().numpy()
                elif isinstance(video[0], np.ndarray):
                    video_data = np.array(video)
                else:
                    video_data = np.array(video)
                print(f" List converted to array shape: {video_data.shape}")
            
            elif hasattr(video, 'frames'):
                frames = video.frames
                if isinstance(frames, torch.Tensor):
                    video_data = frames.cpu().numpy()
                elif isinstance(frames, np.ndarray):
                    video_data = frames
                else:
                    video_data = np.array(frames)
                print(f" Frames attribute shape: {video_data.shape}")
            
            elif hasattr(video, '__array__'):
                try:
                    video_data = np.array(video)
                    print(f" __array__ conversion shape: {video_data.shape}")
                except:
                    raise ValueError(f"Failed to convert video using __array__ method")
            
            else:
                try:
                    video_data = np.array(video)
                    if video_data.shape == ():  
                        raise ValueError(f"Video conversion resulted in empty scalar. Video type: {type(video)}, available attributes: {dir(video)}")
                    print(f" Direct conversion shape: {video_data.shape}")
                except:
                    raise ValueError(f"Cannot convert video input of type {type(video)} to processable format. Available attributes: {[attr for attr in dir(video) if not attr.startswith('_')]}")
            
            if video_data is None or video_data.size == 0:
                raise ValueError("Video data is empty after conversion")
            
            temp_dir = os.path.join(os.path.dirname(__file__), "..", "..", "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            timestamp = int(time.time())
            temp_path = os.path.join(temp_dir, f"loaded_video_{timestamp}.mp4")
            
            if len(video_data.shape) == 4:  
                frames, height, width, channels = video_data.shape
                print(f" Video info: {frames} frames, {height}x{width}, {channels} channels")
            elif len(video_data.shape) == 3:  
                height, width, channels = video_data.shape
                frames = 1
                video_data = np.expand_dims(video_data, axis=0)  
                print(f" Single frame video: {height}x{width}, {channels} channels")
            else:
                raise ValueError(f"Unsupported video tensor shape: {video_data.shape}")
            
            if height <= 0 or width <= 0 or frames <= 0:
                raise ValueError(f"Invalid video dimensions: {frames}x{height}x{width}")
            
            if video_data.max() <= 1.0:
                video_data = (video_data * 255).astype(np.uint8)
            else:
                video_data = video_data.astype(np.uint8)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                temp_path = temp_path.replace('.mp4', '.avi')
                out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
                
                if not out.isOpened():
                    raise RuntimeError("Failed to open video writer with any codec")
            
            for i in range(frames):
                frame = video_data[i]
                
                if channels == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                elif channels == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                elif channels == 1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                
                out.write(frame)
            
            out.release()
            
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                raise RuntimeError(f"Failed to create valid video file at {temp_path}")
            
            print(f" Video saved from LoadVideo to: {temp_path}")
            return ({"video_path": temp_path, "type": "path"},)
            
        except Exception as e:
            print(f" Error processing loaded video: {e}")
            import traceback
            traceback.print_exc()
            return ({"video_path": "", "type": "path"},)


# ─────────────── UNIFIED AUDIO INPUT NODE WITH TTS ──────────────────────────────────────────────
class SyncAudioInputNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "audio": ("AUDIO",),  
                "audio_path": ("STRING", {"default": ""}),
                "audio_url": ("STRING", {"default": ""}),
                "tts_voice_id": ("STRING", {"default": ""}),
                "tts_script": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("SYNC_AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "provide_audio"
    CATEGORY = "Sync.so/Lipsync"
    
    # Add instructions support
    DESCRIPTION = """
    Audio/TTS Input
    
    Provide audio input in one of the four ways:
    
    1. Direct Audio Connection: Connect audio from other nodes
    2. Local Audio File: Enter path to audio file (WAV, MP3, etc.)
    3. Audio URL: Enter direct URL to audio file
    
    Text-to-Speech Option:
    4. TTS Generation: Use ElevenLabs TTS
       - TTS Voice ID: Enter your ElevenLabs voice ID
       - TTS Script: Enter the text to be spoken
    
    Priority: TTS takes priority if both voice ID and script are provided.
    
    """

    def provide_audio(self, audio=None, audio_path="", audio_url="", tts_voice_id="", tts_script=""):
        
        if tts_voice_id and tts_script:
            print(f" Using TTS input: voice_id={tts_voice_id}, script length={len(tts_script)}")
            return ({
                "type": "tts",
                "tts_voice_id": tts_voice_id,
                "tts_script": tts_script,
                "audio_path": "",  
            },)
        
        if audio is not None:
            return self._process_loaded_audio(audio)
        
        if audio_path and audio_path != "":
            if os.path.exists(audio_path):
                print(f" Using manual audio path: {audio_path}")
                return ({"audio_path": audio_path, "type": "path"},)
            else:
                print(f" Manual audio path not found: {audio_path}")
        
        if audio_url and audio_url != "":
            print(f" Using audio URL: {audio_url}")
            return ({"audio_url": audio_url, "type": "url"},)
        
        print(" No valid audio input provided")
        return ({"audio_path": "", "type": "path"},)

    def _process_loaded_audio(self, audio):
        try:
            try:
                import librosa
            except ImportError:
                try:
                    import soundfile as sf
                    import numpy as np
                    print(" Using soundfile as fallback for audio processing")
                except ImportError:
                    error_msg = "Audio processing libraries not found. Please install: pip install librosa soundfile"
                    print(f" Error: {error_msg}")
                    raise RuntimeError(error_msg)
            
            import numpy as np
            import torch
            
            print(f" Processing audio input with type: {type(audio)}")
            
            if audio is None:
                raise ValueError("Audio input is None")
            
            audio_data = None
            sample_rate = 44100  
            
            if isinstance(audio, dict):
                if 'waveform' in audio:
                    audio_data = audio['waveform']
                    if 'sample_rate' in audio:
                        sample_rate = audio['sample_rate']
                    print(f" Dictionary audio format detected, sample_rate: {sample_rate}")
                elif 'audio' in audio:
                    audio_data = audio['audio']
                    if 'sample_rate' in audio:
                        sample_rate = audio['sample_rate']
                    print(f" Dictionary audio format (alt) detected, sample_rate: {sample_rate}")
                else:
                    raise ValueError(f"Dictionary audio format not recognized. Keys: {audio.keys()}")
            
            elif hasattr(audio, 'save_to'):
                print(" Detected AudioFromFile object, saving to temp file...")
                
                temp_dir = os.path.join(os.path.dirname(__file__), "..", "..", "temp")
                os.makedirs(temp_dir, exist_ok=True)
                
                timestamp = int(time.time())
                temp_path = os.path.join(temp_dir, f"loaded_audio_{timestamp}.wav")
                
                audio.save_to(temp_path)
                
                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                    print(f" Audio saved from AudioFromFile to: {temp_path}")
                    return ({"audio_path": temp_path, "type": "path"},)
                else:
                    raise ValueError(f"Failed to save AudioFromFile to {temp_path}")
            
            elif isinstance(audio, torch.Tensor):
                audio_data = audio.cpu().numpy()
                print(f" Tensor input shape: {audio.shape}")
            
            elif isinstance(audio, np.ndarray):
                audio_data = audio
                print(f" NumPy array input shape: {audio.shape}")
            
            elif isinstance(audio, tuple) and len(audio) == 2:
                audio_data, sample_rate = audio
                if isinstance(audio_data, torch.Tensor):
                    audio_data = audio_data.cpu().numpy()
                print(f" Tuple format detected: {type(audio_data)}, sample_rate: {sample_rate}")
            
            elif hasattr(audio, 'audio'):
                audio_data = audio.audio
                if hasattr(audio, 'sample_rate'):
                    sample_rate = audio.sample_rate
                if isinstance(audio_data, torch.Tensor):
                    audio_data = audio_data.cpu().numpy()
                print(f" Audio attribute detected, sample_rate: {sample_rate}")
            
            elif hasattr(audio, '__array__'):
                try:
                    audio_data = np.array(audio)
                    print(f" __array__ conversion shape: {audio_data.shape}")
                except:
                    raise ValueError(f"Failed to convert audio using __array__ method")
            
            else:
                try:
                    audio_data = np.array(audio)
                    if audio_data.shape == ():  
                        raise ValueError(f"Audio conversion resulted in empty scalar. Audio type: {type(audio)}, available attributes: {dir(audio)}")
                    print(f" Direct conversion shape: {audio_data.shape}")
                except:
                    raise ValueError(f"Cannot convert audio input of type {type(audio)} to processable format. Available attributes: {[attr for attr in dir(audio) if not attr.startswith('_')]}")
            
            if audio_data is None or audio_data.size == 0:
                raise ValueError("Audio data is empty after conversion")
            
            if isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.cpu().numpy()
            
            if len(audio_data.shape) == 3:  
                if audio_data.shape[0] == 1:  
                    audio_data = audio_data[0]
                else:
                    raise ValueError(f"Batch size > 1 not supported: {audio_data.shape}")
            
            if len(audio_data.shape) == 2:  
                if audio_data.shape[0] == 1:  
                    audio_data = audio_data[0]
                elif audio_data.shape[0] == 2:  
                    audio_data = np.mean(audio_data, axis=0)
                    print(" Converted stereo to mono by averaging channels")
                else:
                    audio_data = audio_data[0]
                    print(f" Using first channel from {audio_data.shape[0]} channels")
            
            if len(audio_data.shape) == 1: 
                pass
            else:
                raise ValueError(f"Unsupported audio tensor shape: {audio_data.shape}")
            
            print(f" Final audio shape: {audio_data.shape}, sample_rate: {sample_rate}")
            
            if audio_data.max() > 1.0 or audio_data.min() < -1.0:
                if audio_data.max() > 1.0:
                    audio_data = audio_data.astype(np.float32) / 32767.0
                    print(" Normalized audio from int16 range to float32")
            
            temp_dir = os.path.join(os.path.dirname(__file__), "..", "..", "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            timestamp = int(time.time())
            temp_path = os.path.join(temp_dir, f"loaded_audio_{timestamp}.wav")
            
            # Save audio file
            try:
                import librosa
                librosa.output.write_wav if hasattr(librosa.output, 'write_wav') else None
                # For newer librosa versions, use soundfile
                import soundfile as sf
                sf.write(temp_path, audio_data, sample_rate)
            except:
                # Fallback to soundfile if librosa doesn't work
                try:
                    import soundfile as sf
                    sf.write(temp_path, audio_data, sample_rate)
                except Exception as sf_error:
                    raise RuntimeError(f"Failed to save audio file: {sf_error}")
            
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                raise RuntimeError(f"Failed to create valid audio file at {temp_path}")
            
            print(f" Audio saved from LoadAudio to: {temp_path}")
            return ({"audio_path": temp_path, "type": "path"},)
            
        except Exception as e:
            print(f" Error processing loaded audio: {e}")
            import traceback
            traceback.print_exc()
            return ({"audio_path": "", "type": "path"},)


# ─────────────── SIMPLIFIED GENERATE NODE ──────────────────────────────────────────
class SyncLipsyncMainNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("SYNC_API_KEY", {"forceInput": True}),
                "video": ("SYNC_VIDEO", {"forceInput": True}),
                "audio": ("SYNC_AUDIO", {"forceInput": True}),  # Now handles both audio and TTS
                "model": (["lipsync-2-pro", "lipsync-2", "lipsync-1.9.0-beta"],),
                "segment_secs": ("STRING", {"default": ""}),
                "segment_frames": ("STRING", {"default": ""}),
                "sync_mode": (
                    ["loop", "bounce", "cut_off", "silence", "remap"],
                    {"default": "cut_off"},
                ),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "active_speaker": ("BOOLEAN", {"default": False}),
                "occlusion_detection": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "lipsync_generate"
    CATEGORY = "Sync.so/Lipsync"
    
    # Add instructions support
    DESCRIPTION = """
    
    Required Inputs:
    - API Key: Connection from API Key node
    - Video: Connection from Video Input node  
    - Audio: Connection from Audio/TTS Input node
    
    Model Options:
    - lipsync-2-pro: Highest quality (Subscription Needed)
    - lipsync-2: Balanced quality and speed
    - lipsync-1.9.0-beta: Fastest processing
    
    Advanced Settings:
    - Segment Secs: JSON array for time-based segments (e.g. [0, 5, 10])
    - Segment Frames: JSON array for frame-based segments
    - Sync Mode: How to handle audio/video length mismatch
      - cut_off: Trim longer content
      - loop: Loop shorter content
      - bounce: Bounce shorter content
      - silence: Add silence to shorter content
      - remap: Stretch content to match
    - Temperature
    - Active Speaker: Enable speaker detection
    - Occlusion Detection: Detect when face is blocked
    
    """

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        """Validate that required inputs are connected"""
        errors = []
        
        if "video" not in input_types:
            errors.append("video input is required")
        
        if "audio" not in input_types:
            errors.append("audio input is required")
        
        return True if not errors else " | ".join(errors)

    def lipsync_generate(
        self,
        api_key, video, audio, model, segment_secs, segment_frames,
        sync_mode, temperature, active_speaker, occlusion_detection,
    ):
        api_key_str = api_key["api_key"]
        
        video_path_str = ""
        video_url_str = ""
        if video["type"] == "path":
            video_path_str = video["video_path"]
        elif video["type"] == "url":
            video_url_str = video["video_url"]
        
        audio_path_str = ""
        audio_url_str = ""
        tts_voice_id = ""
        tts_script = ""
        is_tts_mode = False
        
        if audio["type"] == "tts":
            # TTS mode
            is_tts_mode = True
            tts_voice_id = audio.get("tts_voice_id", "")
            tts_script = audio.get("tts_script", "")
            print(f" TTS mode detected: voice_id={tts_voice_id}, script_length={len(tts_script)}")
        elif audio["type"] == "path":
            audio_path_str = audio.get("audio_path", "")
            print(f" Audio file mode: {audio_path_str}")
        elif audio["type"] == "url":
            audio_url_str = audio.get("audio_url", "")
            print(f" Audio URL mode: {audio_url_str}")

        MAX_BYTES = 20 * 1024 * 1024
        headers = {"x-api-key": api_key_str, "x-sync-source": "comfyui"}
        poll_iv = 5.0
        print(" lipsync_generate called")

        try:
            job_id = None

            # ──────────────── TTS MODE ────────────────
            if is_tts_mode and tts_voice_id and tts_script:
                print(" Using TTS input instead of audio")
                
                if video_url_str:
                    # Pure JSON request for video URL + TTS
                    payload = {
                        "model": model,
                        "input": [
                            {
                                "type": "text",
                                "provider": {
                                    "name": "elevenlabs",
                                    "voiceId": tts_voice_id,
                                    "script": tts_script,
                                },
                            },
                            {
                                "type": "video",
                                "url": video_url_str
                            }
                        ],
                        "options": {
                            "sync_mode": sync_mode,
                            "temperature": temperature,
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

                    print(" Sending TTS request with video URL...")
                    print(f" Payload: {json.dumps(payload, indent=2)}")

                    # Add Content-Type for JSON requests
                    tts_headers = headers.copy()
                    tts_headers["Content-Type"] = "application/json"
                    
                    res = requests.post("https://api.sync.so/v2/generate", headers=tts_headers, json=payload)
                    
                elif video_path_str and Path(video_path_str).exists():
                    # Multipart form data request for video file + TTS
                    print(" Sending TTS request with video file upload...")
                    
                    input_array = [
                        {
                            "type": "text",
                            "provider": {
                                "name": "elevenlabs",
                                "voiceId": tts_voice_id,
                                "script": tts_script,  # Use "script" as in working standalone example
                            },
                        },
                        {
                            "type": "video"  # File will be attached separately
                        }
                    ]
                    
                    data = {
                        "model": model,
                        "input": json.dumps(input_array),
                    }
                    
                    options = {
                        "sync_mode": sync_mode,
                        "temperature": temperature,
                        "active_speaker": active_speaker,
                    }
                    
                    if segment_secs:
                        options["segments_secs"] = segment_secs
                    if segment_frames:
                        options["segments_frames"] = segment_frames
                    if occlusion_detection:
                        options["active_speaker_detection"] = {
                            "occlusion_detection_enabled": True
                        }
                    
                    data["options"] = json.dumps(options)
                    
                    # Open and attach video file
                    files = {"video": open(video_path_str, "rb")}
                    
                    print(f" Form data: {data}")
                    print(f" Video file: {video_path_str}")
                    
                    file_headers = headers.copy()
                    
                    res = requests.post("https://api.sync.so/v2/generate", headers=file_headers, data=data, files=files)
                    files["video"].close()
                    
                else:
                    raise ValueError("TTS mode requires either a video URL or video file.")

                print(f" Response code: {res.status_code}")
                print(f" Response headers: {dict(res.headers)}")

                # Add detailed error logging
                if res.status_code != 200:
                    print(f" Error response content: {res.text}")
                    try:
                        error_json = res.json()
                        print(f" Error details: {json.dumps(error_json, indent=2)}")
                    except:
                        print(" Could not parse error response as JSON")

                res.raise_for_status()
                job_id = res.json()["id"]
                print(f" Job ID: {job_id}")

            # ───────────── FILE or URL MODE ─────────────
            else:
                if (video_path_str and Path(video_path_str).exists() and getsize(video_path_str) <= MAX_BYTES) or \
                   (audio_path_str and Path(audio_path_str).exists() and getsize(audio_path_str) <= MAX_BYTES):
                    print(" Using file upload (v2)")

                    input_block = [{"type": "video"}, {"type": "audio"}]
                    if segment_secs:
                        try:
                            input_block[0]["segments_secs"] = json.loads(segment_secs)
                        except:
                            print(f" Warning: Could not parse segment_secs: {segment_secs}")
                    if segment_frames:
                        try:
                            input_block[0]["segments_frames"] = json.loads(segment_frames)
                        except:
                            print(f" Warning: Could not parse segment_frames: {segment_frames}")

                    fields = [
                        ("model", model),
                        ("sync_mode", sync_mode),
                        ("temperature", str(temperature)),
                        ("active_speaker", str(active_speaker).lower()),
                        ("input", json.dumps(input_block))
                    ]

                    if occlusion_detection:
                        fields.append(("active_speaker_detection", json.dumps({"occlusion_detection_enabled": True})))

                    files = {}
                    if video_path_str and Path(video_path_str).exists():
                        files["video"] = open(video_path_str, "rb")
                        print(f" Opening video file: {video_path_str}")
                    elif video_url_str:
                        fields.append(("video_url", video_url_str))
                        print(f" Using video URL: {video_url_str}")

                    if audio_path_str and Path(audio_path_str).exists():
                        files["audio"] = open(audio_path_str, "rb")
                        print(f" Opening audio file: {audio_path_str}")
                    elif audio_url_str:
                        fields.append(("audio_url", audio_url_str))
                        print(f" Using audio URL: {audio_url_str}")

                    print(" Sending POST request...")
                    res = requests.post("https://api.sync.so/v2/generate", headers=headers, data=fields, files=files or None)
                    print(f" Response code: {res.status_code}")
                    res.raise_for_status()
                    job_id = res.json()["id"]
                    print(f" Job ID: {job_id}")
                    
                    for file_handle in files.values():
                        file_handle.close()
                        
                else:
                    print(" Using SDK fallback")
                    client = Sync(base_url="https://api.sync.so", api_key=api_key_str).generations
                    video_kwargs = {}
                    if segment_secs:
                        try:
                            video_kwargs["segments_secs"] = eval(segment_secs)
                        except:
                            print(f" Warning: Could not parse segment_secs: {segment_secs}")
                    if segment_frames:
                        try:
                            video_kwargs["segments_frames"] = eval(segment_frames)
                        except:
                            print(f" Warning: Could not parse segment_frames: {segment_frames}")

                    response = client.create(
                        input=[Video(url=video_url_str, **video_kwargs),
                               Audio(url=audio_url_str)],
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

            base = f"sync_output_{timestamp}"
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
            import traceback
            traceback.print_exc()
            return ("",)


# ─────────────── OUTPUT NODE ────────────────────────────────
class SyncLipsyncOutputNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_path": ("STRING", {}),
            },
            "optional": {
                "custom_video_path": ("STRING", {"default": ""}),
                "custom_video_name": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("output_path", "output_video")
    FUNCTION = "passthrough"
    CATEGORY = "Sync.so/Lipsync"
    OUTPUT_NODE = True
    
    # Add instructions support
    DESCRIPTION = """
    Lipsync Output
 
    Optional Settings:
    - Custom Video Path: Directory to copy output video
    - Custom Video Name: Custom filename for the output video
    """

    def passthrough(self, output_path, custom_video_path="", custom_video_name=""):
        try:
            import torch
            import numpy as np
        except ImportError as e:
            print(f" Error importing required libraries: {e}")
            return {
                "ui": {"texts": [" Required libraries (torch/numpy) not available."]}, 
                "result": (output_path, None)
            }
        
        video_frames = None
        final_output_path = output_path
        
        if custom_video_path and custom_video_name and output_path and Path(output_path).exists():
            try:
                import shutil
                
                custom_dir = Path(custom_video_path)
                custom_dir.mkdir(parents=True, exist_ok=True)
                
                # Add .mp4 extension if not present
                if not custom_video_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                    custom_video_name += '.mp4'
                
                custom_full_path = custom_dir / custom_video_name
                
                shutil.copy2(output_path, custom_full_path)
                print(f" Video copied to custom location: {custom_full_path}")
                
                final_output_path = str(custom_full_path)
                
            except Exception as e:
                print(f" Error copying video to custom location: {e}")
                # Continue with original path if copying fails
                final_output_path = output_path
        
        video_to_load = final_output_path if final_output_path and Path(final_output_path).exists() else output_path
        
        if video_to_load and Path(video_to_load).exists():
            try:
                import cv2
                
                print(f" Loading video frames from: {video_to_load}")
                
                cap = cv2.VideoCapture(str(video_to_load))
                
                if not cap.isOpened():
                    print(f" Warning: Could not open video file: {video_to_load}")
                    video_frames = torch.zeros((1, 64, 64, 3), dtype=torch.float32)  
                else:
                    frames = []
                    frame_count = 0
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        frame = frame.astype(np.float32) / 255.0
                        
                        frames.append(frame)
                        frame_count += 1
                    
                    cap.release()
                    
                    if frames:
                        video_frames = torch.from_numpy(np.array(frames))
                        print(f" Loaded {frame_count} frames with shape: {video_frames.shape}")
                    else:
                        print(" Warning: No frames loaded from video")
                        video_frames = torch.zeros((1, 64, 64, 3), dtype=torch.float32)  
                        
            except ImportError as e:
                print(f" Error importing cv2: {e}")
                video_frames = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            except Exception as e:
                print(f" Error loading video frames: {e}")
                import traceback
                traceback.print_exc()
                video_frames = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        else:
            print(" ")
            video_frames = torch.zeros((1, 64, 64, 3), dtype=torch.float32)  
        
        if not final_output_path or not Path(final_output_path).exists():
            return {
                "ui": {"texts": [" No video output was generated."]}, 
                "result": (final_output_path, video_frames)
            }
        
        video_filename = Path(final_output_path).name
        video_subfolder = "" if str(Path(final_output_path).parent) == "output" else str(Path(final_output_path).parent)
        
        return {
            "ui": {"videos": [{"filename": video_filename, "subfolder": video_subfolder, "type": "output"}]},
            "result": (final_output_path, video_frames)
        }

# ────────────── REGISTER (UPDATED) ────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "SyncApiKeyNode": SyncApiKeyNode,
    "SyncVideoInputNode": SyncVideoInputNode,
    "SyncAudioInputNode": SyncAudioInputNode, 
    "SyncLipsyncMainNode": SyncLipsyncMainNode,
    "SyncLipsyncOutputNode": SyncLipsyncOutputNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SyncApiKeyNode": "sync.so lipsync – api key",
    "SyncVideoInputNode": "sync.so lipsync – video input",
    "SyncAudioInputNode": "sync.so lipsync – audio/tts input",  
    "SyncLipsyncMainNode": "sync.so lipsync – generate",
    "SyncLipsyncOutputNode": "sync.so lipsync – output",
}

print("sync.so lipsync node loaded.")
