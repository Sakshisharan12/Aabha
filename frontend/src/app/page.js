"use client";

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Download, AlertCircle, Volume2, Camera, Upload, Eye, RefreshCw, Sparkles, Boxes } from 'lucide-react';

/* ============================================
   Inline SVG Components — Accessibility Themed
   ============================================ */

// Hero: Stylized eye with radiating sound waves — "seeing through sound"
const EyeSoundWaveSvg = () => (
  <svg className="hero-svg" viewBox="0 0 120 120" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
    <path className="sound-wave" d="M90 60c0-16.57-13.43-30-30-30" stroke="#6C3CE1" strokeWidth="2" strokeLinecap="round" fill="none" opacity="0.3"/>
    <path className="sound-wave" d="M97 60c0-20.43-16.57-37-37-37" stroke="#6C3CE1" strokeWidth="1.5" strokeLinecap="round" fill="none" opacity="0.2"/>
    <path className="sound-wave" d="M104 60c0-24.3-19.7-44-44-44" stroke="#6C3CE1" strokeWidth="1" strokeLinecap="round" fill="none" opacity="0.15"/>
    <path className="sound-wave" d="M30 60c0 16.57 13.43 30 30 30" stroke="#E8734A" strokeWidth="2" strokeLinecap="round" fill="none" opacity="0.3"/>
    <path className="sound-wave" d="M23 60c0 20.43 16.57 37 37 37" stroke="#E8734A" strokeWidth="1.5" strokeLinecap="round" fill="none" opacity="0.2"/>
    <path className="sound-wave" d="M16 60c0 24.3 19.7 44 44 44" stroke="#E8734A" strokeWidth="1" strokeLinecap="round" fill="none" opacity="0.15"/>
    <path d="M60 38C45 38 33 50 28 60c5 10 17 22 32 22s27-12 32-22c-5-10-17-22-32-22z" fill="#F3F0FF" stroke="#6C3CE1" strokeWidth="2.5"/>
    <circle cx="60" cy="60" r="12" fill="#6C3CE1" opacity="0.9"/>
    <circle cx="60" cy="60" r="5" fill="#1A1A2E"/>
    <circle cx="56" cy="56" r="2.5" fill="white" opacity="0.8"/>
  </svg>
);

// Upload illustration
const HandImageSvg = () => (
  <svg className="upload-svg" viewBox="0 0 80 80" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
    <rect x="20" y="16" width="40" height="32" rx="4" fill="#F3F0FF" stroke="#6C3CE1" strokeWidth="1.5"/>
    <path d="M24 44l8-10 6 6 8-12 10 16H24z" fill="#E8734A" opacity="0.2"/>
    <circle cx="34" cy="26" r="3" fill="#E8734A" opacity="0.3"/>
    <circle cx="40" cy="60" r="6" stroke="#6C3CE1" strokeWidth="1" fill="none" opacity="0.3">
      <animate attributeName="r" values="6;12" dur="2s" repeatCount="indefinite"/>
      <animate attributeName="opacity" values="0.3;0" dur="2s" repeatCount="indefinite"/>
    </circle>
    <ellipse cx="40" cy="60" rx="4" ry="5" fill="#6C3CE1" opacity="0.15"/>
    <circle cx="40" cy="58" r="2.5" fill="#6C3CE1" opacity="0.4"/>
    <line x1="40" y1="48" x2="40" y2="55" stroke="#6C3CE1" strokeWidth="1" strokeDasharray="2 2" opacity="0.3"/>
  </svg>
);

// Braille loader
const BrailleLoader = () => (
  <div className="braille-loader" aria-hidden="true">
    <div className="braille-dot"></div>
    <div className="braille-dot"></div>
    <div className="braille-dot"></div>
    <div className="braille-dot"></div>
    <div className="braille-dot"></div>
    <div className="braille-dot"></div>
  </div>
);

// Audio waveform
const AudioWaveformSvg = () => (
  <svg width="48" height="24" viewBox="0 0 48 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" style={{ display: 'inline-block', verticalAlign: 'middle', marginRight: '0.5rem' }}>
    {[4, 12, 20, 28, 36, 44].map((x, i) => (
      <rect key={i} x={x - 1.5} y={8 - i % 3 * 2} width="3" rx="1.5" fill="#6C3CE1" opacity="0.5">
        <animate attributeName="height" values={`${8 + i * 2};${14 + (5 - i) * 2};${8 + i * 2}`} dur={`${1 + i * 0.15}s`} repeatCount="indefinite"/>
        <animate attributeName="y" values={`${8 - i % 3 * 2};${5 - (5 - i) % 3};${8 - i % 3 * 2}`} dur={`${1 + i * 0.15}s`} repeatCount="indefinite"/>
      </rect>
    ))}
  </svg>
);

// Detection overlay canvas — draws the uploaded image (contain-fit) plus
// bounding boxes reported by the capable model, scaled to the rendered size.
const DetectionOverlayCanvas = ({ src, detections = [] }) => {
  const canvasRef = useRef(null);
  const wrapRef = useRef(null);
  const [img, setImg] = useState(null);

  useEffect(() => {
    if (!src) return;
    const image = new window.Image();
    image.onload = () => setImg(image);
    image.src = src;
  }, [src]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap || !img) return;

    const draw = () => {
      const { width: cw, height: ch } = wrap.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = cw * dpr;
      canvas.height = ch * dpr;
      const ctx = canvas.getContext('2d');
      ctx.scale(dpr, dpr);

      // Contain-fit math: scale image to fit within (cw, ch)
      const scale = Math.min(cw / img.naturalWidth, ch / img.naturalHeight);
      const dw = img.naturalWidth * scale;
      const dh = img.naturalHeight * scale;
      const dx = (cw - dw) / 2;
      const dy = (ch - dh) / 2;
      ctx.clearRect(0, 0, cw, ch);
      ctx.drawImage(img, dx, dy, dw, dh);

      // Draw bounding boxes in pixel coords returned by the backend
      detections.forEach((det) => {
        const [x1, y1, x2, y2] = det.box || [];
        if (x1 == null || x2 == null) return;
        const rx = dx + x1 * scale;
        const ry = dy + y1 * scale;
        const rw = (x2 - x1) * scale;
        const rh = (y2 - y1) * scale;

        ctx.strokeStyle = '#E8734A';
        ctx.lineWidth = 2;
        ctx.strokeRect(rx, ry, rw, rh);

        ctx.fillStyle = 'rgba(232, 115, 74, 0.9)';
        ctx.font = '600 12px Inter, sans-serif';
        const label = det.label || 'object';
        const tw = ctx.measureText(label).width + 8;
        const ty = ry > 20 ? ry - 18 : ry;
        ctx.fillRect(rx, ty, tw, 18);
        ctx.fillStyle = '#FFFFFF';
        ctx.fillText(label, rx + 4, ty + 13);
      });
    };

    draw();
    window.addEventListener('resize', draw);
    return () => window.removeEventListener('resize', draw);
  }, [img, detections]);

  return (
    <div ref={wrapRef} className="detection-canvas-wrap" style={{ width: '100%', height: '100%', position: 'relative' }}>
      <canvas ref={canvasRef} className="detection-canvas" aria-label="Image preview with detected object bounding boxes" style={{ width: '100%', height: '100%', display: 'block' }} />
    </div>
  );
};

/* ============================================
   Main Page Component
   ============================================ */
export default function Home() {
  // Mode: 'upload' or 'camera'
  const [mode, setMode] = useState('upload');

  // AI Model Selection: 'combined', 'vit', 'blip'
  const [modelChoice, setModelChoice] = useState('combined');

  // Upload mode state
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [language, setLanguage] = useState('en');
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [announcement, setAnnouncement] = useState('');
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0);

  // VQA Chat state
  const [chatHistory, setChatHistory] = useState([]);
  const [chatQuestion, setChatQuestion] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [chatAudioSrc, setChatAudioSrc] = useState("");

  // Camera mode state
  const [cameraActive, setCameraActive] = useState(false);
  const [autoSpeak, setAutoSpeak] = useState(true);
  const [detectionInterval, setDetectionInterval] = useState(3000);
  const [cameraDetection, setCameraDetection] = useState(null);
  const [cameraLoading, setCameraLoading] = useState(false);

  // Refs for camera stability & avoiding closure staleness
  const fileInputRef = useRef(null);
  const audioPlayerRef = useRef(null);
  const capableAudioRef = useRef(null);
  const resultsRef = useRef(null);
  const captionRef = useRef(null);
  const chatAudioPlayerRef = useRef(null);
  const chatBottomRef = useRef(null);
  const chatInputRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const cameraStreamRef = useRef(null);
  const detectionTimerRef = useRef(null);
  const cameraAudioRef = useRef(null);

  // Mutable refs to prevent stale closure bugs in setInterval
  const cameraLoadingRef = useRef(false);
  const modelChoiceRef = useRef(modelChoice);
  const languageRef = useRef(language);
  const autoSpeakRef = useRef(autoSpeak);
  const detectionIntervalRef = useRef(detectionInterval);
  const lastSpokenRef = useRef("");

  useEffect(() => { modelChoiceRef.current = modelChoice; }, [modelChoice]);
  useEffect(() => { languageRef.current = language; }, [language]);
  useEffect(() => { autoSpeakRef.current = autoSpeak; }, [autoSpeak]);
  useEffect(() => { detectionIntervalRef.current = detectionInterval; }, [detectionInterval]);

  const MAX_SIZE_BYTES = 10 * 1024 * 1024; // 10MB
  const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

  // Sound Effects
  const playBeep = (type) => {
    try {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      if (!AudioContext) return;
      const ctx = new AudioContext();

      if (type === 'upload') {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.type = 'sine';
        osc.frequency.setValueAtTime(440, ctx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(880, ctx.currentTime + 0.15);
        gain.gain.setValueAtTime(0.15, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.15);
        osc.start();
        osc.stop(ctx.currentTime + 0.15);
      } else if (type === 'processing') {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(220, ctx.currentTime);
        gain.gain.setValueAtTime(0.05, ctx.currentTime);
        gain.gain.linearRampToValueAtTime(0.05, ctx.currentTime + 0.2);
        gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.4);
        osc.start();
        osc.stop(ctx.currentTime + 0.4);
      } else if (type === 'success') {
        const now = ctx.currentTime;
        [523.25, 659.25].forEach((freq) => {
          const osc = ctx.createOscillator();
          const gain = ctx.createGain();
          osc.connect(gain);
          gain.connect(ctx.destination);
          osc.type = 'sine';
          osc.frequency.setValueAtTime(freq, now);
          gain.gain.setValueAtTime(0.1, now);
          gain.gain.exponentialRampToValueAtTime(0.01, now + 0.3);
          osc.start();
          osc.stop(now + 0.3);
        });
      } else if (type === 'error') {
        const now = ctx.currentTime;
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(150, now);
        gain.gain.setValueAtTime(0.1, now);
        gain.gain.linearRampToValueAtTime(0.1, now + 0.1);
        gain.gain.setValueAtTime(0.0, now + 0.15);
        gain.gain.setValueAtTime(0.1, now + 0.2);
        gain.gain.exponentialRampToValueAtTime(0.01, now + 0.35);
        osc.start();
        osc.stop(now + 0.35);
      } else if (type === 'camera') {
        const now = ctx.currentTime;
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.type = 'sine';
        osc.frequency.setValueAtTime(600, now);
        osc.frequency.exponentialRampToValueAtTime(800, now + 0.1);
        gain.gain.setValueAtTime(0.08, now);
        gain.gain.exponentialRampToValueAtTime(0.01, now + 0.1);
        osc.start();
        osc.stop(now + 0.1);
      }
    } catch (e) {
      console.log("AudioContext failed:", e);
    }
  };

  useEffect(() => {
    return () => {
      if (imagePreviewUrl) {
        URL.revokeObjectURL(imagePreviewUrl);
      }
    };
  }, [imagePreviewUrl]);

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  // Global Keyboard Shortcuts
  useEffect(() => {
    const handleGlobalKeyDown = (e) => {
      const activeTag = document.activeElement?.tagName?.toLowerCase();
      if (activeTag === 'input' || activeTag === 'textarea' || activeTag === 'select') {
        if (e.key === 'Escape') {
          document.activeElement.blur();
        }
        return;
      }

      if (e.key.toLowerCase() === 'u' || (e.altKey && e.key.toLowerCase() === 'u')) {
        e.preventDefault();
        setMode('upload');
        stopCamera();
        setTimeout(() => fileInputRef.current?.click(), 100);
      }

      if (e.key.toLowerCase() === 'c' || (e.altKey && e.key.toLowerCase() === 'c')) {
        e.preventDefault();
        setMode('camera');
      }

      if (e.key.toLowerCase() === 'r' || (e.altKey && e.key.toLowerCase() === 'r')) {
        e.preventDefault();
        if (audioPlayerRef.current) {
          audioPlayerRef.current.currentTime = 0;
          audioPlayerRef.current.playbackRate = playbackSpeed;
          audioPlayerRef.current.play().catch(err => console.log(err));
        }
      }

      if (e.key.toLowerCase() === 'l' || (e.altKey && e.key.toLowerCase() === 'l')) {
        e.preventDefault();
        setLanguage((prev) => {
          if (prev === 'en') return 'hi';
          if (prev === 'hi') return 'mr';
          return 'en';
        });
      }

      if (e.key.toLowerCase() === 's' || (e.altKey && e.key.toLowerCase() === 's')) {
        e.preventDefault();
        chatInputRef.current?.focus();
      }
    };

    window.addEventListener('keydown', handleGlobalKeyDown);
    return () => {
      window.removeEventListener('keydown', handleGlobalKeyDown);
    };
  }, [playbackSpeed]);

  // ===== Upload Mode Handlers =====

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      fileInputRef.current.click();
    }
  };

  const processFile = (file) => {
    setError(null);
    setResults(null);
    setAnnouncement('');

    if (!ALLOWED_TYPES.includes(file.type)) {
      setError(`Unsupported file format (${file.type || 'unknown'}). Please upload a JPEG, PNG, or WEBP image.`);
      setSelectedFile(null);
      setImagePreviewUrl(null);
      return;
    }

    if (file.size > MAX_SIZE_BYTES) {
      const sizeInMB = (file.size / (1024 * 1024)).toFixed(1);
      setError(`The file is too large (${sizeInMB}MB). Maximum allowed size is 10MB.`);
      setSelectedFile(null);
      setImagePreviewUrl(null);
      return;
    }

    setSelectedFile(file);
    const objectUrl = URL.createObjectURL(file);
    setImagePreviewUrl(objectUrl);
    setChatHistory([]);
    setChatAudioSrc("");
    playBeep('upload');
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleGenerate = async () => {
    if (!selectedFile) return;

    setError(null);
    setResults(null);
    setLoading(true);
    playBeep('processing');

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('lang', language);
    formData.append('model_choice', modelChoice);

    try {
      const response = await fetch('/api/caption', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to process request.' }));
        throw new Error(errorData.detail || `Server returned status code ${response.status}`);
      }

      const data = await response.json();
      setResults(data);
      playBeep('success');

      const oursCls = data.ours?.classification?.available ? data.ours.classification.class_name : null;
      const capableCount = data.capable?.available ? (data.capable.detections?.length || 0) : 0;
      setAnnouncement(
        `Comparison ready. Ours detected ${oursCls || 'no object class'}. Capable model detected ${capableCount} object${capableCount === 1 ? '' : 's'}.`
      );

      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth' });
        captionRef.current?.focus();
        if (audioPlayerRef.current) {
          audioPlayerRef.current.playbackRate = playbackSpeed;
          audioPlayerRef.current.play().catch(() => {
            console.log("Autoplay blocked by browser.");
          });
        }
      }, 100);

    } catch (err) {
      console.error(err);
      setError(err.message || 'An unexpected error occurred. Please try again.');
      playBeep('error');
    } finally {
      setLoading(false);
    }
  };

  const handleRadioChange = (e) => {
    setLanguage(e.target.value);
  };

  const handleSpeedChange = (e) => {
    const speed = parseFloat(e.target.value);
    setPlaybackSpeed(speed);
    if (audioPlayerRef.current) {
      audioPlayerRef.current.playbackRate = speed;
    }
  };

  const handleChatSubmit = async (e) => {
    e.preventDefault();
    if (!chatQuestion.trim() || !selectedFile || chatLoading) return;

    const currentQuestion = chatQuestion.trim();
    setChatQuestion("");
    setChatLoading(true);
    setError(null);
    playBeep('processing');

    const newHistory = [...chatHistory, { sender: 'user', text: currentQuestion }];
    setChatHistory(newHistory);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('question', currentQuestion);
    formData.append('lang', language);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to process chat question.' }));
        throw new Error(errorData.detail || `Server returned status code ${response.status}`);
      }

      const data = await response.json();
      const answer = data.answer_translated || data.answer_en;
      const chatAudio = `data:audio/mp3;base64,${data.audio_base64}`;
      
      setChatHistory([...newHistory, { sender: 'ai', text: answer, audioSrc: chatAudio }]);
      setChatAudioSrc(chatAudio);
      playBeep('success');

      setTimeout(() => {
        if (chatAudioPlayerRef.current) {
          chatAudioPlayerRef.current.playbackRate = playbackSpeed;
          chatAudioPlayerRef.current.play().catch(() => {
            console.log("Autoplay blocked for chat narration.");
          });
        }
        chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
      }, 100);

    } catch (err) {
      console.error(err);
      setError(err.message || 'Failed to get answer. Please try again.');
      playBeep('error');
    } finally {
      setChatLoading(false);
    }
  };

  // ===== Camera Mode Handlers =====

  const captureFrame = () => {
    if (!videoRef.current || !canvasRef.current) return null;

    const video = videoRef.current;
    const canvas = canvasRef.current;

    // Check if video metadata is ready and dimensions exist
    if (video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0) {
      return null;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    return canvas.toDataURL('image/jpeg', 0.65);
  };

  const detectFromFrame = async () => {
    if (cameraLoadingRef.current) return;

    const frameData = captureFrame();
    if (!frameData) return;

    cameraLoadingRef.current = true;
    setCameraLoading(true);

    try {
      const formData = new FormData();
      formData.append('frame', frameData);
      formData.append('lang', languageRef.current);
      formData.append('model_choice', modelChoiceRef.current);

      const response = await fetch('/api/detect', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Detection HTTP Error (${response.status})`);
      }

      const data = await response.json();
      setCameraDetection(data);

      // Auto-speak if enabled and ours description changed
      const oursDesc = data.ours?.translated || data.ours?.description || '';
      if (autoSpeakRef.current && oursDesc && oursDesc !== lastSpokenRef.current) {
        lastSpokenRef.current = oursDesc;
        const audioSrc = `data:audio/mp3;base64,${data.ours?.audio_base64 || ''}`;
        if (cameraAudioRef.current && data.ours?.audio_base64) {
          cameraAudioRef.current.src = audioSrc;
          cameraAudioRef.current.playbackRate = playbackSpeed;
          cameraAudioRef.current.play().catch(() => {});
        }
      }

      playBeep('camera');
    } catch (err) {
      console.error("Live detection error:", err);
    } finally {
      cameraLoadingRef.current = false;
      setCameraLoading(false);
    }
  };

  const startDetectionLoop = useCallback(() => {
    if (detectionTimerRef.current) {
      clearInterval(detectionTimerRef.current);
    }

    // Trigger initial detection after video stream settles
    setTimeout(() => {
      detectFromFrame();
    }, 800);

    detectionTimerRef.current = setInterval(() => {
      detectFromFrame();
    }, detectionIntervalRef.current);
  }, []);

  const startCamera = async () => {
    setError(null);
    setCameraDetection(null);
    lastSpokenRef.current = "";

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });

      cameraStreamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();

        videoRef.current.onloadedmetadata = () => {
          startDetectionLoop();
        };
        if (videoRef.current.readyState >= 1) {
          startDetectionLoop();
        }
      } else {
        startDetectionLoop();
      }

      setCameraActive(true);
      playBeep('camera');
    } catch (err) {
      console.error("Camera access error:", err);
      if (err.name === 'NotAllowedError') {
        setError("Camera access was denied. Please grant camera permissions in your browser settings.");
      } else if (err.name === 'NotFoundError') {
        setError("No camera device was found.");
      } else {
        setError(`Could not access camera: ${err.message}`);
      }
      playBeep('error');
    }
  };

  const stopCamera = () => {
    if (detectionTimerRef.current) {
      clearInterval(detectionTimerRef.current);
      detectionTimerRef.current = null;
    }

    if (cameraStreamRef.current) {
      cameraStreamRef.current.getTracks().forEach(track => track.stop());
      cameraStreamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setCameraActive(false);
    setCameraLoading(false);
    cameraLoadingRef.current = false;
  };

  // Restart detection loop when interval changes
  useEffect(() => {
    if (cameraActive) {
      startDetectionLoop();
    }
    return () => {
      if (detectionTimerRef.current) {
        clearInterval(detectionTimerRef.current);
      }
    };
  }, [detectionInterval, cameraActive, startDetectionLoop]);

  const audioSrc = results?.ours?.audio_base64 ? `data:audio/mp3;base64,${results.ours.audio_base64}` : '';
  const capableAudioSrc = results?.capable?.audio_base64 ? `data:audio/mp3;base64,${results.capable.audio_base64}` : '';

  return (
    <>
      {/* ===== Header with Hero Illustration ===== */}
      <header className="app-header">
        <div className="header-container">
          <div className="hero-illustration">
            <EyeSoundWaveSvg />
          </div>
          <h1 id="app-title" className="brand-title">Aabha</h1>
          <p className="brand-subtitle">
            Transforming images into spoken narratives — empowering visually impaired users to perceive the world through sound.
          </p>

          {/* Mode Tabs */}
          <div className="mode-tabs">
            <button
              type="button"
              className={`mode-tab ${mode === 'upload' ? 'active' : ''}`}
              onClick={() => { setMode('upload'); stopCamera(); }}
              aria-label="Switch to image upload mode"
            >
              <Upload size={16} /> Upload Image
            </button>
            <button
              type="button"
              className={`mode-tab ${mode === 'camera' ? 'active' : ''}`}
              onClick={() => setMode('camera')}
              aria-label="Switch to live camera mode"
            >
              <Camera size={16} /> Live Camera
            </button>
          </div>

          <div className="shortcuts-hint">
            <span className="kbd">U Upload</span>
            <span className="kbd">C Camera</span>
            <span className="kbd">R Replay</span>
            <span className="kbd">L Language</span>
            <span className="kbd">S Chat</span>
          </div>
        </div>
      </header>

      <main className="main-container">

        {/* Model Selection Panel (Shared between modes) */}
        <section className="card" style={{ padding: '1.25rem 2.25rem' }}>
          <fieldset className="language-selector">
            <legend className="section-title">AI Vision Model / AI मॉडेल निवडा</legend>
            <div className="radio-group">
              <label className="radio-label" htmlFor="model-combined">
                <input 
                  type="radio" id="model-combined" name="modelChoice" value="combined" 
                  checked={modelChoice === 'combined'} onChange={(e) => setModelChoice(e.target.value)}
                />
                <span className="custom-radio"></span>
                <span className="label-text">Combined (Custom ViT + BLIP)</span>
              </label>
              <label className="radio-label" htmlFor="model-vit">
                <input 
                  type="radio" id="model-vit" name="modelChoice" value="vit" 
                  checked={modelChoice === 'vit'} onChange={(e) => setModelChoice(e.target.value)}
                />
                <span className="custom-radio"></span>
                <span className="label-text">Custom ViT Only (Classifier)</span>
              </label>
              <label className="radio-label" htmlFor="model-blip">
                <input 
                  type="radio" id="model-blip" name="modelChoice" value="blip" 
                  checked={modelChoice === 'blip'} onChange={(e) => setModelChoice(e.target.value)}
                />
                <span className="custom-radio"></span>
                <span className="label-text">BLIP Only (Captioner)</span>
              </label>
            </div>
          </fieldset>
        </section>

        {/* ===================================================================
            UPLOAD MODE
            =================================================================== */}
        {mode === 'upload' && (
          <>
            {/* Upload & Settings Card */}
            <section className="card" aria-labelledby="uploader-heading">
              <h2 id="uploader-heading" className="sr-only">Image Uploader and Settings</h2>
              
              <div 
                id="drop-zone" 
                className={`upload-area ${dragActive ? 'drag-over' : ''}`}
                tabIndex={0} 
                role="button" 
                aria-controls="file-input" 
                aria-describedby="upload-instructions"
                onKeyDown={handleKeyDown}
                onClick={() => fileInputRef.current.click()}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
              >
                <div className="upload-illustration">
                  <HandImageSvg />
                </div>
                
                <div id="upload-instructions" className="upload-text">
                  {selectedFile ? (
                    <>Selected: <span className="highlight">{selectedFile.name}</span></>
                  ) : (
                    <><span className="highlight">Drag & drop your image here</span> or <span className="browse-link">browse files</span></>
                  )}
                </div>
                
                <div className="upload-info">Supports JPEG, PNG, WEBP · Max 10MB</div>
                
                <input 
                  type="file" 
                  id="file-input" 
                  className="file-input" 
                  accept=".jpg,.jpeg,.png,.webp" 
                  ref={fileInputRef}
                  onChange={handleFileChange}
                  aria-label="Upload image file"
                />
              </div>

              {/* Language Selector */}
              <div className="controls-panel">
                <fieldset className="language-selector">
                  <legend className="section-title">Caption Language / भाषा निवडा</legend>
                  <div className="radio-group">
                    <label className="radio-label" htmlFor="lang-en">
                      <input 
                        type="radio" id="lang-en" name="language" value="en" 
                        checked={language === 'en'} onChange={handleRadioChange}
                      />
                      <span className="custom-radio"></span>
                      <span className="label-text">English</span>
                    </label>
                    <label className="radio-label" htmlFor="lang-hi">
                      <input 
                        type="radio" id="lang-hi" name="language" value="hi" 
                        checked={language === 'hi'} onChange={handleRadioChange}
                      />
                      <span className="custom-radio"></span>
                      <span className="label-text">Hindi / हिंदी</span>
                    </label>
                    <label className="radio-label" htmlFor="lang-mr">
                      <input 
                        type="radio" id="lang-mr" name="language" value="mr" 
                        checked={language === 'mr'} onChange={handleRadioChange}
                      />
                      <span className="custom-radio"></span>
                      <span className="label-text">Marathi / मराठी</span>
                    </label>
                  </div>
                </fieldset>
              </div>

              {/* Error */}
              {error && (
                <div id="error-message" className="error-container" role="alert" aria-live="assertive">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <AlertCircle size={18} />
                    <span>{error}</span>
                  </div>
                </div>
              )}

              {/* Generate Button */}
              <div className="action-bar">
                <button 
                  type="button" id="btn-generate" className="btn-primary" 
                  disabled={!selectedFile || loading} onClick={handleGenerate}
                  aria-describedby={!selectedFile ? "btn-generate-desc" : undefined}
                >
                  {loading ? 'Analyzing...' : 'Generate Caption & Audio'}
                </button>
                {!selectedFile && (
                  <div id="btn-generate-desc" className="sr-only">Upload an image to enable this button.</div>
                )}
              </div>
            </section>

            {/* Loading — Braille Dots */}
            {loading && (
              <div id="loading-spinner" className="spinner-container" role="status" aria-live="polite">
                <BrailleLoader />
                <p id="loading-text" className="loading-message">Reading your image...</p>
              </div>
            )}

            {/* Results — OURS vs CAPABLE comparison */}
            {results && (
              <section id="results-section" ref={resultsRef} className="card results-fade-in" aria-labelledby="results-heading">
                <h2 id="results-heading" className="section-title" style={{ display: 'flex', alignItems: 'center', marginBottom: '1.5rem' }}>
                  <AudioWaveformSvg />
                  Model Comparison — Ours vs Capable
                </h2>
                <p id="results-live" className="sr-only" role="status" aria-live="polite">{announcement}</p>

                <div className="results-grid">
                  {/* Image Preview with capable-model bounding boxes */}
                  <div className="preview-panel">
                    <h3 className="panel-subtitle">Uploaded Image</h3>
                    <div className="image-preview-container">
                      {imagePreviewUrl && (
                        <DetectionOverlayCanvas
                          src={imagePreviewUrl}
                          detections={results.capable?.detections || []}
                        />
                      )}
                    </div>
                    {results.capable?.available && results.capable?.detections?.length > 0 && (
                      <p className="box-detect-note">
                        <Boxes size={14} aria-hidden="true" /> Boxes drawn by Capable (Florence-2). Yours reports a single class.
                      </p>
                    )}
                  </div>

                  {/* Comparison panels */}
                  <div className="output-panel">
                    {/* ===== OURS Panel ===== */}
                    <div className="compare-panel compare-panel-ours">
                      <div className="compare-panel-header">
                        <div className="compare-panel-title">
                          <Eye size={18} aria-hidden="true" />
                          <h3 className="panel-subtitle" style={{ margin: 0 }}>Ours · Custom ViT + BLIP</h3>
                        </div>
                        <span className="compare-badge">fixed 10-class</span>
                      </div>

                      {modelChoice !== 'blip' && results.ours?.classification && (
                        <div className="classification-result">
                          <div className="classification-icon">
                            <Eye size={20} />
                          </div>
                          <div className="classification-details">
                            <div className="classification-label">Custom ViT Detection</div>
                            {results.ours.classification.available ? (
                              <>
                                <div className="classification-name">{results.ours.classification.class_name}</div>
                                <div className="classification-confidence">
                                  {(results.ours.classification.confidence * 100).toFixed(1)}% confidence
                                </div>
                              </>
                            ) : (
                              <div className="classification-unavailable">
                                ViT model not trained yet — run train.py to enable
                              </div>
                            )}
                          </div>
                        </div>
                      )}

                      <div className="caption-container">
                        {results.ours?.caption_en && results.ours?.description !== results.ours?.caption_en && (
                          <div className="caption-block">
                            <div className="caption-header">
                              <span className="lang-tag">Scene caption</span>
                            </div>
                            <p className="caption-text" tabIndex={0} aria-label="Scene caption">
                              {results.ours.caption_en}
                            </p>
                          </div>
                        )}
                        <div className="caption-block">
                          <div className="caption-header">
                            <span className="lang-tag">Narration {results.ours?.translated ? '(answer translated)' : '(English)'}</span>
                          </div>
                          <p
                            id="text-ours" className="caption-text" tabIndex={0}
                            ref={captionRef} aria-label="Narration text"
                          >
                            {results.ours?.translated || results.ours?.description}
                          </p>
                        </div>
                      </div>

                      <div className="audio-container">
                        <h3 className="panel-subtitle">Audio Narration</h3>
                        <div className="audio-wrapper">
                          <audio
                            id="audio-player" className="native-audio" controls src={audioSrc}
                            ref={audioPlayerRef}
                            onPlay={(e) => { e.target.playbackRate = playbackSpeed; }}
                            onCanPlay={(e) => { e.target.playbackRate = playbackSpeed; }}
                            aria-label={`Spoken narration: ${results.ours?.translated || results.ours?.description}`}
                          />
                          <div className="speed-control-container">
                            <label htmlFor="playback-speed-select" className="sr-only">Playback Speed</label>
                            <select
                              id="playback-speed-select" className="speed-select"
                              value={playbackSpeed} onChange={handleSpeedChange}
                              aria-label="Select playback speed"
                            >
                              <option value="0.75">0.75×</option>
                              <option value="1.0">1.0× Normal</option>
                              <option value="1.25">1.25×</option>
                              <option value="1.5">1.5×</option>
                              <option value="2.0">2.0×</option>
                            </select>
                          </div>
                          <a
                            id="btn-download" href={audioSrc} className="btn-secondary"
                            download={`caption_${selectedFile?.name?.split('.')[0] || 'narration'}.mp3`}
                            aria-label="Download audio narration as MP3 file"
                          >
                            <Download size={18} className="btn-icon" aria-hidden="true" />
                            Download
                          </a>
                        </div>
                      </div>
                    </div>

                    {/* ===== CAPABLE Panel ===== */}
                    <div className={`compare-panel compare-panel-capable ${results.capable?.available ? '' : 'is-muted'}`}>
                      <div className="compare-panel-header">
                        <div className="compare-panel-title">
                          <Sparkles size={18} aria-hidden="true" />
                          <h3 className="panel-subtitle" style={{ margin: 0 }}>Capable · Florence-2</h3>
                        </div>
                        <span className="compare-badge">open-vocabulary</span>
                      </div>

                      {results.capable?.available ? (
                        <>
                          {results.capable.detections?.length > 0 && (
                            <div className="detection-chips">
                              <span className="detection-chips-title">Detected objects</span>
                              <div className="chip-list">
                                {results.capable.detections.map((d, i) => (
                                  <span key={i} className="detection-chip">{d.label}</span>
                                ))}
                              </div>
                            </div>
                          )}

                          <div className="caption-block">
                            <div className="caption-header">
                              <span className="lang-tag">Detailed caption</span>
                            </div>
                            <p className="caption-text" tabIndex={0} aria-label="Florence-2 detailed caption">
                              {results.capable.caption || 'No caption generated.'}
                            </p>
                          </div>

                          {results.capable?.translated && (
                            <div className="caption-block">
                              <div className="caption-header">
                                <span className="lang-tag">Translated</span>
                              </div>
                              <p className="caption-text" tabIndex={0} aria-label="Translated capable caption">
                                {results.capable.translated}
                              </p>
                            </div>
                          )}

                          {capableAudioSrc && (
                            <div className="audio-container">
                              <h3 className="panel-subtitle">Audio Narration</h3>
                              <div className="audio-wrapper">
                                <audio
                                  className="native-audio" controls src={capableAudioSrc}
                                  ref={capableAudioRef}
                                  onPlay={(e) => { e.target.playbackRate = playbackSpeed; }}
                                  onCanPlay={(e) => { e.target.playbackRate = playbackSpeed; }}
                                  aria-label={`Spoken capable narration: ${results.capable.translated || results.capable.caption}`}
                                />
                                <a
                                  href={capableAudioSrc} className="btn-secondary"
                                  download={`capable_${selectedFile?.name?.split('.')[0] || 'narration'}.mp3`}
                                  aria-label="Download capable audio narration as MP3 file"
                                >
                                  <Download size={18} className="btn-icon" aria-hidden="true" />
                                  Download
                                </a>
                              </div>
                            </div>
                          )}
                        </>
                      ) : (
                        <div className="classification-unavailable">
                          Capable model not loaded — backend will run in ours-only mode.
                          {results.capable?.error && <div className="caption-text" style={{ marginTop: '0.5rem' }}>{results.capable.error}</div>}
                        </div>
                      )}
                    </div>

                    {/* VQA Chat */}
                    <div className="chat-interface-card">
                      <h3 className="panel-subtitle">Ask about this image / सवाल पूछें</h3>

                      <div className="chat-messages-box" aria-live="polite">
                        {chatHistory.length === 0 ? (
                          <p className="chat-placeholder">Ask anything — e.g., &quot;What color is the car?&quot; or &quot;How many people?&quot;</p>
                        ) : (
                          chatHistory.map((msg, idx) => (
                            <div key={idx} className={`chat-message ${msg.sender}-message`}>
                              <div className="message-content">
                                <span className="message-sender-label">{msg.sender === 'user' ? 'You' : 'Aabha'}</span>
                                <p className="message-text">{msg.text}</p>
                              </div>
                              {msg.audioSrc && (
                                <button
                                  type="button" className="chat-audio-btn" aria-label="Replay audio answer"
                                  onClick={() => {
                                    setChatAudioSrc(msg.audioSrc);
                                    setTimeout(() => {
                                      if (chatAudioPlayerRef.current) {
                                        chatAudioPlayerRef.current.playbackRate = playbackSpeed;
                                        chatAudioPlayerRef.current.play().catch(err => console.log(err));
                                      }
                                    }, 50);
                                  }}
                                >
                                  <Volume2 size={14} />
                                </button>
                              )}
                            </div>
                          ))
                        )}
                        <div ref={chatBottomRef} />
                      </div>

                      <form onSubmit={handleChatSubmit} className="chat-input-form">
                        <input
                          type="text" id="chat-question-input" ref={chatInputRef}
                          className="chat-text-input" value={chatQuestion}
                          onChange={(e) => setChatQuestion(e.target.value)}
                          placeholder="Ask a question about the image..."
                          disabled={chatLoading} aria-label="Type your question about the image"
                        />
                        <button
                          type="submit" className="btn-chat-submit"
                          disabled={!chatQuestion.trim() || chatLoading}
                        >
                          {chatLoading ? 'Thinking...' : 'Ask'}
                        </button>
                      </form>

                      <audio ref={chatAudioPlayerRef} src={chatAudioSrc} style={{ display: 'none' }} />
                    </div>
                  </div>
                </div>
              </section>
            )}
          </>
        )}

        {/* ===================================================================
            CAMERA MODE
            =================================================================== */}
        {mode === 'camera' && (
          <section className="card" aria-labelledby="camera-heading">
            <h2 id="camera-heading" className="section-title" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem' }}>
              <Camera size={22} />
              Live Camera Detection
            </h2>

            <div className="camera-section">
              {/* Camera Feed */}
              <div className="camera-feed-container">
                {cameraActive ? (
                  <>
                    <video 
                      ref={videoRef} 
                      className="camera-feed mirror" 
                      autoPlay 
                      playsInline 
                      muted
                      aria-label="Live camera feed"
                    />
                    {/* Detection overlay on video */}
                    {cameraDetection && (
                      <div className="detection-overlay" aria-live="polite">
                        <div className="detection-overlay-icon">
                          <Eye size={16} />
                        </div>
                        <div className="detection-overlay-text">
                          {cameraDetection.ours?.classification?.available && cameraDetection.ours.classification.class_name ? (
                            <>
                              <div className="detection-overlay-class">
                                {cameraDetection.ours.classification.class_name}
                                <span style={{ opacity: 0.7, fontSize: '0.8em', marginLeft: '0.5rem' }}>
                                  {(cameraDetection.ours.classification.confidence * 100).toFixed(0)}%
                                </span>
                              </div>
                              <div className="detection-overlay-caption">
                                {cameraDetection.ours?.translated || cameraDetection.ours?.description}
                              </div>
                            </>
                          ) : (
                            <div className="detection-overlay-caption">
                              {cameraDetection.ours?.translated || cameraDetection.ours?.description}
                            </div>
                          )}
                          {cameraDetection.capable?.available && cameraDetection.capable?.detections?.length > 0 && (
                            <div className="detection-overlay-capable">
                              <Sparkles size={12} aria-hidden="true" />
                              {cameraDetection.capable.detections.slice(0, 4).map((d) => d.label).join(' · ')}
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="camera-placeholder">
                    <Camera size={64} className="camera-placeholder-icon" />
                    <p>Click &quot;Start Camera&quot; to begin live object detection. Your camera feed stays on your device.</p>
                  </div>
                )}
              </div>

              {/* Hidden canvas for frame capture */}
              <canvas ref={canvasRef} style={{ display: 'none' }} />

              {/* Camera Controls */}
              <div className="camera-controls">
                {!cameraActive ? (
                  <button 
                    type="button" 
                    className="btn-camera btn-camera-start"
                    onClick={startCamera}
                    aria-label="Start camera for live detection"
                  >
                    <Camera size={18} /> Start Camera
                  </button>
                ) : (
                  <>
                    <button 
                      type="button" 
                      className="btn-camera btn-camera-stop"
                      onClick={stopCamera}
                      aria-label="Stop camera"
                    >
                      Stop Camera
                    </button>
                    <button
                      type="button"
                      className="btn-secondary"
                      onClick={detectFromFrame}
                      disabled={cameraLoading}
                      aria-label="Trigger immediate detection"
                    >
                      <RefreshCw size={16} className={cameraLoading ? 'spin' : ''} /> Detect Now
                    </button>
                  </>
                )}

                {/* Auto-speak toggle */}
                <label className="auto-speak-toggle">
                  <div 
                    className={`toggle-switch ${autoSpeak ? 'active' : ''}`}
                    onClick={() => setAutoSpeak(!autoSpeak)}
                    role="switch"
                    aria-checked={autoSpeak}
                    tabIndex={0}
                    onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setAutoSpeak(!autoSpeak); }}}
                  />
                  Auto-speak
                </label>

                {/* Detection interval */}
                <select 
                  className="interval-select"
                  value={detectionInterval}
                  onChange={(e) => setDetectionInterval(parseInt(e.target.value))}
                  aria-label="Detection interval"
                >
                  <option value="2000">Every 2s</option>
                  <option value="3000">Every 3s</option>
                  <option value="5000">Every 5s</option>
                  <option value="10000">Every 10s</option>
                </select>

                {/* Language selector for camera */}
                <select 
                  className="interval-select"
                  value={language}
                  onChange={(e) => setLanguage(e.target.value)}
                  aria-label="Detection language"
                >
                  <option value="en">English</option>
                  <option value="hi">Hindi</option>
                  <option value="mr">Marathi</option>
                </select>

                {/* Status indicator */}
                <div className="camera-status">
                  <div className={`status-dot ${cameraActive ? 'active' : ''}`} />
                  {cameraActive ? (cameraLoading ? 'Detecting...' : 'Active') : 'Inactive'}
                </div>
              </div>

              {/* Error */}
              {error && (
                <div className="error-container" role="alert" aria-live="assertive">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <AlertCircle size={18} />
                    <span>{error}</span>
                  </div>
                </div>
              )}

              {/* Detection Results Panel */}
              {cameraDetection && (
                <div className="camera-results results-fade-in">
                  <div className="camera-compare-grid">
                    {/* OURS */}
                    <div className="compare-panel compare-panel-ours">
                      <div className="compare-panel-header">
                        <div className="compare-panel-title">
                          <Eye size={16} aria-hidden="true" />
                          <span className="compare-panel-name">Ours · ViT + BLIP</span>
                        </div>
                      </div>
                      {cameraDetection.ours?.classification?.available && modelChoice !== 'blip' && (
                        <div className="classification-result">
                          <div className="classification-icon">
                            <Eye size={20} />
                          </div>
                          <div className="classification-details">
                            <div className="classification-label">Custom ViT Detection</div>
                            <div className="classification-name">{cameraDetection.ours.classification.class_name}</div>
                            <div className="classification-confidence">
                              {(cameraDetection.ours.classification.confidence * 100).toFixed(1)}% confidence
                              {cameraDetection.ours.classification.top3 && cameraDetection.ours.classification.top3.length > 1 && (
                                <span style={{ marginLeft: '0.75rem', opacity: 0.7 }}>
                                  Also: {cameraDetection.ours.classification.top3.slice(1).map(p =>
                                    `${p.class_name} ${(p.confidence * 100).toFixed(0)}%`
                                  ).join(', ')}
                                </span>
                              )}
                            </div>
                          </div>
                        </div>
                      )}
                      <div className="caption-block">
                        <div className="caption-header">
                          <span className="lang-tag">Narration</span>
                        </div>
                        <p className="caption-text" tabIndex={0} aria-label="Ours narration">
                          {cameraDetection.ours?.translated || cameraDetection.ours?.description}
                        </p>
                      </div>
                    </div>

                    {/* CAPABLE */}
                    <div className={`compare-panel compare-panel-capable ${cameraDetection.capable?.available ? '' : 'is-muted'}`}>
                      <div className="compare-panel-header">
                        <div className="compare-panel-title">
                          <Sparkles size={16} aria-hidden="true" />
                          <span className="compare-panel-name">Capable · Florence-2</span>
                        </div>
                      </div>
                      {cameraDetection.capable?.available ? (
                        <>
                          {cameraDetection.capable.detections?.length > 0 && (
                            <div className="detection-chips">
                              <span className="detection-chips-title">Detected objects</span>
                              <div className="chip-list">
                                {cameraDetection.capable.detections.slice(0, 8).map((d, i) => (
                                  <span key={i} className="detection-chip">{d.label}</span>
                                ))}
                              </div>
                            </div>
                          )}
                          <div className="caption-block">
                            <div className="caption-header">
                              <span className="lang-tag">Detailed caption</span>
                            </div>
                            <p className="caption-text" tabIndex={0} aria-label="Capable detailed caption">
                              {cameraDetection.capable.caption || 'No caption generated.'}
                            </p>
                          </div>
                        </>
                      ) : (
                        <div className="classification-unavailable">
                          Capable model not loaded — ours-only mode.
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* Hidden audio element for camera narration */}
              <audio ref={cameraAudioRef} style={{ display: 'none' }} />
            </div>
          </section>
        )}
      </main>

      {/* ===== Footer with Braille Decoration ===== */}
      <footer className="app-footer">
        <div className="footer-container">
          <div className="footer-braille-art" aria-hidden="true">
            {[1,0,1,0,1,1,0,1,0,1,1,0].map((filled, i) => (
              <div key={i} className="dot" style={{ opacity: filled ? 1 : 0.2 }}></div>
            ))}
          </div>
          <p>&copy; 2026 Aabha. Designed with accessibility at the core.</p>
        </div>
      </footer>
    </>
  );
}
