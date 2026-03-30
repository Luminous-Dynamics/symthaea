// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
'use client';

import { useState, useRef, useCallback } from 'react';
import { useMutation } from '@tanstack/react-query';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/hooks/useAuth';
import { Sidebar } from '@/components/layout/Sidebar';
import { Header } from '@/components/layout/Header';
import { Player } from '@/components/player/Player';
import { cn, formatDuration } from '@/lib/utils';
import {
  Upload,
  Music2,
  Image as ImageIcon,
  FileAudio,
  Check,
  ChevronRight,
  ChevronLeft,
  X,
  Loader2,
  AlertCircle,
  Play,
  Pause,
} from 'lucide-react';
import Image from 'next/image';
import { redirect } from 'next/navigation';

type UploadStep = 'file' | 'metadata' | 'artwork' | 'review';

interface UploadState {
  audioFile: File | null;
  audioUrl: string | null;
  audioDuration: number;
  coverFile: File | null;
  coverPreview: string | null;
  title: string;
  genre: string;
  description: string;
  lyrics: string;
  tags: string[];
  isExplicit: boolean;
  releaseDate: string;
}

const GENRES = [
  'Electronic', 'Hip Hop', 'Pop', 'Rock', 'Jazz', 'Classical',
  'R&B', 'Country', 'Ambient', 'Lo-Fi', 'House', 'Techno',
  'Drum & Bass', 'Dubstep', 'Indie', 'Metal', 'Folk', 'Latin',
];

const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100MB
const ALLOWED_AUDIO_TYPES = ['audio/mpeg', 'audio/wav', 'audio/flac', 'audio/aac', 'audio/ogg'];
const ALLOWED_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

export default function UploadPage() {
  const router = useRouter();
  const { authenticated } = useAuth();
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPreviewPlaying, setIsPreviewPlaying] = useState(false);

  const [step, setStep] = useState<UploadStep>('file');
  const [uploadState, setUploadState] = useState<UploadState>({
    audioFile: null,
    audioUrl: null,
    audioDuration: 0,
    coverFile: null,
    coverPreview: null,
    title: '',
    genre: '',
    description: '',
    lyrics: '',
    tags: [],
    isExplicit: false,
    releaseDate: new Date().toISOString().split('T')[0],
  });
  const [tagInput, setTagInput] = useState('');
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [uploadProgress, setUploadProgress] = useState(0);

  if (!authenticated) {
    redirect('/');
  }

  const steps: { id: UploadStep; label: string; icon: React.ReactNode }[] = [
    { id: 'file', label: 'Audio File', icon: <FileAudio className="w-5 h-5" /> },
    { id: 'metadata', label: 'Details', icon: <Music2 className="w-5 h-5" /> },
    { id: 'artwork', label: 'Artwork', icon: <ImageIcon className="w-5 h-5" /> },
    { id: 'review', label: 'Review', icon: <Check className="w-5 h-5" /> },
  ];

  const currentStepIndex = steps.findIndex((s) => s.id === step);

  const uploadMutation = useMutation({
    mutationFn: async () => {
      const formData = new FormData();
      formData.append('audio', uploadState.audioFile!);
      if (uploadState.coverFile) {
        formData.append('cover', uploadState.coverFile);
      }
      formData.append('title', uploadState.title);
      formData.append('genre', uploadState.genre);
      formData.append('description', uploadState.description);
      formData.append('lyrics', uploadState.lyrics);
      formData.append('tags', JSON.stringify(uploadState.tags));
      formData.append('isExplicit', String(uploadState.isExplicit));
      formData.append('releaseDate', uploadState.releaseDate);
      formData.append('duration', String(uploadState.audioDuration));

      const response = await fetch('/api/v2/songs/upload', {
        method: 'POST',
        body: formData,
        // Note: Don't set Content-Type header - browser will set it with boundary
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Upload failed');
      }

      return response.json();
    },
    onSuccess: (data) => {
      router.push(`/dashboard/songs?uploaded=${data.songId}`);
    },
  });

  const handleAudioDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleAudioSelect(file);
  }, []);

  const handleAudioSelect = (file: File) => {
    setErrors({});

    if (!ALLOWED_AUDIO_TYPES.includes(file.type)) {
      setErrors({ audio: 'Please upload an MP3, WAV, FLAC, AAC, or OGG file' });
      return;
    }

    if (file.size > MAX_FILE_SIZE) {
      setErrors({ audio: 'File size must be under 100MB' });
      return;
    }

    const url = URL.createObjectURL(file);
    const audio = new Audio(url);

    audio.addEventListener('loadedmetadata', () => {
      setUploadState((prev) => ({
        ...prev,
        audioFile: file,
        audioUrl: url,
        audioDuration: audio.duration,
        title: prev.title || file.name.replace(/\.[^/.]+$/, '').replace(/[-_]/g, ' '),
      }));
    });

    audio.addEventListener('error', () => {
      setErrors({ audio: 'Could not read audio file. Please try another file.' });
      URL.revokeObjectURL(url);
    });
  };

  const handleCoverDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleCoverSelect(file);
  }, []);

  const handleCoverSelect = (file: File) => {
    setErrors((prev) => ({ ...prev, cover: '' }));

    if (!ALLOWED_IMAGE_TYPES.includes(file.type)) {
      setErrors((prev) => ({ ...prev, cover: 'Please upload a JPG, PNG, or WebP image' }));
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setErrors((prev) => ({ ...prev, cover: 'Image must be under 10MB' }));
      return;
    }

    const url = URL.createObjectURL(file);
    setUploadState((prev) => ({
      ...prev,
      coverFile: file,
      coverPreview: url,
    }));
  };

  const addTag = () => {
    const tag = tagInput.trim().toLowerCase();
    if (tag && !uploadState.tags.includes(tag) && uploadState.tags.length < 10) {
      setUploadState((prev) => ({
        ...prev,
        tags: [...prev.tags, tag],
      }));
      setTagInput('');
    }
  };

  const removeTag = (tag: string) => {
    setUploadState((prev) => ({
      ...prev,
      tags: prev.tags.filter((t) => t !== tag),
    }));
  };

  const validateStep = (): boolean => {
    const newErrors: Record<string, string> = {};

    if (step === 'file') {
      if (!uploadState.audioFile) {
        newErrors.audio = 'Please select an audio file';
      }
    } else if (step === 'metadata') {
      if (!uploadState.title.trim()) {
        newErrors.title = 'Title is required';
      }
      if (!uploadState.genre) {
        newErrors.genre = 'Please select a genre';
      }
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const nextStep = () => {
    if (validateStep()) {
      const nextIndex = currentStepIndex + 1;
      if (nextIndex < steps.length) {
        setStep(steps[nextIndex].id);
      }
    }
  };

  const prevStep = () => {
    const prevIndex = currentStepIndex - 1;
    if (prevIndex >= 0) {
      setStep(steps[prevIndex].id);
    }
  };

  const togglePreview = () => {
    if (!audioRef.current) return;
    if (isPreviewPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
    setIsPreviewPlaying(!isPreviewPlaying);
  };

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />

      <main className="ml-64 pb-24">
        <Header />

        <div className="max-w-3xl mx-auto px-6 py-8">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold mb-2">Upload Track</h1>
            <p className="text-muted-foreground">
              Share your music with the world
            </p>
          </div>

          {/* Progress Steps */}
          <div className="flex items-center justify-between mb-12">
            {steps.map((s, index) => (
              <div key={s.id} className="flex items-center">
                <div
                  className={cn(
                    'flex items-center gap-2 px-4 py-2 rounded-full transition-colors',
                    index < currentStepIndex
                      ? 'bg-primary/20 text-primary'
                      : index === currentStepIndex
                      ? 'bg-primary text-black'
                      : 'bg-white/10 text-muted-foreground'
                  )}
                >
                  {index < currentStepIndex ? (
                    <Check className="w-5 h-5" />
                  ) : (
                    s.icon
                  )}
                  <span className="text-sm font-medium">{s.label}</span>
                </div>
                {index < steps.length - 1 && (
                  <div
                    className={cn(
                      'w-12 h-0.5 mx-2',
                      index < currentStepIndex ? 'bg-primary' : 'bg-white/10'
                    )}
                  />
                )}
              </div>
            ))}
          </div>

          {/* Step Content */}
          <div className="bg-white/5 rounded-xl p-8">
            {/* Step 1: Audio File */}
            {step === 'file' && (
              <div>
                <h2 className="text-xl font-semibold mb-6">Upload Audio File</h2>

                {!uploadState.audioFile ? (
                  <div
                    onDrop={handleAudioDrop}
                    onDragOver={(e) => e.preventDefault()}
                    className={cn(
                      'border-2 border-dashed rounded-xl p-12 text-center transition-colors',
                      errors.audio
                        ? 'border-red-500 bg-red-500/10'
                        : 'border-white/20 hover:border-primary/50 hover:bg-white/5'
                    )}
                  >
                    <input
                      type="file"
                      accept={ALLOWED_AUDIO_TYPES.join(',')}
                      onChange={(e) => e.target.files?.[0] && handleAudioSelect(e.target.files[0])}
                      className="hidden"
                      id="audio-upload"
                    />
                    <label htmlFor="audio-upload" className="cursor-pointer">
                      <Upload className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
                      <p className="text-lg font-medium mb-2">
                        Drop your audio file here
                      </p>
                      <p className="text-sm text-muted-foreground mb-4">
                        or click to browse
                      </p>
                      <p className="text-xs text-muted-foreground">
                        MP3, WAV, FLAC, AAC, OGG • Max 100MB
                      </p>
                    </label>
                  </div>
                ) : (
                  <div className="p-6 bg-white/5 rounded-xl">
                    <div className="flex items-center gap-4">
                      <div className="w-16 h-16 rounded-lg bg-primary/20 flex items-center justify-center">
                        <FileAudio className="w-8 h-8 text-primary" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="font-medium truncate">{uploadState.audioFile.name}</p>
                        <p className="text-sm text-muted-foreground">
                          {(uploadState.audioFile.size / (1024 * 1024)).toFixed(2)} MB •{' '}
                          {formatDuration(uploadState.audioDuration)}
                        </p>
                      </div>
                      <button
                        onClick={togglePreview}
                        className="w-12 h-12 rounded-full bg-primary flex items-center justify-center hover:scale-105 transition-transform"
                      >
                        {isPreviewPlaying ? (
                          <Pause className="w-5 h-5 text-black" />
                        ) : (
                          <Play className="w-5 h-5 text-black ml-0.5" />
                        )}
                      </button>
                      <button
                        onClick={() => {
                          if (uploadState.audioUrl) URL.revokeObjectURL(uploadState.audioUrl);
                          setUploadState((prev) => ({
                            ...prev,
                            audioFile: null,
                            audioUrl: null,
                            audioDuration: 0,
                          }));
                        }}
                        className="p-2 text-muted-foreground hover:text-red-500"
                      >
                        <X className="w-5 h-5" />
                      </button>
                    </div>
                    {uploadState.audioUrl && (
                      <audio
                        ref={audioRef}
                        src={uploadState.audioUrl}
                        onEnded={() => setIsPreviewPlaying(false)}
                      />
                    )}
                  </div>
                )}

                {errors.audio && (
                  <p className="mt-3 text-sm text-red-500 flex items-center gap-2">
                    <AlertCircle className="w-4 h-4" />
                    {errors.audio}
                  </p>
                )}
              </div>
            )}

            {/* Step 2: Metadata */}
            {step === 'metadata' && (
              <div className="space-y-6">
                <h2 className="text-xl font-semibold mb-6">Track Details</h2>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    Title <span className="text-red-500">*</span>
                  </label>
                  <input
                    type="text"
                    value={uploadState.title}
                    onChange={(e) => setUploadState((prev) => ({ ...prev, title: e.target.value }))}
                    className={cn(
                      'w-full px-4 py-3 bg-white/10 rounded-lg focus:outline-none focus:ring-2',
                      errors.title ? 'ring-2 ring-red-500' : 'focus:ring-primary'
                    )}
                    placeholder="Track title"
                  />
                  {errors.title && (
                    <p className="mt-1 text-sm text-red-500">{errors.title}</p>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    Genre <span className="text-red-500">*</span>
                  </label>
                  <div className="flex flex-wrap gap-2">
                    {GENRES.map((genre) => (
                      <button
                        key={genre}
                        type="button"
                        onClick={() => setUploadState((prev) => ({ ...prev, genre }))}
                        className={cn(
                          'px-4 py-2 rounded-full text-sm font-medium transition-colors',
                          uploadState.genre === genre
                            ? 'bg-primary text-black'
                            : 'bg-white/10 hover:bg-white/20'
                        )}
                      >
                        {genre}
                      </button>
                    ))}
                  </div>
                  {errors.genre && (
                    <p className="mt-2 text-sm text-red-500">{errors.genre}</p>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">Description</label>
                  <textarea
                    value={uploadState.description}
                    onChange={(e) => setUploadState((prev) => ({ ...prev, description: e.target.value }))}
                    className="w-full px-4 py-3 bg-white/10 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary resize-none h-24"
                    placeholder="Tell listeners about this track..."
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">Tags</label>
                  <div className="flex gap-2 mb-2">
                    <input
                      type="text"
                      value={tagInput}
                      onChange={(e) => setTagInput(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && (e.preventDefault(), addTag())}
                      className="flex-1 px-4 py-2 bg-white/10 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
                      placeholder="Add tags..."
                    />
                    <button
                      type="button"
                      onClick={addTag}
                      className="px-4 py-2 bg-white/10 rounded-lg hover:bg-white/20"
                    >
                      Add
                    </button>
                  </div>
                  {uploadState.tags.length > 0 && (
                    <div className="flex flex-wrap gap-2">
                      {uploadState.tags.map((tag) => (
                        <span
                          key={tag}
                          className="inline-flex items-center gap-1 px-3 py-1 bg-primary/20 text-primary rounded-full text-sm"
                        >
                          #{tag}
                          <button onClick={() => removeTag(tag)}>
                            <X className="w-3 h-3" />
                          </button>
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                <div className="grid grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium mb-2">Release Date</label>
                    <input
                      type="date"
                      value={uploadState.releaseDate}
                      onChange={(e) => setUploadState((prev) => ({ ...prev, releaseDate: e.target.value }))}
                      className="w-full px-4 py-3 bg-white/10 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium mb-2">Content Rating</label>
                    <button
                      type="button"
                      onClick={() => setUploadState((prev) => ({ ...prev, isExplicit: !prev.isExplicit }))}
                      className={cn(
                        'w-full px-4 py-3 rounded-lg text-left transition-colors',
                        uploadState.isExplicit
                          ? 'bg-red-500/20 text-red-400 border border-red-500/30'
                          : 'bg-white/10 hover:bg-white/20'
                      )}
                    >
                      {uploadState.isExplicit ? 'Explicit Content' : 'Clean / Non-Explicit'}
                    </button>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">Lyrics (optional)</label>
                  <textarea
                    value={uploadState.lyrics}
                    onChange={(e) => setUploadState((prev) => ({ ...prev, lyrics: e.target.value }))}
                    className="w-full px-4 py-3 bg-white/10 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary resize-none h-32 font-mono text-sm"
                    placeholder="Paste lyrics here..."
                  />
                </div>
              </div>
            )}

            {/* Step 3: Artwork */}
            {step === 'artwork' && (
              <div>
                <h2 className="text-xl font-semibold mb-6">Cover Artwork</h2>

                <div className="grid grid-cols-2 gap-8">
                  <div>
                    {!uploadState.coverPreview ? (
                      <div
                        onDrop={handleCoverDrop}
                        onDragOver={(e) => e.preventDefault()}
                        className={cn(
                          'aspect-square border-2 border-dashed rounded-xl flex flex-col items-center justify-center transition-colors',
                          errors.cover
                            ? 'border-red-500 bg-red-500/10'
                            : 'border-white/20 hover:border-primary/50 hover:bg-white/5'
                        )}
                      >
                        <input
                          type="file"
                          accept={ALLOWED_IMAGE_TYPES.join(',')}
                          onChange={(e) => e.target.files?.[0] && handleCoverSelect(e.target.files[0])}
                          className="hidden"
                          id="cover-upload"
                        />
                        <label htmlFor="cover-upload" className="cursor-pointer text-center p-8">
                          <ImageIcon className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
                          <p className="font-medium mb-2">Upload Cover Art</p>
                          <p className="text-sm text-muted-foreground">
                            JPG, PNG, WebP • Min 1000x1000px
                          </p>
                        </label>
                      </div>
                    ) : (
                      <div className="relative aspect-square rounded-xl overflow-hidden">
                        <Image
                          src={uploadState.coverPreview}
                          alt="Cover preview"
                          fill
                          className="object-cover"
                        />
                        <button
                          onClick={() => {
                            if (uploadState.coverPreview) URL.revokeObjectURL(uploadState.coverPreview);
                            setUploadState((prev) => ({
                              ...prev,
                              coverFile: null,
                              coverPreview: null,
                            }));
                          }}
                          className="absolute top-2 right-2 p-2 bg-black/60 rounded-full hover:bg-black/80"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      </div>
                    )}
                    {errors.cover && (
                      <p className="mt-3 text-sm text-red-500 flex items-center gap-2">
                        <AlertCircle className="w-4 h-4" />
                        {errors.cover}
                      </p>
                    )}
                  </div>

                  <div className="space-y-4">
                    <h3 className="font-medium">Artwork Guidelines</h3>
                    <ul className="space-y-2 text-sm text-muted-foreground">
                      <li className="flex items-start gap-2">
                        <Check className="w-4 h-4 text-green-500 mt-0.5" />
                        Square image (1:1 aspect ratio)
                      </li>
                      <li className="flex items-start gap-2">
                        <Check className="w-4 h-4 text-green-500 mt-0.5" />
                        Minimum 1000x1000 pixels
                      </li>
                      <li className="flex items-start gap-2">
                        <Check className="w-4 h-4 text-green-500 mt-0.5" />
                        JPG, PNG, or WebP format
                      </li>
                      <li className="flex items-start gap-2">
                        <Check className="w-4 h-4 text-green-500 mt-0.5" />
                        No explicit imagery unless marked
                      </li>
                      <li className="flex items-start gap-2">
                        <Check className="w-4 h-4 text-green-500 mt-0.5" />
                        You own rights to the image
                      </li>
                    </ul>

                    <p className="text-sm text-muted-foreground pt-4">
                      No artwork? A default cover will be generated based on your track's audio characteristics.
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Step 4: Review */}
            {step === 'review' && (
              <div>
                <h2 className="text-xl font-semibold mb-6">Review & Upload</h2>

                <div className="flex gap-6 mb-8">
                  {/* Cover Preview */}
                  <div className="w-48 h-48 rounded-lg overflow-hidden flex-shrink-0">
                    {uploadState.coverPreview ? (
                      <Image
                        src={uploadState.coverPreview}
                        alt="Cover"
                        width={192}
                        height={192}
                        className="object-cover w-full h-full"
                      />
                    ) : (
                      <div className="w-full h-full bg-gradient-to-br from-purple-600 to-fuchsia-700 flex items-center justify-center">
                        <Music2 className="w-16 h-16 text-white/60" />
                      </div>
                    )}
                  </div>

                  {/* Track Info */}
                  <div className="flex-1">
                    <h3 className="text-2xl font-bold mb-1">{uploadState.title}</h3>
                    <p className="text-muted-foreground mb-4">{uploadState.genre}</p>

                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-muted-foreground">Duration</p>
                        <p className="font-medium">{formatDuration(uploadState.audioDuration)}</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Release Date</p>
                        <p className="font-medium">{uploadState.releaseDate}</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Content</p>
                        <p className="font-medium">
                          {uploadState.isExplicit ? 'Explicit' : 'Clean'}
                        </p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">File</p>
                        <p className="font-medium truncate">{uploadState.audioFile?.name}</p>
                      </div>
                    </div>

                    {uploadState.tags.length > 0 && (
                      <div className="mt-4">
                        <p className="text-muted-foreground text-sm mb-2">Tags</p>
                        <div className="flex flex-wrap gap-1">
                          {uploadState.tags.map((tag) => (
                            <span
                              key={tag}
                              className="px-2 py-0.5 bg-white/10 rounded text-xs"
                            >
                              #{tag}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {uploadState.description && (
                  <div className="mb-6">
                    <p className="text-muted-foreground text-sm mb-2">Description</p>
                    <p className="text-sm">{uploadState.description}</p>
                  </div>
                )}

                {/* Upload Progress */}
                {uploadMutation.isPending && (
                  <div className="mb-6 p-4 bg-white/5 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium">Uploading...</span>
                      <span className="text-sm text-muted-foreground">{uploadProgress}%</span>
                    </div>
                    <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-primary transition-all duration-300"
                        style={{ width: `${uploadProgress}%` }}
                      />
                    </div>
                    <p className="text-xs text-muted-foreground mt-2">
                      Your track will be processed for streaming after upload
                    </p>
                  </div>
                )}

                {uploadMutation.isError && (
                  <div className="mb-6 p-4 bg-red-500/10 border border-red-500/20 rounded-lg flex items-start gap-3">
                    <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-medium text-red-400">Upload Failed</p>
                      <p className="text-sm text-muted-foreground">
                        {uploadMutation.error?.message || 'Please try again'}
                      </p>
                    </div>
                  </div>
                )}

                {/* Terms */}
                <div className="p-4 bg-white/5 rounded-lg text-sm text-muted-foreground">
                  <p>
                    By uploading, you confirm that you own or have the necessary rights to this content
                    and agree to our Terms of Service. Your track will be available for streaming after
                    processing is complete.
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Navigation */}
          <div className="flex items-center justify-between mt-8">
            <button
              onClick={prevStep}
              disabled={currentStepIndex === 0}
              className="flex items-center gap-2 px-6 py-3 bg-white/10 rounded-lg font-medium hover:bg-white/20 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ChevronLeft className="w-5 h-5" />
              Back
            </button>

            {step === 'review' ? (
              <button
                onClick={() => uploadMutation.mutate()}
                disabled={uploadMutation.isPending}
                className="flex items-center gap-2 px-8 py-3 bg-primary text-black rounded-lg font-medium hover:bg-primary/90 disabled:opacity-50"
              >
                {uploadMutation.isPending ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Uploading...
                  </>
                ) : (
                  <>
                    <Upload className="w-5 h-5" />
                    Upload Track
                  </>
                )}
              </button>
            ) : (
              <button
                onClick={nextStep}
                className="flex items-center gap-2 px-8 py-3 bg-primary text-black rounded-lg font-medium hover:bg-primary/90"
              >
                Continue
                <ChevronRight className="w-5 h-5" />
              </button>
            )}
          </div>
        </div>
      </main>

      <Player />
    </div>
  );
}
