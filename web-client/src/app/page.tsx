"use client"

import type React from "react"

import { useState, useCallback, useRef, useEffect } from "react"
import { Upload, X, FileImage, FileVideo, CloudUpload, Mic, Shirt, Sparkles, Grid3X3, ExternalLink } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card"
import { cn } from "@/lib/utils"

interface MediaFile {
  file: File
  url: string
  type: "image" | "video"
  duration?: number
}

interface Vibe {
  id: string
  name: string
  confidence: number
}

interface ContentAnalysis {
  audio_transcription: string | null
  clothing_description: string | null
  vibes: Vibe[]
}

interface Keyframe {
  filename: string
  timecode: string
}

interface FashionMatch {
  product_id: string
  score: number
  product_name: string
  title: string
  description: string
  product_type: string
  price: string
  tags: string
  collections: string
}

interface CroppedKeyframe {
  filename: string
  class_name: string
  original_class_name: string
  bbox: number[]
  crop_id: string
  imageUrl?: string
  fashion_matches?: FashionMatch[]
}

interface VideoAnalysis {
  video_id: string
  original_filename: string
  file_hash: string
  cached: boolean
  cached_from_video_id?: string
  scenes_detected: number
  keyframes_generated: number
  keyframes: Keyframe[]
  keyframes_url: string
  cropped_keyframes_generated: number
  cropped_keyframes: CroppedKeyframe[]
  cropped_keyframes_url: string
  masked_keyframes_generated: number
  masked_keyframes: any[]
  masked_keyframes_url: string
  content_analysis: ContentAnalysis
}

export default function MediaPreview() {
  const [mediaFile, setMediaFile] = useState<MediaFile | null>(null)
  const [videoAnalysis, setVideoAnalysis] = useState<VideoAnalysis | null>(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [isLoadingKeyframes, setIsLoadingKeyframes] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [keyframesLoaded, setKeyframesLoaded] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const acceptedTypes = ["image/png", "image/jpeg", "image/jpg", "video/mp4"]

  const validateFile = (file: File): boolean => {
    return acceptedTypes.includes(file.type)
  }

  const getFileType = (file: File): "image" | "video" => {
    return file.type.startsWith("image/") ? "image" : "video"
  }

  // Load keyframe images after analysis is received
  useEffect(() => {
    if (videoAnalysis?.cropped_keyframes && videoAnalysis.cropped_keyframes.length > 0 && !keyframesLoaded) {
      loadKeyframeImages()
    }
  }, [videoAnalysis, keyframesLoaded])

  const loadKeyframeImages = async () => {
    if (!videoAnalysis || keyframesLoaded) return

    console.log('Loading keyframe images for analysis:', videoAnalysis)
    console.log('Cropped keyframes:', videoAnalysis.cropped_keyframes)
    console.log('Cropped keyframes URL:', videoAnalysis.cropped_keyframes_url)

    setIsLoadingKeyframes(true)
    try {
      // Fetch simplified query data for fashion matches
      let simplifiedData = null
      try {
        const simplifiedResponse = await fetch(`http://localhost:8000/simplified_query?video_id=${videoAnalysis.video_id}`)
        if (simplifiedResponse.ok) {
          simplifiedData = await simplifiedResponse.json()
          console.log('Simplified query data:', simplifiedData)
        }
      } catch (error) {
        console.warn('Failed to fetch simplified query data:', error)
      }

      const updatedKeyframes = videoAnalysis.cropped_keyframes.map((keyframe) => {
        // The filename already contains the full filename like "headgear_1-keyframe_011.png"
        const imageUrl = `http://localhost:8000${videoAnalysis.cropped_keyframes_url}/${keyframe.filename}`
        console.log('Constructed image URL:', imageUrl)

        // Find fashion matches for this filename
        let fashion_matches: FashionMatch[] = []
        if (simplifiedData?.crop_matches) {
          const cropMatch = simplifiedData.crop_matches.find((match: any) => match.filename === keyframe.filename)
          if (cropMatch?.fashion_matches) {
            fashion_matches = cropMatch.fashion_matches
          }
        }

        return { ...keyframe, imageUrl, fashion_matches }
      })

      console.log('Updated keyframes with image URLs:', updatedKeyframes)

      setVideoAnalysis({
        ...videoAnalysis,
        cropped_keyframes: updatedKeyframes,
      })

      setKeyframesLoaded(true)
    } catch (error) {
      console.error("Error loading keyframe images:", error)
    } finally {
      setIsLoadingKeyframes(false)
    }
  }

  const processFile = useCallback(
    async (file: File) => {
      if (!validateFile(file)) return

      // Clean up previous file URL if it exists
      if (mediaFile) {
        URL.revokeObjectURL(mediaFile.url)
      }

      // Clear previous analysis
      setVideoAnalysis(null)
      setKeyframesLoaded(false)

      const url = URL.createObjectURL(file)
      const type = getFileType(file)

      // Check video duration if it's a video file
      if (type === "video") {
        try {
          const duration = await getVideoDuration(file)
          if (duration > 180) {
            // 3 minutes = 180 seconds
            URL.revokeObjectURL(url)
            alert("Video must be 3 minutes or less in duration")
            return
          }

          const newMediaFile: MediaFile = {
            file,
            url,
            type,
            duration,
          }
          setMediaFile(newMediaFile)
        } catch (error) {
          URL.revokeObjectURL(url)
          alert("Error reading video file")
          return
        }
      } else {
        const newMediaFile: MediaFile = {
          file,
          url,
          type,
        }
        setMediaFile(newMediaFile)
      }
    },
    [mediaFile],
  )

  const getVideoDuration = (file: File): Promise<number> => {
    return new Promise((resolve, reject) => {
      const video = document.createElement("video")
      video.preload = "metadata"

      video.onloadedmetadata = () => {
        window.URL.revokeObjectURL(video.src)
        resolve(video.duration)
      }

      video.onerror = () => {
        reject(new Error("Error loading video"))
      }

      video.src = URL.createObjectURL(file)
    })
  }

  const uploadFile = async () => {
    if (!mediaFile) return

    setIsUploading(true)
    setUploadError(null)
    setKeyframesLoaded(false)

    try {
      const formData = new FormData()
      formData.append("file", mediaFile.file)

      const response = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const result = await response.json()
      console.log("Upload response:", result)

      // Store the full video analysis
      setVideoAnalysis(result)

    } catch (error) {
      console.error("Upload error:", error)
      setUploadError(error instanceof Error ? error.message : "Upload failed")
    } finally {
      setIsUploading(false)
    }
  }

  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, "0")}`
  }

  const formatConfidence = (confidence: number): string => {
    return `${Math.round(confidence * 100)}%`
  }

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragOver(false)

      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        processFile(e.dataTransfer.files[0])
      }
    },
    [processFile],
  )

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }, [])

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files && e.target.files[0]) {
        processFile(e.target.files[0])
      }
    },
    [processFile],
  )

  const handlePaste = useCallback(
    async (e: React.ClipboardEvent) => {
      e.preventDefault()
      
      const items = Array.from(e.clipboardData.items)
      const imageItem = items.find(item => item.type.startsWith('image/'))
      
      if (imageItem) {
        const file = imageItem.getAsFile()
        if (file) {
          await processFile(file)
        }
      }
    },
    [processFile],
  )

  const clearFile = useCallback(() => {
    if (mediaFile) {
      URL.revokeObjectURL(mediaFile.url)
      setMediaFile(null)
      setVideoAnalysis(null)
      setKeyframesLoaded(false)
    }
  }, [mediaFile])

  // Group keyframes by class type
  const groupedKeyframes =
    videoAnalysis?.cropped_keyframes.reduce(
      (acc, keyframe) => {
        const category = keyframe.original_class_name
        if (!acc[category]) {
          acc[category] = []
        }
        acc[category].push(keyframe)
        return acc
      },
      {} as Record<string, CroppedKeyframe[]>,
    ) || {}

  // Get unique categories for tabs
  const categories = Object.keys(groupedKeyframes).sort()

  console.log('Grouped keyframes:', groupedKeyframes)
  console.log('Categories:', categories)

  return (
    <div className="min-h-screen bg-background p-4 md:p-8">
      <div className="mx-auto max-w-7xl space-y-6">
        <div className="text-center space-y-2">
          <h1 className="text-3xl font-bold text-foreground">Fashion Tagging Engine</h1>
          <p className="text-muted-foreground">Upload, drop, or paste an image (PNG, JPEG) or video (MP4) to preview</p>
        </div>

        {/* Preview and Analysis Panes */}
        {mediaFile && (
          <div className={cn("grid gap-6", videoAnalysis?.content_analysis ? "lg:grid-cols-2" : "grid-cols-1")}>
            {/* Preview Pane */}
            <Card className="border-border bg-card shadow-sm">
              <CardContent className="p-6">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 text-card-foreground">
                      {mediaFile.type === "image" ? (
                        <FileImage className="w-5 h-5" />
                      ) : (
                        <FileVideo className="w-5 h-5" />
                      )}
                      <span className="font-medium truncate">{mediaFile.file.name}</span>
                      <span className="text-sm text-muted-foreground">
                        ({(mediaFile.file.size / 1024 / 1024).toFixed(2)} MB
                        {mediaFile.duration && ` • ${formatDuration(mediaFile.duration)}`})
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Button
                        onClick={uploadFile}
                        disabled={isUploading}
                        size="sm"
                        className="bg-primary text-primary-foreground hover:bg-primary/90"
                      >
                        {isUploading ? (
                          <>
                            <div className="w-4 h-4 mr-1 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                            Uploading...
                          </>
                        ) : (
                          <>
                            <CloudUpload className="w-4 h-4 mr-1" />
                            Upload
                          </>
                        )}
                      </Button>
                      <Button
                        onClick={clearFile}
                        size="sm"
                        variant="outline"
                        className="border-border text-muted-foreground hover:bg-accent hover:text-accent-foreground"
                      >
                        <X className="w-4 h-4 mr-1" />
                        Remove
                      </Button>
                    </div>
                  </div>

                  {uploadError && (
                    <div className="p-3 bg-destructive/10 border border-destructive/20 rounded-lg">
                      <p className="text-sm text-destructive">Upload failed: {uploadError}</p>
                    </div>
                  )}

                  <div className="flex justify-center bg-muted rounded-lg p-4">
                    {mediaFile.type === "image" ? (
                      <img
                        src={mediaFile.url || "/placeholder.svg"}
                        alt={mediaFile.file.name}
                        className="max-w-full max-h-96 object-contain rounded-lg shadow-sm"
                      />
                    ) : (
                      <video
                        src={mediaFile.url}
                        controls
                        className="max-w-full max-h-96 object-contain rounded-lg shadow-sm"
                        preload="metadata"
                      />
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Content Analysis Pane */}
            {videoAnalysis?.content_analysis && (
              <Card className="border-border bg-card shadow-sm">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Sparkles className="w-5 h-5" />
                    Content Analysis
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* Audio Transcription */}
                  {videoAnalysis.content_analysis.audio_transcription && (
                    <div className="space-y-3">
                      <div className="flex items-center gap-2">
                        <Mic className="w-4 h-4 text-muted-foreground" />
                        <h3 className="font-semibold text-foreground">Audio Transcription</h3>
                      </div>
                      <div className="bg-muted rounded-lg p-4">
                        <p className="text-sm text-foreground leading-relaxed">
                          {videoAnalysis.content_analysis.audio_transcription}
                        </p>
                      </div>
                    </div>
                  )}

                  {/* Clothing Description */}
                  {videoAnalysis.content_analysis.clothing_description && (
                    <div className="space-y-3">
                      <div className="flex items-center gap-2">
                        <Shirt className="w-4 h-4 text-muted-foreground" />
                        <h3 className="font-semibold text-foreground">Clothing Description</h3>
                      </div>
                      <div className="bg-muted rounded-lg p-4">
                        <p className="text-sm text-foreground leading-relaxed">
                          {videoAnalysis.content_analysis.clothing_description}
                        </p>
                      </div>
                    </div>
                  )}

                  {/* Vibes */}
                  {videoAnalysis.content_analysis.vibes && videoAnalysis.content_analysis.vibes.length > 0 && (
                    <div className="space-y-3">
                      <div className="flex items-center gap-2">
                        <Sparkles className="w-4 h-4 text-muted-foreground" />
                        <h3 className="font-semibold text-foreground">Vibes</h3>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {videoAnalysis.content_analysis.vibes.map((vibe, index) => (
                          <Badge
                            key={index}
                            variant="secondary"
                            className="bg-primary/10 text-primary hover:bg-primary/20"
                          >
                            {vibe.name} ({formatConfidence(vibe.confidence)})
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Video Stats */}
                  <div className="space-y-3">
                    <div className="flex items-center gap-2">
                      <Grid3X3 className="w-4 h-4 text-muted-foreground" />
                      <h3 className="font-semibold text-foreground">Video Stats</h3>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div className="bg-muted rounded-lg p-3">
                        <p className="font-medium">Scenes Detected</p>
                        <p className="text-muted-foreground">{videoAnalysis.scenes_detected}</p>
                      </div>
                      <div className="bg-muted rounded-lg p-3">
                        <p className="font-medium">Keyframes Generated</p>
                        <p className="text-muted-foreground">{videoAnalysis.keyframes_generated}</p>
                      </div>
                      <div className="bg-muted rounded-lg p-3">
                        <p className="font-medium">Cropped Keyframes</p>
                        <p className="text-muted-foreground">{videoAnalysis.cropped_keyframes_generated}</p>
                      </div>
                      <div className="bg-muted rounded-lg p-3">
                        <p className="font-medium">Video ID</p>
                        <p className="text-muted-foreground truncate">{videoAnalysis.video_id}</p>
                      </div>
                    </div>
                  </div>

                  {/* Empty state for content analysis */}
                  {!videoAnalysis.content_analysis.audio_transcription &&
                    !videoAnalysis.content_analysis.clothing_description &&
                    (!videoAnalysis.content_analysis.vibes || videoAnalysis.content_analysis.vibes.length === 0) && (
                      <div className="text-center py-8">
                        <p className="text-muted-foreground">No content analysis available</p>
                      </div>
                    )}
                </CardContent>
              </Card>
            )}
          </div>
        )}

        {/* Keyframes Gallery */}
        {videoAnalysis?.cropped_keyframes && videoAnalysis.cropped_keyframes.length > 0 && (
          <Card className="border-border bg-card shadow-sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Grid3X3 className="w-5 h-5" />
                Detected Items ({videoAnalysis.cropped_keyframes_generated})
              </CardTitle>
            </CardHeader>
            <CardContent>
              {isLoadingKeyframes ? (
                <div className="flex justify-center items-center py-12">
                  <div className="w-8 h-8 border-4 border-primary/30 border-t-primary rounded-full animate-spin"></div>
                </div>
              ) : categories.length > 0 ? (
                <Tabs defaultValue={categories[0]} className="w-full">
                  <TabsList className="mb-4 flex flex-wrap">
                    {categories.map((category) => (
                      <TabsTrigger key={category} value={category} className="capitalize">
                        {category} ({groupedKeyframes[category].length})
                      </TabsTrigger>
                    ))}
                  </TabsList>

                  {categories.map((category) => (
                    <TabsContent key={category} value={category} className="mt-0">
                      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
                        {groupedKeyframes[category].map((keyframe, index) => (
                          <div key={index} className="space-y-2">
                            <HoverCard>
                              <HoverCardTrigger asChild>
                                <div className="relative aspect-square bg-muted rounded-lg overflow-hidden cursor-pointer hover:ring-2 hover:ring-primary/50 transition-all">
                                  <img
                                    src={keyframe.imageUrl || "/placeholder.svg?height=200&width=200"}
                                    alt={keyframe.class_name}
                                    className="w-full h-full object-cover"
                                    onError={(e) => {
                                      console.error('Failed to load image:', keyframe.imageUrl)
                                      console.error('Image error event:', e)
                                    }}
                                    onLoad={() => {
                                      console.log('Successfully loaded image:', keyframe.imageUrl)
                                    }}
                                  />
                                  <div className="absolute bottom-0 left-0 right-0 bg-black/60 p-2">
                                    <p className="text-xs text-white truncate capitalize">{keyframe.class_name}</p>
                                  </div>
                                  {keyframe.fashion_matches && keyframe.fashion_matches.length > 0 && (
                                    <div className="absolute top-2 right-2">
                                      <Badge variant="secondary" className="bg-primary/90 text-primary-foreground text-xs">
                                        {keyframe.fashion_matches.length} match{keyframe.fashion_matches.length !== 1 ? 'es' : ''}
                                      </Badge>
                                    </div>
                                  )}
                                </div>
                              </HoverCardTrigger>
                              <HoverCardContent className="w-80 p-4" side="top">
                                <div className="space-y-3">
                                  <div className="flex items-center gap-2">
                                    <img
                                      src={keyframe.imageUrl || "/placeholder.svg?height=40&width=40"}
                                      alt={keyframe.class_name}
                                      className="w-10 h-10 rounded object-cover"
                                    />
                                    <div>
                                      <p className="font-semibold text-sm capitalize">{keyframe.class_name}</p>
                                      <p className="text-xs text-muted-foreground">{keyframe.filename}</p>
                                    </div>
                                  </div>

                                  {keyframe.fashion_matches && keyframe.fashion_matches.length > 0 ? (
                                    <div className="space-y-3">
                                      <div className="flex items-center gap-1">
                                        <Sparkles className="w-4 h-4 text-primary" />
                                        <span className="font-medium text-sm">Fashion Matches</span>
                                      </div>
                                      <div className="space-y-2 max-h-64 overflow-y-auto">
                                        {keyframe.fashion_matches.slice(0, 3).map((match, matchIndex) => (
                                          <div key={matchIndex} className="p-3 bg-muted rounded-lg space-y-2">
                                            <div className="flex items-start justify-between">
                                              <div className="flex-1">
                                                <p className="font-medium text-sm line-clamp-1">{match.title}</p>
                                                <p className="text-xs text-muted-foreground">{match.product_type}</p>
                                              </div>
                                              <div className="text-right">
                                                <p className="font-semibold text-sm">₹{match.price}</p>
                                                <Badge variant="outline" className="text-xs">
                                                  {Math.round(match.score * 100)}% match
                                                </Badge>
                                              </div>
                                            </div>
                                            <p className="text-xs text-muted-foreground line-clamp-2">{match.description}</p>
                                            {match.tags && (
                                              <div className="flex flex-wrap gap-1">
                                                {match.tags.split(',').slice(0, 3).map((tag, tagIndex) => (
                                                  <Badge key={tagIndex} variant="secondary" className="text-xs">
                                                    {tag.trim()}
                                                  </Badge>
                                                ))}
                                              </div>
                                            )}
                                          </div>
                                        ))}
                                        {keyframe.fashion_matches.length > 3 && (
                                          <p className="text-xs text-muted-foreground text-center">
                                            +{keyframe.fashion_matches.length - 3} more matches
                                          </p>
                                        )}
                                      </div>
                                    </div>
                                  ) : (
                                    <div className="text-center py-4">
                                      <p className="text-sm text-muted-foreground">No fashion matches found</p>
                                    </div>
                                  )}
                                </div>
                              </HoverCardContent>
                            </HoverCard>
                            <div className="flex items-center justify-between text-xs text-muted-foreground">
                              <span className="truncate">{keyframe.filename}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </TabsContent>
                  ))}
                </Tabs>
              ) : (
                <div className="text-center py-8">
                  <p className="text-muted-foreground">No keyframes to display</p>
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Dropzone - Only show when no file is loaded */}
        {!mediaFile && (
          <Card className="border-border bg-card shadow-sm">
            <CardContent className="p-6">
              <div
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onPaste={handlePaste}
                className={cn(
                  "relative border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer",
                  isDragOver
                    ? "border-primary bg-accent"
                    : "border-border hover:border-muted-foreground hover:bg-accent/50",
                )}
                onClick={() => fileInputRef.current?.click()}
                tabIndex={0}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".png,.jpg,.jpeg,.mp4"
                  onChange={handleFileSelect}
                  className="hidden"
                />

                <div className="space-y-4">
                  <div className="mx-auto w-12 h-12 rounded-full bg-muted flex items-center justify-center">
                    <Upload className="w-6 h-6 text-muted-foreground" />
                  </div>
                  <div className="space-y-2">
                    <p className="text-lg font-medium text-foreground">Drop file here, paste image, or click to upload</p>
                    <p className="text-sm text-muted-foreground">Supports PNG, JPEG, JPG, and MP4 files</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Empty State */}
        {!mediaFile && (
          <div className="text-center py-12">
            <div className="mx-auto w-16 h-16 rounded-full bg-muted flex items-center justify-center mb-4">
              <FileImage className="w-8 h-8 text-muted-foreground" />
            </div>
            <p className="text-muted-foreground">No media file uploaded yet</p>
          </div>
        )}
      </div>
    </div>
  )
}
