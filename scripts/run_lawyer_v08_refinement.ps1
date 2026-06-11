param(
  [string]$V08Root = "human_talk_workspace\tata_v0.8_raw_pipeline",
  [string]$V06Root = "human_talk_workspace\tata_v0.6_raw_pipeline",

  [string]$Config = "configs\lawyer_v08_human_talk.json",
  [string]$SegmentPredictionsCsv = "",
  [string]$ParentCsv = "",

  [string]$ModeName = "lawyer_v08",

  [switch]$AddSilenceSignalFeatures,

  [int]$SampleRate = 16000,
  [double]$SegmentSec = 1.0,
  [double]$HopSec = 1.0,

  [double]$RmsSilenceThresholdDb = -45.0,
  [double]$SpeechActivityThreshold = 0.15,
  [double]$SpeechFrameDbThreshold = -45.0
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Get-Location).Path

if ([string]::IsNullOrWhiteSpace($SegmentPredictionsCsv)) {
  $SegmentPredictionsCsv = Join-Path $V06Root "raw_tata_pseudo_routing\raw_segment_predictions.csv"
}

if ([string]::IsNullOrWhiteSpace($ParentCsv)) {
  $ParentCsv = Join-Path $V06Root "raw_tata_pseudo_routing\hybrid\hybrid_parent_predictions_all.csv"
}

$RoutingRoot = Join-Path $V08Root "raw_tata_pseudo_routing"
$OutDir = Join-Path $RoutingRoot $ModeName

New-Item -ItemType Directory -Force -Path $RoutingRoot | Out-Null
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host ""
Write-Host "== LAWYER v0.8 config-driven refinement ==" -ForegroundColor Cyan
Write-Host "V08Root               = $V08Root"
Write-Host "Config                = $Config"
Write-Host "SegmentPredictionsCsv = $SegmentPredictionsCsv"
Write-Host "ParentCsv             = $ParentCsv"
Write-Host "OutDir                = $OutDir"
Write-Host "ModeName              = $ModeName"
Write-Host "AddSilenceSignal      = $AddSilenceSignalFeatures"
Write-Host ""

if (-not (Test-Path $Config)) {
  throw "Config not found: $Config"
}

if (-not (Test-Path $SegmentPredictionsCsv)) {
  throw "SegmentPredictionsCsv not found: $SegmentPredictionsCsv"
}

$LawyerInputCsv = $SegmentPredictionsCsv

if ($AddSilenceSignalFeatures) {
  $EnrichedCsv = Join-Path $RoutingRoot "raw_segment_predictions_with_silence_signal.csv"
  $SilenceSummary = Join-Path $RoutingRoot "raw_segment_predictions_with_silence_signal.summary.json"

  $SilenceArgs = @(
    "scripts\add_lawyer_silence_signal_features_v08.py",
    "--input_csv", $SegmentPredictionsCsv,
    "--output_csv", $EnrichedCsv,
    "--summary_json", $SilenceSummary,
    "--sample_rate", "$SampleRate",
    "--segment_sec", "$SegmentSec",
    "--hop_sec", "$HopSec",
    "--rms_silence_threshold_db", "$RmsSilenceThresholdDb",
    "--speech_activity_threshold", "$SpeechActivityThreshold",
    "--speech_frame_db_threshold", "$SpeechFrameDbThreshold"
  )

  Write-Host "== Adding silence signal features ==" -ForegroundColor Yellow
  Write-Host ("python " + ($SilenceArgs -join " "))

  & python @SilenceArgs

  if ($LASTEXITCODE -ne 0) {
    throw "Silence signal feature generation failed with exit code $LASTEXITCODE"
  }

  $LawyerInputCsv = $EnrichedCsv
}

$ArgsList = @(
  "scripts\lawyer_refine_weak_labels_v08.py",
  "--config", $Config,
  "--segment_predictions_csv", $LawyerInputCsv,
  "--out_dir", $OutDir,
  "--mode_name", $ModeName
)

if ((Test-Path $ParentCsv)) {
  $ArgsList += "--parent_csv"
  $ArgsList += $ParentCsv
} else {
  Write-Host "ParentCsv not found, continuing without parent context: $ParentCsv" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "== Running LAWYER ==" -ForegroundColor Yellow
Write-Host ("python " + ($ArgsList -join " "))

& python @ArgsList

if ($LASTEXITCODE -ne 0) {
  throw "LAWYER refinement failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "LAWYER v0.8 outputs:" -ForegroundColor Green
Write-Host $OutDir
