# scripts/run_full.ps1

param(
  [string]$DataRoot   = "data\moth_sounds",
  [string]$CacheRoot  = "data_caches",
  [string]$CacheId    = "",
  [string]$Config     = "configs\audio_moth.yaml",
  [string]$RunsRoot   = "runs",
  [string]$Variant    = "V0",
  [string]$Policy     = "greedy",
  [string]$Device     = "cpu",
  [double]$SegmentSec = 1.0,
  [double]$HopSec     = 0.5,
  [int]$NMels         = 64,
  [int]$SampleRate    = 16000,
  [double]$SilenceDbfs = -40,
  # Use "100,3000" for moth-style filtering, "50,7600" for wider C-class filtering,
  # "none" to disable bandpass, or "config" to let scripts/prep_segments.py read YAML.
  [string]$Bandpass   = "100,3000",
  [int]$NFFT          = 1024,
  [int]$WinMs         = 25,
  [int]$FeatHopMs     = 10,
  [switch]$NoCMVN,

  # Generic K-exit control
  # 3 exits -> "1,3"
  # 5 exits -> "1,2,3,4"
  [string]$TapBlocks  = "1,2,3,4",

  [switch]$RunClipPolicy,
  [double]$TimeConf   = 0.95,
  [int]$TimeStableK   = 2,
  [int]$TimeMinWindows = 2,
  [int]$EvalFixedKWindows = 3,
  [double]$TimeMargin = 0.0,

  # Unified preprocessing / segmentation controls
  # InputMode:
  #   segment -> raw mixed-length audio is cleaned, split, segmented, then exported
  #   ready   -> already clipped audio is cleaned/exported; if DataRoot has
  #              train/val/test/<class> folders, those splits are preserved
  [ValidateSet("segment", "ready")]
  [string]$InputMode = "segment",

  # Optional comma-separated labels. Empty = auto-discover class folders under DataRoot.
  # Example: -Labels "gun_shot,fireworks,rain,wind"
  [string]$Labels = "",

  # Keep short audio if duration >= MinKeepSec by padding to SegmentSec.
  [double]$MinKeepSec = 0.25,

  # Generic cap. 0 = keep all segments per parent file.
  [int]$MaxSegmentsPerFileDefault = 0,

  # Per-label override. Example:
  # -MaxSegmentsPerLabelJson '{"fireworks":5,"rain":5,"wind":5,"gun_shot":0}'
  [string]$MaxSegmentsPerLabelJson = "",

  # Leakage control:
  #   file  -> all segments from one source file stay in one split
  #   group -> related files stay together based on GroupRegex or folder structure
  [ValidateSet("file", "group")]
  [string]$SplitUnit = "file",
  [string]$GroupRegex = "",

  # Rebuild cache even if segments/features already exist
  [switch]$ForceRebuild,

  # CLI override for hint passing.
  # Use:
  #   -ExitHint "true"   -> force hint enabled
  #   -ExitHint "false"  -> force hint disabled
  #   omit / leave empty -> use YAML default
  [string]$ExitHint = ""
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Get-Location).Path
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

function Format-ArgForDisplay([string]$s) {
  if ($null -eq $s) { return "''" }
  if ($s -match '[\s"''{}:,\\]') {
    return '"' + ($s -replace '"', '\"') + '"'
  }
  return $s
}

function Invoke-PythonArgs([string[]]$ArgsList) {
  $display = ($ArgsList | ForEach-Object { Format-ArgForDisplay $_ }) -join " "
  Write-Host "  python $display" -ForegroundColor DarkGray
  & python @ArgsList
  if ($LASTEXITCODE -ne 0) {
    throw "Python command failed ($LASTEXITCODE): python $display"
  }
}

function ConvertTo-Id([double]$x) {
  return ($x.ToString("0.###") -replace '\.', 'p')
}

function Is-BandpassDisabledOrConfig([string]$bp) {
  if ($null -eq $bp) { return $true }
  $b = $bp.Trim().ToLower()
  return ($b -eq "" -or $b -eq "none" -or $b -eq "null" -or $b -eq "false" -or $b -eq "off" -or $b -eq "config")
}

function Get-BandpassId([string]$bp) {
  if ($null -eq $bp) { return "bpConfig" }
  $b = $bp.Trim()
  if ($b.ToLower() -eq "config" -or $b -eq "") { return "bpConfig" }
  if ($b.ToLower() -in @("none", "null", "false", "off")) { return "bpNone" }
  return "bp" + (($b -replace '\s+', '') -replace ',', '-')
}

function Split-Bandpass([string]$bp) {
  if (Is-BandpassDisabledOrConfig $bp) { return @() }
  $items = @($bp.Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })
  if ($items.Count -ne 2) {
    throw "Bandpass must be 'low,high', 'none', or 'config'. Got: $bp"
  }
  return $items
}

# --------------------- Validate / normalize ExitHint ---------------------
if ($ExitHint -ne "") {
  $eh = $ExitHint.Trim().ToLower()
  if ($eh -notin @("true", "false")) {
    throw "ExitHint must be either 'true' or 'false'."
  }
  $ExitHint = $eh
}

# --------------------- Run directory scheme: runs/<Variant>/<Variant>_### ---------------------
$VariantSafe = ($Variant -replace '[^A-Za-z0-9_-]', '_')
$variantRunDir = Join-Path $RunsRoot $VariantSafe

New-Item -ItemType Directory -Path $RunsRoot      -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Path $variantRunDir -ErrorAction SilentlyContinue | Out-Null

$variantEsc = [regex]::Escape($VariantSafe)
$maxN = 0
Get-ChildItem -Path $variantRunDir -Directory -ErrorAction SilentlyContinue | ForEach-Object {
  if ($_.Name -match "^$variantEsc`_(\d+)$") {
    $n = [int]$Matches[1]
    if ($n -gt $maxN) { $maxN = $n }
  }
}
$nextN  = $maxN + 1
$runId  = "{0}_{1:000}" -f $VariantSafe, $nextN
$runPath = Join-Path $variantRunDir $runId
if (Test-Path $runPath) { throw "Target run folder already exists: $runPath" }

# --------------------- Cache directory scheme ---------------------
if ([string]::IsNullOrWhiteSpace($CacheId)) {
  $segId = ConvertTo-Id $SegmentSec
  $hopId = ConvertTo-Id $HopSec
  $tapId = (($TapBlocks -replace '\s+', '') -replace ',', '-')
  $capId = $(if ($MaxSegmentsPerFileDefault -gt 0) { "cap$MaxSegmentsPerFileDefault" } else { "capAll" })
  $modeId = $InputMode
  $splitId = $SplitUnit
  $bpId = Get-BandpassId $Bandpass
  $CacheId = "mode$modeId" + "_seg$segId" + "_hop$hopId" + "_$bpId" + "_mels$NMels" + "_tap$tapId" + "_$capId" + "_split$splitId"
}

$CacheIdSafe     = ($CacheId -replace '[^A-Za-z0-9_-]', '_')
$variantCacheDir = Join-Path (Join-Path $CacheRoot $VariantSafe) $CacheIdSafe

if ($ForceRebuild -and (Test-Path $variantCacheDir)) {
  Write-Host "`n[cache] ForceRebuild enabled. Removing cache: $variantCacheDir" -ForegroundColor Yellow
  Remove-Item -Recurse -Force $variantCacheDir
}

New-Item -ItemType Directory -Path $CacheRoot -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Path (Join-Path $CacheRoot $VariantSafe) -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Path $variantCacheDir -ErrorAction SilentlyContinue | Out-Null

$SegCsv     = Join-Path $variantCacheDir "segments.csv"
$FeatRoot   = Join-Path $variantCacheDir "features"
$SegWavRoot = Join-Path $variantCacheDir "segment_wavs"

$pipelineStart = Get-Date

Write-Host "== ASHADIP: full pipeline run ==" -ForegroundColor Cyan
Write-Host "  DataRoot        = $DataRoot"        -ForegroundColor DarkGray
Write-Host "  CacheRoot       = $CacheRoot"       -ForegroundColor DarkGray
Write-Host "  CacheId         = $CacheIdSafe"     -ForegroundColor DarkGray
Write-Host "  CacheDir        = $variantCacheDir" -ForegroundColor DarkGray
Write-Host "  Config          = $Config"          -ForegroundColor DarkGray
Write-Host "  RunsRoot        = $RunsRoot"        -ForegroundColor DarkGray
Write-Host "  Variant         = $Variant"         -ForegroundColor DarkGray
Write-Host "  Policy          = $Policy"          -ForegroundColor DarkGray
Write-Host "  Device          = $Device"          -ForegroundColor DarkGray
Write-Host "  SegmentSec      = $SegmentSec"      -ForegroundColor DarkGray
Write-Host "  HopSec          = $HopSec"          -ForegroundColor DarkGray
Write-Host "  NMels           = $NMels"           -ForegroundColor DarkGray
Write-Host "  SampleRate      = $SampleRate"      -ForegroundColor DarkGray
Write-Host "  SilenceDbfs     = $SilenceDbfs"     -ForegroundColor DarkGray
Write-Host "  Bandpass        = $Bandpass"        -ForegroundColor DarkGray
Write-Host "  NFFT/Win/HopMs  = $NFFT/$WinMs/$FeatHopMs" -ForegroundColor DarkGray
Write-Host "  CMVN            = $(-not [bool]$NoCMVN)" -ForegroundColor DarkGray
Write-Host "  TapBlocks       = $TapBlocks"       -ForegroundColor DarkGray
Write-Host "  InputMode       = $InputMode"       -ForegroundColor DarkGray
if ($InputMode -eq "ready") {
  Write-Host "  Ready layout    = train/val/test/<class> preserved when present" -ForegroundColor DarkGray
}
Write-Host "  Labels          = $(if ($Labels -ne '') { $Labels } else { 'auto-discover' })" -ForegroundColor DarkGray
Write-Host "  MinKeepSec      = $MinKeepSec"      -ForegroundColor DarkGray
Write-Host "  MaxSeg/File     = $MaxSegmentsPerFileDefault" -ForegroundColor DarkGray
Write-Host "  MaxSeg/Label    = $(if ($MaxSegmentsPerLabelJson -ne '') { $MaxSegmentsPerLabelJson } else { '{}' })" -ForegroundColor DarkGray
Write-Host "  SplitUnit       = $SplitUnit"       -ForegroundColor DarkGray
Write-Host "  GroupRegex      = $(if ($GroupRegex -ne '') { $GroupRegex } else { '<none>' })" -ForegroundColor DarkGray
if ($ExitHint -ne "") {
  Write-Host "  ExitHint        = $ExitHint (CLI override)" -ForegroundColor DarkGray
} else {
  Write-Host "  ExitHint        = YAML default" -ForegroundColor DarkGray
}
Write-Host "  ForceRebuild    = $ForceRebuild"    -ForegroundColor DarkGray
Write-Host "  RunDir          = $runPath"         -ForegroundColor DarkGray
Write-Host "  RunClipPolicy   = $RunClipPolicy"   -ForegroundColor DarkGray

$cacheReady = (Test-Path $SegCsv) -and (Test-Path $FeatRoot) -and (Test-Path $SegWavRoot)

if ($cacheReady -and -not $ForceRebuild) {
  Write-Host "`n[cache] Reusing existing cache: $variantCacheDir" -ForegroundColor Green
}
else {
  # --------------------- 1/10) Prep segments ---------------------
  Write-Host "`n[1/10] Prep segments and export physical segment WAVs..." -ForegroundColor Yellow

  $prepArgs = @(
    "-m", "scripts.prep_segments",
    "--root", $DataRoot,
    "--cache", $variantCacheDir,
    "--sr", ([string]$SampleRate),
    "--segment_sec", ([string]$SegmentSec),
    "--hop", ([string]$HopSec),
    "--silence_dbfs", ([string]$SilenceDbfs),
    "--config", $Config,
    "--input_mode", $InputMode,
    "--min_keep_sec", ([string]$MinKeepSec),
    "--max_segments_per_file_default", ([string]$MaxSegmentsPerFileDefault),
    "--split_unit", $SplitUnit,
    "--export_segment_wavs"
  )

  $bpItems = Split-Bandpass $Bandpass
  if ($bpItems.Count -eq 2) {
    $prepArgs += "--bandpass"
    $prepArgs += $bpItems[0]
    $prepArgs += $bpItems[1]
  }

  if ($Labels.Trim() -ne "") {
    $labelItems = @($Labels.Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })
    if ($labelItems.Count -gt 0) {
      $prepArgs += "--labels"
      foreach ($lab in $labelItems) {
        $prepArgs += $lab
      }
    }
  }

  if ($MaxSegmentsPerLabelJson.Trim() -ne "") {
    # IMPORTANT: pass as a single argv item. This prevents PowerShell from stripping JSON quotes.
    $prepArgs += "--max_segments_per_label_json"
    $prepArgs += $MaxSegmentsPerLabelJson
  }

  if ($GroupRegex.Trim() -ne "") {
    $prepArgs += "--group_regex"
    $prepArgs += $GroupRegex
  }

  Invoke-PythonArgs $prepArgs

  # --------------------- 2/10) Extract features ---------------------
  Write-Host "`n[2/10] Extract features from physical segment WAVs..." -ForegroundColor Yellow
  $featureArgs = @(
    "-m", "scripts.extract_features",
    "--cache", $variantCacheDir,
    "--n_mels", ([string]$NMels),
    "--n_fft", ([string]$NFFT),
    "--win_ms", ([string]$WinMs),
    "--hop_ms", ([string]$FeatHopMs),
    "--pad_short"
  )
  if (-not $NoCMVN) {
    $featureArgs += "--cmvn"
  }
  Invoke-PythonArgs $featureArgs
}

# --------------------- 3/10) Train ExitNet ---------------------
Write-Host "`n[3/10] Train ExitNet..." -ForegroundColor Yellow
$trainArgs = @(
  "-m", "training.train",
  "--config", $Config,
  "--run_dir", $runPath,
  "--cache_dir", $variantCacheDir,
  "--device", $Device,
  "--segment_sec", ([string]$SegmentSec),
  "--hop_sec", ([string]$HopSec),
  "--variant", $Variant,
  "--tap_blocks", $TapBlocks
)
if ($ExitHint -ne "") {
  $trainArgs += "--exit_hint_enable"
  $trainArgs += $ExitHint
}
Invoke-PythonArgs $trainArgs

Write-Host "Using run: $runPath" -ForegroundColor Green

# Save meta.json for traceability
$createdAtIso = Get-Date -Format o
$meta = @{
  run_id                        = $runId
  variant                       = $Variant
  variant_safe                  = $VariantSafe
  created_at                    = $createdAtIso
  runs_root                     = $RunsRoot
  variant_dir                   = $variantRunDir
  cache_root                    = $CacheRoot
  cache_id                      = $CacheIdSafe
  cache_dir                     = $variantCacheDir
  data_root                     = $DataRoot
  device                        = $Device
  policy                        = $Policy
  segment_sec                   = $SegmentSec
  hop_sec                       = $HopSec
  n_mels                        = $NMels
  sample_rate                   = $SampleRate
  silence_dbfs                  = $SilenceDbfs
  bandpass                      = $Bandpass
  n_fft                         = $NFFT
  win_ms                        = $WinMs
  feat_hop_ms                   = $FeatHopMs
  cmvn                          = [bool](-not $NoCMVN)
  tap_blocks                    = $TapBlocks
  input_mode                    = $InputMode
  labels                        = $(if ($Labels -ne "") { $Labels } else { "auto_discover" })
  min_keep_sec                  = $MinKeepSec
  max_segments_per_file_default = $MaxSegmentsPerFileDefault
  max_segments_per_label_json   = $MaxSegmentsPerLabelJson
  split_unit                    = $SplitUnit
  group_regex                   = $GroupRegex
  force_rebuild                 = [bool]$ForceRebuild
  segment_wavs_root             = $SegWavRoot
  exit_hint_override            = $(if ($ExitHint -ne "") { $ExitHint } else { "yaml_default" })
  run_clip_policy               = [bool]$RunClipPolicy
  time_conf                     = $TimeConf
  time_stable_k                 = $TimeStableK
  time_min_windows              = $TimeMinWindows
  eval_fixed_k_windows          = $EvalFixedKWindows
  time_margin                   = $TimeMargin
}
New-Item -ItemType Directory -Path $runPath -ErrorAction SilentlyContinue | Out-Null
$meta | ConvertTo-Json -Depth 8 | Out-File -FilePath (Join-Path $runPath "meta.json") -Encoding UTF8

# --------------------- 4/10) Calibrate temperatures ---------------------
Write-Host "`n[4/10] Calibrate temperatures..." -ForegroundColor Yellow
Invoke-PythonArgs @(
  "-m", "training.calibrate",
  "--run_dir", $runPath,
  "--segments_csv", $SegCsv,
  "--features_root", $FeatRoot,
  "--tap_blocks", $TapBlocks,
  "--n_mels", ([string]$NMels)
)

# --------------------- 5/10) Select threshold (greedy path) ---------------------
Write-Host "`n[5/10] Select threshold (tau)..." -ForegroundColor Yellow
Invoke-PythonArgs @(
  "-m", "training.thresholds_offline",
  "--run_dir", $runPath,
  "--segments_csv", $SegCsv,
  "--features_root", $FeatRoot,
  "--tap_blocks", $TapBlocks,
  "--n_mels", ([string]$NMels)
)

# --------------------- 6/10) Segment policy test ---------------------
Write-Host "`n[6/10] Segment policy test..." -ForegroundColor Yellow
Invoke-PythonArgs @(
  "-m", "scripts.policy_test",
  "--run_dir", $runPath,
  "--segments_csv", $SegCsv,
  "--features_root", $FeatRoot,
  "--tap_blocks", $TapBlocks,
  "--n_mels", ([string]$NMels)
)

# --------------------- Guard: current clip tester is greedy-only ---------------------
if ($RunClipPolicy -and $Policy -ne "greedy") {
  throw "Current scripts.clip_policy_test.py is greedy-only. Use -Policy greedy, or create an EA-compatible clip_policy_test.py first."
}

# --------------------- 6b/10) Clip policy test (optional) ---------------------
if ($RunClipPolicy) {
  Write-Host "`n[6b/10] Clip policy test..." -ForegroundColor Yellow
  Invoke-PythonArgs @(
    "-m", "scripts.clip_policy_test",
    "--run_dir", $runPath,
    "--segments_csv", $SegCsv,
    "--features_root", $FeatRoot,
    "--device", $Device,
    "--tap_blocks", $TapBlocks,
    "--n_mels", ([string]$NMels),
    "--time_conf", ([string]$TimeConf),
    "--time_stable_k", ([string]$TimeStableK),
    "--time_min_windows", ([string]$TimeMinWindows),
    "--fixed_k_windows", ([string]$EvalFixedKWindows),
    "--time_margin", ([string]$TimeMargin)
  )
}

# --------------------- 7/10) Summarise run ---------------------
Write-Host "`n[7/10] Summarise run..." -ForegroundColor Yellow
Invoke-PythonArgs @(
  "-m", "scripts.summarize_run",
  "--run_dir", $runPath,
  "--segments_csv", $SegCsv,
  "--features_root", $FeatRoot,
  "--tap_blocks", $TapBlocks,
  "--n_mels", ([string]$NMels)
)

# --------------------- 8/10) Analyse run ---------------------
Write-Host "`n[8/10] Analyse run..." -ForegroundColor Yellow
Invoke-PythonArgs @(
  "-m", "scripts.analyse_run",
  "--run_dir", $runPath,
  "--segments_csv", $SegCsv,
  "--features_root", $FeatRoot,
  "--tap_blocks", $TapBlocks,
  "--n_mels", ([string]$NMels)
)

# --------------------- 9/10) Profile latency ---------------------
Write-Host "`n[9/10] Profile latency..." -ForegroundColor Yellow
Invoke-PythonArgs @(
  "-m", "scripts.profile_latency",
  "--run_dir", $runPath,
  "--segments_csv", $SegCsv,
  "--features_root", $FeatRoot,
  "--variant", $Variant,
  "--device", $Device,
  "--tap_blocks", $TapBlocks,
  "--n_mels", ([string]$NMels)
)

# --------------------- Timing & logging ---------------------
$pipelineEnd   = Get-Date
$elapsed       = $pipelineEnd - $pipelineStart
$totalSeconds  = [Math]::Round($elapsed.TotalSeconds, 2)
$totalMinutes  = [Math]::Round($elapsed.TotalMinutes, 2)
$timestampIso  = Get-Date -Format o

Write-Host ""
Write-Host ("Total wall-clock time: {0} seconds (~{1} minutes)" -f $totalSeconds, $totalMinutes) -ForegroundColor Cyan

$analysisDir = "analysis"
New-Item -ItemType Directory -Path $analysisDir -ErrorAction SilentlyContinue | Out-Null
$runtimeCsv = Join-Path $analysisDir "pipeline_runtime.csv"

if (-not (Test-Path $runtimeCsv)) {
  "timestamp,variant,policy,segment_sec,hop_sec,device,cache_dir,runs_root,run_id,total_seconds,total_minutes,run_clip_policy,tap_blocks,exit_hint_override,input_mode,split_unit,max_segments_per_file_default,sample_rate,silence_dbfs,bandpass,n_fft,win_ms,feat_hop_ms,cmvn" | Out-File $runtimeCsv -Encoding UTF8
}

$csvLine = "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20},{21},{22},{23}" -f `
  $timestampIso, `
  $Variant, `
  $Policy, `
  $SegmentSec, `
  $HopSec, `
  $Device, `
  $variantCacheDir, `
  $RunsRoot, `
  $runId, `
  $totalSeconds, `
  $totalMinutes, `
  [bool]$RunClipPolicy, `
  $TapBlocks, `
  $(if ($ExitHint -ne "") { $ExitHint } else { "yaml_default" }), `
  $InputMode, `
  $SplitUnit, `
  $MaxSegmentsPerFileDefault, `
  $SampleRate, `
  $SilenceDbfs, `
  $Bandpass, `
  $NFFT, `
  $WinMs, `
  $FeatHopMs, `
  [bool](-not $NoCMVN)

Add-Content -Path $runtimeCsv -Value $csvLine
Write-Host "Pipeline runtime logged to: $runtimeCsv" -ForegroundColor DarkGray

# --------------------- 10/10) Reports & LaTeX ---------------------
Write-Host "`n[10/10] Generate reports & LaTeX tables..." -ForegroundColor Yellow
powershell -ExecutionPolicy Bypass -File scripts\run_reports.ps1 `
  -RunDir $runPath `
  -Variant $Variant `
  -DeviceFilter $Device `
  -SegmentsCsv $SegCsv `
  -FeaturesRoot $FeatRoot `
  -RunsRoot $RunsRoot

Write-Host "`n== Done. Artifacts at: $runPath ==" -ForegroundColor Cyan
Write-Host "Cache used: $variantCacheDir" -ForegroundColor DarkGray
Write-Host "Physical segment WAVs: $SegWavRoot" -ForegroundColor DarkGray
