param(
  [string]$Manifest = "human_talk_workspace\tata_2\feature_cache\metadata\multilabel_features_manifest.csv",
  [string]$FeaturesRoot = "human_talk_workspace\tata_2\feature_cache\features",
  [string]$LabelsJson = "human_talk_workspace\tata_2\segment_cache\metadata\tata_labels.json",

  [string]$WorkspaceRoot = "human_talk_workspace\tata_2",
  [string]$RunsRoot = "human_talk_workspace\tata_2\runs",
  [string]$PackagesRoot = "human_talk_workspace\tata_2\packages",
  [string]$LogsRoot = "human_talk_workspace\tata_2\logs",

  [string]$Variant = "tata_2_3exit_weakclip",
  [string]$TapBlocks = "1,3",

  [int]$Epochs = 40,
  [int]$BatchSize = 64,
  [int]$LogEvery = 0,
  [double]$LR = 0.001,
  [double]$Threshold = 0.5,
  [string]$Device = "cpu",

  [switch]$UsePosWeight,
  [double]$PosWeightMax = 20.0,

  [switch]$IncludeCheckpoint
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Get-Location).Path
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

New-Item -ItemType Directory -Force -Path $RunsRoot | Out-Null
New-Item -ItemType Directory -Force -Path $PackagesRoot | Out-Null
New-Item -ItemType Directory -Force -Path $LogsRoot | Out-Null

$TranscriptPath = Join-Path $LogsRoot "${Variant}_${Timestamp}_cli_output.txt"
$ZipPath = Join-Path $PackagesRoot "${Variant}_results_to_share_${Timestamp}.zip"

Write-Host "== TATA weak-clip experiment ==" -ForegroundColor Cyan
Write-Host "Transcript: $TranscriptPath" -ForegroundColor DarkGray
Write-Host "Package:    $ZipPath" -ForegroundColor DarkGray

$StartTime = Get-Date

Start-Transcript -Path $TranscriptPath -Force | Out-Null

try {
    Write-Host ""
    Write-Host "== Inputs ==" -ForegroundColor Cyan
    Write-Host "Manifest     = $Manifest"
    Write-Host "FeaturesRoot = $FeaturesRoot"
    Write-Host "LabelsJson   = $LabelsJson"
    Write-Host "RunsRoot     = $RunsRoot"
    Write-Host "Variant      = $Variant"
    Write-Host "TapBlocks    = $TapBlocks"
    Write-Host "Epochs       = $Epochs"
    Write-Host "BatchSize    = $BatchSize"
    Write-Host "LogEvery     = $LogEvery"
    Write-Host "LR           = $LR"
    Write-Host "Threshold    = $Threshold"
    Write-Host "Device       = $Device"
    Write-Host "UsePosWeight = $UsePosWeight"

    if (-not (Test-Path $Manifest)) {
        throw "Manifest not found: $Manifest"
    }

    if (-not (Test-Path $FeaturesRoot)) {
        throw "FeaturesRoot not found: $FeaturesRoot"
    }

    if (-not (Test-Path $LabelsJson)) {
        throw "LabelsJson not found: $LabelsJson"
    }

    $TrainArgs = @(
        "-m", "training.train_multilabel",
        "--manifest", $Manifest,
        "--features_root", $FeaturesRoot,
        "--labels_json", $LabelsJson,
        "--runs_root", $RunsRoot,
        "--variant", $Variant,
        "--tap_blocks", $TapBlocks,
        "--epochs", "$Epochs",
        "--batch_size", "$BatchSize",
        "--log_every", "$LogEvery",
        "--lr", "$LR",
        "--threshold", "$Threshold",
        "--device", $Device
    )

    if ($UsePosWeight) {
        $TrainArgs += "--use_pos_weight"
        $TrainArgs += "--pos_weight_max"
        $TrainArgs += "$PosWeightMax"
    }

    Write-Host ""
    Write-Host "== Training command ==" -ForegroundColor Yellow
    Write-Host ("python " + ($TrainArgs -join " "))

    & python @TrainArgs

    if ($LASTEXITCODE -ne 0) {
        throw "Training failed with exit code $LASTEXITCODE"
    }

    $EndTime = Get-Date

    $LatestRun = Get-ChildItem $RunsRoot -Directory |
        Where-Object { $_.Name -like "$Variant*" } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if ($null -eq $LatestRun) {
        throw "Training finished, but no run directory found under $RunsRoot for variant $Variant"
    }

    $RunDir = $LatestRun.FullName

    Write-Host ""
    Write-Host "== Training complete ==" -ForegroundColor Green
    Write-Host "RunDir = $RunDir"

    $Meta = [ordered]@{
        variant = $Variant
        tap_blocks = $TapBlocks
        started_at = $StartTime.ToString("o")
        finished_at = $EndTime.ToString("o")
        elapsed_seconds = [Math]::Round(($EndTime - $StartTime).TotalSeconds, 2)
        manifest = $Manifest
        features_root = $FeaturesRoot
        labels_json = $LabelsJson
        runs_root = $RunsRoot
        run_dir = $RunDir
        transcript = $TranscriptPath
        device = $Device
        epochs = $Epochs
        batch_size = $BatchSize
        log_every = $LogEvery
        lr = $LR
        threshold = $Threshold
        use_pos_weight = [bool]$UsePosWeight
        pos_weight_max = $PosWeightMax
        include_checkpoint_in_package = [bool]$IncludeCheckpoint
    }

    $MetaPath = Join-Path $RunDir "tata_experiment_meta.json"
    $Meta | ConvertTo-Json -Depth 8 | Out-File -FilePath $MetaPath -Encoding UTF8
}
finally {
    Stop-Transcript | Out-Null
}

# Copy transcript into run folder after transcript is closed.
if ($RunDir -and (Test-Path $RunDir)) {
    Copy-Item -Force $TranscriptPath (Join-Path $RunDir "training_cli_output.txt")
}

# ---------------- Package share outputs ----------------

$TempShare = Join-Path $env:TEMP "tata_share_${Variant}_${Timestamp}"

if (Test-Path $TempShare) {
    Remove-Item -Recurse -Force $TempShare
}

New-Item -ItemType Directory -Force -Path $TempShare | Out-Null

$ShareRunDir = Join-Path $TempShare "run"
New-Item -ItemType Directory -Force -Path $ShareRunDir | Out-Null

$FilesToCopy = @(
    "metrics.json",
    "config_used.json",
    "tata_experiment_meta.json",
    "training_cli_output.txt"
)

foreach ($FileName in $FilesToCopy) {
    $Src = Join-Path $RunDir $FileName
    if (Test-Path $Src) {
        Copy-Item -Force $Src (Join-Path $ShareRunDir $FileName)
    }
}

if ($IncludeCheckpoint) {
    $CkptDir = Join-Path $RunDir "ckpt"
    if (Test-Path $CkptDir) {
        Copy-Item -Recurse -Force $CkptDir (Join-Path $ShareRunDir "ckpt")
    }
}

# Copy useful metadata, not heavy feature arrays.
$MetaShare = Join-Path $TempShare "metadata"
New-Item -ItemType Directory -Force -Path $MetaShare | Out-Null

$MetaFiles = @(
    "human_talk_workspace\tata_2\segment_cache\metadata\tata_segment_manifest_summary.json",
    "human_talk_workspace\tata_2\segment_cache\metadata\tata_segment_manifest_errors.csv",
    "human_talk_workspace\tata_2\segment_cache\metadata\tata_labels.json",
    "human_talk_workspace\tata_2\feature_cache\metadata\multilabel_features_manifest.csv"
)

foreach ($MetaFile in $MetaFiles) {
    if (Test-Path $MetaFile) {
        Copy-Item -Force $MetaFile (Join-Path $MetaShare (Split-Path $MetaFile -Leaf))
    }
}

$PackageManifest = [ordered]@{
    packaged_at = (Get-Date).ToString("o")
    run_dir = $RunDir
    transcript_original_path = $TranscriptPath
    zip_path = $ZipPath
    note = "This package copies selected outputs only. Original files are not moved or deleted."
    included_files = Get-ChildItem $TempShare -Recurse -File | ForEach-Object {
        $_.FullName.Replace($TempShare, "").TrimStart("\")
    }
}

$PackageManifest | ConvertTo-Json -Depth 8 |
    Out-File -FilePath (Join-Path $TempShare "package_manifest.json") -Encoding UTF8

if (Test-Path $ZipPath) {
    Remove-Item -Force $ZipPath
}

Compress-Archive -Path (Join-Path $TempShare "*") -DestinationPath $ZipPath -Force

Write-Host ""
Write-Host "== Package complete ==" -ForegroundColor Green
Write-Host "ZIP created:" -ForegroundColor Cyan
Write-Host $ZipPath
Write-Host ""
Write-Host "Original run files remain at:" -ForegroundColor DarkGray
Write-Host $RunDir
