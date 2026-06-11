# scripts/run_human_talk_confusion_eval_all.ps1
#
# Export and package confusion matrices for existing clean-stage trained runs.
# This does NOT retrain models and does NOT rebuild features.
#
# Fix v2:
#   - Uses short package staging paths to avoid Windows MAX_PATH errors.
#   - Adds -ZipOnly so packaging can be repeated without recomputing confusion matrices.
#
# Example:
#   powershell -ExecutionPolicy Bypass -File .\scripts\run_human_talk_confusion_eval_all.ps1 `
#     -WorkspaceRoot human_talk_workspace `
#     -Device cpu `
#     -ZipResults
#
# Package existing outputs only:
#   powershell -ExecutionPolicy Bypass -File .\scripts\run_human_talk_confusion_eval_all.ps1 `
#     -WorkspaceRoot human_talk_workspace `
#     -ZipOnly `
#     -ZipResults

param(
    [string]$WorkspaceRoot = "human_talk_workspace",
    [string[]]$Stages = @("clean2_balanced", "clean3_balanced", "clean4_balanced", "clean5_balanced"),
    [string]$Device = "cpu",
    [string]$ThresholdMode = "fixed_0p5",
    [switch]$ZipOnly,
    [switch]$ZipResults
)

$ErrorActionPreference = "Stop"

function Get-TimeStamp {
    return Get-Date -Format "yyyyMMdd_HHmmss"
}

function Get-VariantPrefix {
    param([string]$StageName)
    $Core = $StageName -replace "_balanced$", ""
    return "human_talk_$Core"
}

function Get-StageCode {
    param([string]$StageName)
    if ($StageName -match "clean(\d)_balanced") {
        return "c$($Matches[1])"
    }
    return ($StageName -replace "[^A-Za-z0-9_]", "_")
}

function Get-ModelCodeFromRunName {
    param([string]$RunName)
    if ($RunName -match "3exit") {
        return "3e"
    }
    if ($RunName -match "5exit") {
        return "5e"
    }
    return "model"
}

function Get-LatestRunDir {
    param(
        [string]$RunsRoot,
        [string]$Pattern
    )

    $Run = Get-ChildItem $RunsRoot -Directory -Filter $Pattern |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if (-not $Run) {
        throw "No run directory found in $RunsRoot with pattern $Pattern"
    }

    return $Run.FullName
}

function Copy-DirIfExists {
    param(
        [string]$Src,
        [string]$Dest
    )

    if (Test-Path $Src) {
        if (Test-Path $Dest) {
            Remove-Item $Dest -Recurse -Force -ErrorAction SilentlyContinue
        }
        New-Item -ItemType Directory -Force -Path (Split-Path $Dest -Parent) | Out-Null
        Copy-Item $Src $Dest -Recurse -Force
    }
}

$LogsRoot = Join-Path $WorkspaceRoot "logs"
$PackagesRoot = Join-Path $WorkspaceRoot "packages"
New-Item -ItemType Directory -Force -Path $LogsRoot | Out-Null
New-Item -ItemType Directory -Force -Path $PackagesRoot | Out-Null

$Timestamp = Get-TimeStamp
$TranscriptPath = Join-Path $LogsRoot ("human_talk_confusion_eval_${Timestamp}_cli_output.txt")

Start-Transcript -Path $TranscriptPath -Force | Out-Null

try {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Human-talk confusion matrix export"
    Write-Host "============================================================"
    Write-Host "WorkspaceRoot:  $WorkspaceRoot"
    Write-Host "Stages:         $($Stages -join ', ')"
    Write-Host "Device:         $Device"
    Write-Host "ThresholdMode:  $ThresholdMode"
    Write-Host "ZipOnly:        $ZipOnly"
    Write-Host "ZipResults:     $ZipResults"
    Write-Host "Transcript:     $TranscriptPath"
    Write-Host "============================================================"

    if (-not $ZipOnly) {
        foreach ($Stage in $Stages) {
            $VariantPrefix = Get-VariantPrefix -StageName $Stage
            $RunsRoot = Join-Path $WorkspaceRoot ("stages\$Stage\runs")

            if (-not (Test-Path $RunsRoot)) {
                throw "RunsRoot not found: $RunsRoot"
            }

            Write-Host ""
            Write-Host "------------------------------------------------------------"
            Write-Host "Stage: $Stage"
            Write-Host "------------------------------------------------------------"

            $Run3 = Get-LatestRunDir -RunsRoot $RunsRoot -Pattern "${VariantPrefix}_3exit_nohint_*"
            $Run5 = Get-LatestRunDir -RunsRoot $RunsRoot -Pattern "${VariantPrefix}_5exit_nohint_*"

            foreach ($EvalMode in @("segment", "clip")) {
                Write-Host ""
                Write-Host "3-exit $EvalMode confusion: $Run3"
                python .\scripts\multilabel_confusion_export.py `
                    --run_dir "$Run3" `
                    --name "${VariantPrefix}_3exit_nohint" `
                    --device $Device `
                    --split test `
                    --threshold_mode $ThresholdMode `
                    --eval_mode $EvalMode `
                    --min_exit 2 `
                    --stable_k 2 `
                    --time_min_windows 2 `
                    --time_stable_k 2

                Write-Host ""
                Write-Host "5-exit $EvalMode confusion: $Run5"
                python .\scripts\multilabel_confusion_export.py `
                    --run_dir "$Run5" `
                    --name "${VariantPrefix}_5exit_nohint" `
                    --device $Device `
                    --split test `
                    --threshold_mode $ThresholdMode `
                    --eval_mode $EvalMode `
                    --min_exit 3 `
                    --stable_k 2 `
                    --time_min_windows 2 `
                    --time_stable_k 2
            }
        }
    }
    else {
        Write-Host "ZipOnly enabled: skipping confusion recomputation and packaging existing outputs."
    }

    if ($ZipResults) {
        $ZipPath = Join-Path $PackagesRoot ("human_talk_confusion_eval_results_${Timestamp}.zip")

        # IMPORTANT:
        # Use a very short temp root under repo root to avoid Windows path length problems.
        $TempRoot = Join-Path (Get-Location) ("_cmx_$Timestamp")

        if (Test-Path $TempRoot) {
            Remove-Item $TempRoot -Recurse -Force -ErrorAction SilentlyContinue
        }
        New-Item -ItemType Directory -Force -Path $TempRoot | Out-Null

        foreach ($Stage in $Stages) {
            $VariantPrefix = Get-VariantPrefix -StageName $Stage
            $StageCode = Get-StageCode -StageName $Stage
            $RunsRoot = Join-Path $WorkspaceRoot ("stages\$Stage\runs")
            $StageDest = Join-Path $TempRoot $StageCode
            New-Item -ItemType Directory -Force -Path $StageDest | Out-Null

            foreach ($Pattern in @("${VariantPrefix}_3exit_nohint_*", "${VariantPrefix}_5exit_nohint_*")) {
                $Run = Get-ChildItem $RunsRoot -Directory -Filter $Pattern |
                    Sort-Object LastWriteTime -Descending |
                    Select-Object -First 1

                if ($Run) {
                    $ModelCode = Get-ModelCodeFromRunName -RunName $Run.Name
                    $DestRun = Join-Path $StageDest $ModelCode
                    New-Item -ItemType Directory -Force -Path $DestRun | Out-Null

                    # Keep provenance without using long directory names.
                    @"
stage=$Stage
run_name=$($Run.Name)
run_path=$($Run.FullName)
model_code=$ModelCode
"@ | Out-File (Join-Path $DestRun "run_info.txt") -Encoding utf8

                    # Copy confusion outputs into short paths.
                    Copy-DirIfExists `
                        -Src (Join-Path $Run.FullName "multilabel_greedy_policy\confusion") `
                        -Dest (Join-Path $DestRun "segment_confusion")

                    Copy-DirIfExists `
                        -Src (Join-Path $Run.FullName "multilabel_clip_window_policy\confusion") `
                        -Dest (Join-Path $DestRun "clip_confusion")

                    # Copy compact top-level run files only.
                    Copy-Item "$($Run.FullName)\*.json" $DestRun -ErrorAction SilentlyContinue
                    Copy-Item "$($Run.FullName)\*.csv"  $DestRun -ErrorAction SilentlyContinue
                    Copy-Item "$($Run.FullName)\*.md"   $DestRun -ErrorAction SilentlyContinue
                    Copy-Item "$($Run.FullName)\*.txt"  $DestRun -ErrorAction SilentlyContinue
                    Copy-Item "$($Run.FullName)\*.yaml" $DestRun -ErrorAction SilentlyContinue
                    Copy-Item "$($Run.FullName)\*.yml"  $DestRun -ErrorAction SilentlyContinue
                }
            }
        }

        $LogDest = Join-Path $TempRoot "logs"
        New-Item -ItemType Directory -Force -Path $LogDest | Out-Null
        Copy-Item $TranscriptPath (Join-Path $LogDest (Split-Path $TranscriptPath -Leaf)) -Force
        Copy-Item ".\scripts\multilabel_confusion_export.py" (Join-Path $TempRoot "multilabel_confusion_export.py") -Force
        Copy-Item ".\scripts\run_human_talk_confusion_eval_all.ps1" (Join-Path $TempRoot "run_human_talk_confusion_eval_all.ps1") -Force

        @"
Human-talk confusion matrix package

Included:
- Short path structure to avoid Windows path length errors:
  - c2/3e/segment_confusion
  - c2/3e/clip_confusion
  - c2/5e/segment_confusion
  - c2/5e/clip_confusion
  - ...
- run_info.txt inside each model folder maps the short folder back to the original run.
- confusion matrix CSV/PNG outputs
- normalized confusion matrix CSV/PNG outputs
- per-label TP/FP/TN/FN tables
- CLI transcript
- exporter and runner scripts

Excluded:
- raw audio
- generated segment WAVs
- feature .npy files
- model checkpoints
"@ | Out-File (Join-Path $TempRoot "README_confusion_eval_package.txt") -Encoding utf8

        Compress-Archive -Path (Join-Path $TempRoot "*") -DestinationPath $ZipPath -Force
        Remove-Item $TempRoot -Recurse -Force -ErrorAction SilentlyContinue

        Write-Host ""
        Write-Host "Created ZIP:"
        Write-Host "  $ZipPath"
    }

    Write-Host ""
    Write-Host "Finished confusion matrix export."
    Write-Host "Transcript:"
    Write-Host "  $TranscriptPath"
}
finally {
    Stop-Transcript | Out-Null
}
