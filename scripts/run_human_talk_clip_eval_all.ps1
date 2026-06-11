# scripts/run_human_talk_clip_eval_all.ps1
#
# Rerun clip/window-level evaluation for existing clean-stage trained runs.
# This does NOT retrain models and does NOT rebuild features.
#
# Example:
#   powershell -ExecutionPolicy Bypass -File .\scripts\run_human_talk_clip_eval_all.ps1 `
#     -WorkspaceRoot human_talk_workspace `
#     -Device cpu `
#     -ZipResults

param(
    [string]$WorkspaceRoot = "human_talk_workspace",
    [string[]]$Stages = @("clean2_balanced", "clean3_balanced", "clean4_balanced", "clean5_balanced"),
    [string]$Device = "cpu",
    [string]$ThresholdMode = "fixed_0p5",
    [int]$TimeMinWindows = 2,
    [int]$TimeStableK = 2,
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

$LogsRoot = Join-Path $WorkspaceRoot "logs"
$PackagesRoot = Join-Path $WorkspaceRoot "packages"
New-Item -ItemType Directory -Force -Path $LogsRoot | Out-Null
New-Item -ItemType Directory -Force -Path $PackagesRoot | Out-Null

$Timestamp = Get-TimeStamp
$TranscriptPath = Join-Path $LogsRoot ("human_talk_clip_window_eval_${Timestamp}_cli_output.txt")
Start-Transcript -Path $TranscriptPath -Force | Out-Null

try {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host " Human-talk clip/window-level evaluation"
    Write-Host "============================================================"
    Write-Host "WorkspaceRoot:  $WorkspaceRoot"
    Write-Host "Stages:         $($Stages -join ', ')"
    Write-Host "Device:         $Device"
    Write-Host "ThresholdMode:  $ThresholdMode"
    Write-Host "TimeMinWindows: $TimeMinWindows"
    Write-Host "TimeStableK:    $TimeStableK"
    Write-Host "Transcript:     $TranscriptPath"
    Write-Host "============================================================"

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

        Write-Host "3-exit run: $Run3"
        python .\scripts\multilabel_clip_window_policy.py `
            --run_dir "$Run3" `
            --name "${VariantPrefix}_3exit_nohint" `
            --device $Device `
            --split test `
            --threshold_mode $ThresholdMode `
            --min_exit 2 `
            --stable_k 2 `
            --time_min_windows $TimeMinWindows `
            --time_stable_k $TimeStableK

        Write-Host "5-exit run: $Run5"
        python .\scripts\multilabel_clip_window_policy.py `
            --run_dir "$Run5" `
            --name "${VariantPrefix}_5exit_nohint" `
            --device $Device `
            --split test `
            --threshold_mode $ThresholdMode `
            --min_exit 3 `
            --stable_k 2 `
            --time_min_windows $TimeMinWindows `
            --time_stable_k $TimeStableK
    }

    if ($ZipResults) {
        $ZipPath = Join-Path $PackagesRoot ("human_talk_clip_window_eval_results_${Timestamp}.zip")
        $TempRoot = Join-Path $PackagesRoot ("_clip_eval_share_${Timestamp}")
        if (Test-Path $TempRoot) {
            Remove-Item $TempRoot -Recurse -Force
        }
        New-Item -ItemType Directory -Force -Path $TempRoot | Out-Null

        foreach ($Stage in $Stages) {
            $VariantPrefix = Get-VariantPrefix -StageName $Stage
            $RunsRoot = Join-Path $WorkspaceRoot ("stages\$Stage\runs")
            $StageDest = Join-Path $TempRoot $Stage
            New-Item -ItemType Directory -Force -Path $StageDest | Out-Null

            foreach ($Pattern in @("${VariantPrefix}_3exit_nohint_*", "${VariantPrefix}_5exit_nohint_*")) {
                $Run = Get-ChildItem $RunsRoot -Directory -Filter $Pattern |
                    Sort-Object LastWriteTime -Descending |
                    Select-Object -First 1

                if ($Run) {
                    $DestRun = Join-Path $StageDest $Run.Name
                    New-Item -ItemType Directory -Force -Path $DestRun | Out-Null

                    $ClipDir = Join-Path $Run.FullName "multilabel_clip_window_policy"
                    if (Test-Path $ClipDir) {
                        Copy-Item $ClipDir (Join-Path $DestRun "multilabel_clip_window_policy") -Recurse -Force
                    }

                    $SegDir = Join-Path $Run.FullName "multilabel_greedy_policy"
                    if (Test-Path $SegDir) {
                        Copy-Item $SegDir (Join-Path $DestRun "multilabel_greedy_policy") -Recurse -Force
                    }

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
        Copy-Item ".\scripts\multilabel_clip_window_policy.py" (Join-Path $TempRoot "multilabel_clip_window_policy.py") -Force
        Copy-Item ".\scripts\run_human_talk_clip_eval_all.ps1" (Join-Path $TempRoot "run_human_talk_clip_eval_all.ps1") -Force

        @"
Human-talk clip/window-level evaluation package

Included:
- multilabel_clip_window_policy outputs for latest 3-exit and 5-exit runs per stage
- existing segment-level multilabel_greedy_policy outputs when present
- top-level run configs/summaries
- CLI transcript

Excluded:
- raw audio
- generated segment WAVs
- feature .npy files
- model checkpoints
"@ | Out-File (Join-Path $TempRoot "README_clip_window_eval_package.txt") -Encoding utf8

        Compress-Archive -Path (Join-Path $TempRoot "*") -DestinationPath $ZipPath -Force
        Remove-Item $TempRoot -Recurse -Force -ErrorAction SilentlyContinue

        Write-Host ""
        Write-Host "Created ZIP:"
        Write-Host "  $ZipPath"
    }

    Write-Host ""
    Write-Host "Finished clip/window-level evaluation."
    Write-Host "Transcript:"
    Write-Host "  $TranscriptPath"
}
finally {
    Stop-Transcript | Out-Null
}
