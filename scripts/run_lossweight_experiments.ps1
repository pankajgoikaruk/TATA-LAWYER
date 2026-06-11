param(
  [string]$Device = "cpu",
  [switch]$SkipTraining,
  [switch]$SkipThresholds,
  [switch]$SkipStaticSummary,
  [switch]$SkipPolicyEval,
  [switch]$ZipResults
)

$ErrorActionPreference = "Stop"

function Write-Section {
  param([string]$Title)
  Write-Host ""
  Write-Host "============================================================" -ForegroundColor Cyan
  Write-Host " $Title" -ForegroundColor Cyan
  Write-Host "============================================================" -ForegroundColor Cyan
}

function Require-Path {
  param(
    [string]$PathValue,
    [string]$Name
  )

  if (-not (Test-Path $PathValue)) {
    throw "Missing required path for ${Name}: $PathValue"
  }
}

function Invoke-Step {
  param(
    [string]$Title,
    [scriptblock]$Block
  )

  Write-Section $Title
  $start = Get-Date
  & $Block
  $end = Get-Date
  $elapsed = New-TimeSpan -Start $start -End $end
  Write-Host "Completed: $Title in $($elapsed.ToString())" -ForegroundColor Green
}

function Get-LatestRunDir {
  param(
    [string]$Root,
    [string]$Pattern
  )

  $run = Get-ChildItem $Root -Directory |
    Where-Object { $_.Name -like $Pattern } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

  if ($null -eq $run) {
    throw "Could not find run directory in '$Root' using pattern '$Pattern'"
  }

  return $run.FullName
}

function Run-MultilabelGreedyPolicy {
  param(
    [string]$RunDir,
    [string]$Name,
    [int]$MinExit,
    [int]$StableK,
    [string]$SweepMinExits,
    [string]$SweepStableK,
    [string]$OutDir,
    [switch]$AllowEmptyStop
  )

  $args = @(
    "scripts\multilabel_greedy_policy.py",
    "--run_dir", $RunDir,
    "--name", $Name,
    "--device", $Device,
    "--threshold_mode", "tuned_per_exit",
    "--min_exit", $MinExit,
    "--stable_k", $StableK,
    "--sweep_min_exits", $SweepMinExits,
    "--sweep_stable_k", $SweepStableK,
    "--out_dir", $OutDir
  )

  if ($AllowEmptyStop) {
    $args += "--allow_empty_stop"
  }

  python @args
}

Write-Section "NeuroAccuExit multi-label early-exit loss-weight experiment"

$ExpectedBranch = "kexit_multi-label_EE_lossweight"

try {
  $CurrentBranch = (git branch --show-current).Trim()
  Write-Host "Current branch: $CurrentBranch"
  if ($CurrentBranch -ne $ExpectedBranch) {
    Write-Host "Switching to branch: $ExpectedBranch" -ForegroundColor Yellow
    git fetch origin
    git checkout $ExpectedBranch
  }
} catch {
  Write-Host "Git branch check failed. Continuing, but make sure you are in the project root." -ForegroundColor Yellow
}

$MANIFEST = "multilabel_cache\metadata\multilabel_features_manifest.csv"
$FEATURES_ROOT = "multilabel_cache\features"
$LABELS_JSON = "multilabel_data\metadata\labels.json"

$ROOT = "runs_multilabel"
$LW_ROOT = "$ROOT\lossweight"

$TRAIN_LW = "$LW_ROOT\training\multilabel_posweight_lossweight"
$POLICY_LW = "$LW_ROOT\policy_eval\multilabel_greedy_policy"
$SUMMARY_LW = "$LW_ROOT\summary"

Require-Path $MANIFEST "MANIFEST"
Require-Path $FEATURES_ROOT "FEATURES_ROOT"
Require-Path $LABELS_JSON "LABELS_JSON"

New-Item -ItemType Directory -Force -Path $TRAIN_LW | Out-Null
New-Item -ItemType Directory -Force -Path $POLICY_LW | Out-Null
New-Item -ItemType Directory -Force -Path $SUMMARY_LW | Out-Null

Write-Host "DEVICE        = $Device"
Write-Host "MANIFEST      = $MANIFEST"
Write-Host "FEATURES_ROOT = $FEATURES_ROOT"
Write-Host "LABELS_JSON   = $LABELS_JSON"
Write-Host "TRAIN_LW      = $TRAIN_LW"
Write-Host "POLICY_LW     = $POLICY_LW"
Write-Host "SUMMARY_LW    = $SUMMARY_LW"

if (-not $SkipTraining) {
  Invoke-Step "Train LW-001: 3-exit pos-weight, loss_weights 0.6,0.6,1.0" {
    python -m training.train_multilabel `
      --manifest $MANIFEST `
      --features_root $FEATURES_ROOT `
      --labels_json $LABELS_JSON `
      --runs_root $TRAIN_LW `
      --variant "multilabel_lw_001_3exit_posweight_w060_060_100" `
      --tap_blocks "1,3" `
      --epochs 40 `
      --batch_size 64 `
      --lr 0.001 `
      --device $Device `
      --use_pos_weight `
      --pos_weight_max 20.0 `
      --loss_weights "0.6,0.6,1.0"
  }

  Invoke-Step "Train LW-002: 3-exit pos-weight, loss_weights 0.8,0.6,1.0" {
    python -m training.train_multilabel `
      --manifest $MANIFEST `
      --features_root $FEATURES_ROOT `
      --labels_json $LABELS_JSON `
      --runs_root $TRAIN_LW `
      --variant "multilabel_lw_002_3exit_posweight_w080_060_100" `
      --tap_blocks "1,3" `
      --epochs 40 `
      --batch_size 64 `
      --lr 0.001 `
      --device $Device `
      --use_pos_weight `
      --pos_weight_max 20.0 `
      --loss_weights "0.8,0.6,1.0"
  }

  Invoke-Step "Train LW-003: 5-exit pos-weight, loss_weights 0.6,0.6,0.7,0.9,1.0" {
    python -m training.train_multilabel `
      --manifest $MANIFEST `
      --features_root $FEATURES_ROOT `
      --labels_json $LABELS_JSON `
      --runs_root $TRAIN_LW `
      --variant "multilabel_lw_003_5exit_posweight_w060_060_070_090_100" `
      --tap_blocks "1,2,3,4" `
      --epochs 40 `
      --batch_size 64 `
      --lr 0.001 `
      --device $Device `
      --use_pos_weight `
      --pos_weight_max 20.0 `
      --loss_weights "0.6,0.6,0.7,0.9,1.0"
  }

  Invoke-Step "Train LW-004: 5-exit pos-weight, loss_weights 0.8,0.7,0.7,0.9,1.0" {
    python -m training.train_multilabel `
      --manifest $MANIFEST `
      --features_root $FEATURES_ROOT `
      --labels_json $LABELS_JSON `
      --runs_root $TRAIN_LW `
      --variant "multilabel_lw_004_5exit_posweight_w080_070_070_090_100" `
      --tap_blocks "1,2,3,4" `
      --epochs 40 `
      --batch_size 64 `
      --lr 0.001 `
      --device $Device `
      --use_pos_weight `
      --pos_weight_max 20.0 `
      --loss_weights "0.8,0.7,0.7,0.9,1.0"
  }
} else {
  Write-Host "Skipping training." -ForegroundColor Yellow
}

Invoke-Step "Re-detect latest loss-weight run directories" {
  $script:RUN_LW_001_3EXIT = Get-LatestRunDir -Root $TRAIN_LW -Pattern "multilabel_lw_001_3exit_posweight_w060_060_100_*"
  $script:RUN_LW_002_3EXIT = Get-LatestRunDir -Root $TRAIN_LW -Pattern "multilabel_lw_002_3exit_posweight_w080_060_100_*"
  $script:RUN_LW_003_5EXIT = Get-LatestRunDir -Root $TRAIN_LW -Pattern "multilabel_lw_003_5exit_posweight_w060_060_070_090_100_*"
  $script:RUN_LW_004_5EXIT = Get-LatestRunDir -Root $TRAIN_LW -Pattern "multilabel_lw_004_5exit_posweight_w080_070_070_090_100_*"

  Write-Host "RUN_LW_001_3EXIT = $RUN_LW_001_3EXIT"
  Write-Host "RUN_LW_002_3EXIT = $RUN_LW_002_3EXIT"
  Write-Host "RUN_LW_003_5EXIT = $RUN_LW_003_5EXIT"
  Write-Host "RUN_LW_004_5EXIT = $RUN_LW_004_5EXIT"
}

$RUN_LW_DIRS = @(
  $RUN_LW_001_3EXIT,
  $RUN_LW_002_3EXIT,
  $RUN_LW_003_5EXIT,
  $RUN_LW_004_5EXIT
)

$LW_NAMES = @(
  "3exit_lw060_posweight",
  "3exit_lw080_posweight",
  "5exit_lw060_posweight",
  "5exit_lw080_posweight"
)

Invoke-Step "Verify saved loss weights" {
  foreach ($r in $RUN_LW_DIRS) {
    $cfgPath = Join-Path $r "config_used.json"
    Require-Path $cfgPath "config_used.json"

    $cfg = Get-Content $cfgPath -Raw | ConvertFrom-Json
    $lw = ($cfg.loss_weights -join ",")
    Write-Host "$($r): loss_weights = [$lw]" -ForegroundColor Green
  }
}

if (-not $SkipThresholds) {
  Invoke-Step "Tune thresholds for LW-001" {
    python scripts\tune_multilabel_thresholds.py `
      --run_dir $RUN_LW_001_3EXIT `
      --device $Device
  }

  Invoke-Step "Tune thresholds for LW-002" {
    python scripts\tune_multilabel_thresholds.py `
      --run_dir $RUN_LW_002_3EXIT `
      --device $Device
  }

  Invoke-Step "Tune thresholds for LW-003" {
    python scripts\tune_multilabel_thresholds.py `
      --run_dir $RUN_LW_003_5EXIT `
      --device $Device
  }

  Invoke-Step "Tune thresholds for LW-004" {
    python scripts\tune_multilabel_thresholds.py `
      --run_dir $RUN_LW_004_5EXIT `
      --device $Device
  }
} else {
  Write-Host "Skipping threshold tuning." -ForegroundColor Yellow
}

Invoke-Step "Verify threshold files" {
  foreach ($r in $RUN_LW_DIRS) {
    $path = Join-Path $r "threshold_tuning\threshold_comparison.json"
    if (Test-Path $path) {
      Write-Host "FOUND:   $path" -ForegroundColor Green
    } else {
      throw "MISSING threshold file: $path"
    }
  }
}

if (-not $SkipStaticSummary) {
  Invoke-Step "Static threshold summary for loss-weight models" {
    python scripts\summarize_multilabel_threshold_runs.py `
      --run_dirs $RUN_LW_DIRS `
      --names $LW_NAMES `
      --out_dir "$SUMMARY_LW\threshold_summary_001_lossweight_static_tuned"
  }
} else {
  Write-Host "Skipping static summary." -ForegroundColor Yellow
}

if (-not $SkipPolicyEval) {
  Invoke-Step "Policy 001: min_exit=2, stable_k=2" {
    $EXP_LW_001 = "$POLICY_LW\lossweight_policy_001_minexit2_stable2"

    Run-MultilabelGreedyPolicy `
      -RunDir $RUN_LW_001_3EXIT `
      -Name "3exit_lw060_posweight" `
      -MinExit 2 `
      -StableK 2 `
      -SweepMinExits "2" `
      -SweepStableK "1,2,3" `
      -OutDir "$EXP_LW_001\3exit_lw060_posweight"

    Run-MultilabelGreedyPolicy `
      -RunDir $RUN_LW_002_3EXIT `
      -Name "3exit_lw080_posweight" `
      -MinExit 2 `
      -StableK 2 `
      -SweepMinExits "2" `
      -SweepStableK "1,2,3" `
      -OutDir "$EXP_LW_001\3exit_lw080_posweight"

    Run-MultilabelGreedyPolicy `
      -RunDir $RUN_LW_003_5EXIT `
      -Name "5exit_lw060_posweight" `
      -MinExit 2 `
      -StableK 2 `
      -SweepMinExits "2,3" `
      -SweepStableK "1,2,3" `
      -OutDir "$EXP_LW_001\5exit_lw060_posweight"

    Run-MultilabelGreedyPolicy `
      -RunDir $RUN_LW_004_5EXIT `
      -Name "5exit_lw080_posweight" `
      -MinExit 2 `
      -StableK 2 `
      -SweepMinExits "2,3" `
      -SweepStableK "1,2,3" `
      -OutDir "$EXP_LW_001\5exit_lw080_posweight"
  }

  Invoke-Step "Policy 002: min_exit=1, stable_k=2" {
    $EXP_LW_002 = "$POLICY_LW\lossweight_policy_002_minexit1_stable2"

    Run-MultilabelGreedyPolicy `
      -RunDir $RUN_LW_001_3EXIT `
      -Name "3exit_lw060_posweight" `
      -MinExit 1 `
      -StableK 2 `
      -SweepMinExits "1,2" `
      -SweepStableK "1,2,3" `
      -OutDir "$EXP_LW_002\3exit_lw060_posweight"

    Run-MultilabelGreedyPolicy `
      -RunDir $RUN_LW_002_3EXIT `
      -Name "3exit_lw080_posweight" `
      -MinExit 1 `
      -StableK 2 `
      -SweepMinExits "1,2" `
      -SweepStableK "1,2,3" `
      -OutDir "$EXP_LW_002\3exit_lw080_posweight"

    Run-MultilabelGreedyPolicy `
      -RunDir $RUN_LW_003_5EXIT `
      -Name "5exit_lw060_posweight" `
      -MinExit 1 `
      -StableK 2 `
      -SweepMinExits "1,2,3" `
      -SweepStableK "1,2,3" `
      -OutDir "$EXP_LW_002\5exit_lw060_posweight"

    Run-MultilabelGreedyPolicy `
      -RunDir $RUN_LW_004_5EXIT `
      -Name "5exit_lw080_posweight" `
      -MinExit 1 `
      -StableK 2 `
      -SweepMinExits "1,2,3" `
      -SweepStableK "1,2,3" `
      -OutDir "$EXP_LW_002\5exit_lw080_posweight"
  }

  Invoke-Step "Best 5-exit sweep target: min_exit=3, stable_k=2" {
    $EXP_LW_BEST5 = "$POLICY_LW\lossweight_policy_best_5exit_minexit3_stable2"

    Run-MultilabelGreedyPolicy `
      -RunDir $RUN_LW_003_5EXIT `
      -Name "5exit_lw060_posweight" `
      -MinExit 3 `
      -StableK 2 `
      -SweepMinExits "3" `
      -SweepStableK "1,2,3" `
      -OutDir "$EXP_LW_BEST5\5exit_lw060_posweight"

    Run-MultilabelGreedyPolicy `
      -RunDir $RUN_LW_004_5EXIT `
      -Name "5exit_lw080_posweight" `
      -MinExit 3 `
      -StableK 2 `
      -SweepMinExits "3" `
      -SweepStableK "1,2,3" `
      -OutDir "$EXP_LW_BEST5\5exit_lw080_posweight"
  }
} else {
  Write-Host "Skipping policy evaluation." -ForegroundColor Yellow
}

if ($ZipResults) {
  Invoke-Step "Zip loss-weight results" {
    $zipPath = "$ROOT\lossweight_results.zip"
    if (Test-Path $zipPath) {
      Remove-Item $zipPath -Force
    }

    Compress-Archive `
      -Path "$LW_ROOT\*" `
      -DestinationPath $zipPath `
      -Force

    Write-Host "Created ZIP: $zipPath" -ForegroundColor Green
  }
}

Write-Section "Finished loss-weight experiment sequence"

Write-Host "Main output root:" -ForegroundColor Green
Write-Host "  $LW_ROOT"

Write-Host ""
Write-Host "Recommended folder to share for analysis:" -ForegroundColor Green
Write-Host "  runs_multilabel\lossweight\"

Write-Host ""
Write-Host "To zip results manually:" -ForegroundColor Yellow
Write-Host '  Compress-Archive -Path "runs_multilabel\lossweight\*" -DestinationPath "runs_multilabel\lossweight_results.zip" -Force'
