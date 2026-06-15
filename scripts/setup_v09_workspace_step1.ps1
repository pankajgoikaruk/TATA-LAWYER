[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\Users\wwwsa\PycharmProjects\TATA-LAWYER",
    [string]$LegacyRoot = "C:\Users\wwwsa\PycharmProjects\NeuroAccuExit-ASHADIP"
)

$ErrorActionPreference = "Stop"
$V09Root = Join-Path $RepoRoot "human_talk_workspace\tata_v0.9_pipeline"

$Directories = @(
    "$V09Root\tata_triage_model\metadata",
    "$V09Root\tata_triage_model\feature_cache\metadata",
    "$V09Root\tata_triage_model\feature_cache\features",
    "$V09Root\tata_triage_model\runs",
    "$V09Root\tata_triage_model\routing_outputs",
    "$V09Root\neuroaccuexit_main_model\metadata",
    "$V09Root\neuroaccuexit_main_model\feature_cache\features",
    "$V09Root\neuroaccuexit_main_model\runs",
    "$V09Root\neuroaccuexit_main_model\evaluation",
    "$V09Root\shared\corrected_holdout",
    "$V09Root\shared\correction_ledgers"
)

Write-Host "`n[STEP 1] Creating canonical v0.9 directories..." -ForegroundColor Cyan
foreach ($Directory in $Directories) {
    New-Item -ItemType Directory -Path $Directory -Force | Out-Null
}

$BaselineFiles = @(
    @{
        Name = "tata2_parent_manifest_12label_2074_BASELINE.csv"
        Source = Join-Path $LegacyRoot "human_talk_workspace\tata_2\metadata\tata_clip_level_manifest_training_ready.csv"
        Destination = Join-Path $V09Root "tata_triage_model\metadata\tata2_parent_manifest_12label_2074_BASELINE.csv"
        Purpose = "Verified 2,074-parent tata_2 source manifest before 10-label conversion"
    },
    @{
        Name = "tata_seed_features_manifest_10label_12469_BASELINE.csv"
        Source = Join-Path $LegacyRoot "human_talk_workspace\tata_v0.6_scratch\feature_cache\metadata\multilabel_features_manifest.csv"
        Destination = Join-Path $V09Root "tata_triage_model\feature_cache\metadata\tata_seed_features_manifest_10label_12469_BASELINE.csv"
        Purpose = "Verified 12,469-segment 10-label feature manifest used by the v0.6 TATA triage model"
    },
    @{
        Name = "human_talk_10label_schema.json"
        Source = Join-Path $LegacyRoot "human_talk_workspace\tata_v0.6_scratch\metadata\tata_v06_labels.json"
        Destination = Join-Path $V09Root "shared\human_talk_10label_schema.json"
        Purpose = "Canonical 10-label schema with audience reaction merge"
    }
)

function Copy-ImmutableBaseline {
    param(
        [Parameter(Mandatory=$true)][string]$Source,
        [Parameter(Mandatory=$true)][string]$Destination
    )

    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
        throw "Required source file was not found: $Source"
    }

    $DestinationDirectory = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Path $DestinationDirectory -Force | Out-Null

    if (Test-Path -LiteralPath $Destination -PathType Leaf) {
        $SourceHash = (Get-FileHash -LiteralPath $Source -Algorithm SHA256).Hash
        $DestinationHash = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash

        if ($SourceHash -ne $DestinationHash) {
            throw "Baseline destination already exists with different content: $Destination"
        }

        Write-Host "  Unchanged baseline already exists: $Destination" -ForegroundColor Yellow
        return
    }

    Copy-Item -LiteralPath $Source -Destination $Destination
    Write-Host "  Copied: $Destination" -ForegroundColor Green
}

Write-Host "`n[STEP 1] Copying verified baseline metadata..." -ForegroundColor Cyan
foreach ($File in $BaselineFiles) {
    Copy-ImmutableBaseline -Source $File.Source -Destination $File.Destination
}

$InventoryRows = foreach ($File in $BaselineFiles) {
    $Info = Get-Item -LiteralPath $File.Destination
    $Hash = Get-FileHash -LiteralPath $File.Destination -Algorithm SHA256

    [PSCustomObject]@{
        asset_name       = $File.Name
        purpose          = $File.Purpose
        source_path      = $File.Source
        baseline_path    = $File.Destination
        size_bytes       = $Info.Length
        sha256           = $Hash.Hash
        captured_utc     = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
        immutable_status = "BASELINE_DO_NOT_EDIT"
    }
}

$InventoryPath = Join-Path $V09Root "shared\correction_ledgers\v09_baseline_inventory.csv"
$InventoryRows | Export-Csv -LiteralPath $InventoryPath -NoTypeInformation -Encoding UTF8

$ReadmePath = Join-Path $V09Root "README_V09_WORKSPACE.md"
$ReadmeContent = @"
# TATA-LAWYER v0.9 Workspace

## Model separation

- `tata_triage_model/`: reviewed seed manifests, TATA feature cache, TATA training runs, and routing outputs.
- `neuroaccuexit_main_model/`: expanded/corrected main-model manifest, model runs, and evaluation.
- `shared/`: canonical label schema, corrected holdout, and correction ledgers.

## Baseline rule

Files containing `BASELINE` are immutable evidence from earlier verified experiments.
Do not edit them. Generate corrected v0.9 manifests as new files.

## Next generated file

The next step will create:

`tata_triage_model/metadata/tata_seed_parent_manifest_v09_REVIEW.csv`

This will convert the verified 2,074-parent 12-label manifest into the canonical 10-label schema and add audit columns for manual correction.
"@
$ReadmeContent | Set-Content -LiteralPath $ReadmePath -Encoding UTF8

Write-Host "`n[COMPLETE] v0.9 workspace created safely." -ForegroundColor Green
Write-Host "Root:      $V09Root"
Write-Host "Inventory: $InventoryPath"
Write-Host "Readme:    $ReadmePath"

Write-Host "`nBaseline assets:" -ForegroundColor Cyan
$InventoryRows |
    Select-Object asset_name, size_bytes, immutable_status |
    Format-Table -AutoSize
