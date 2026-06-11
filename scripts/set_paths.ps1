$code = @'
param(
  [ValidateSet("v0","v1")] [string]$Mode,
  [string]$Config = "configs\audio_moth.yaml",
  [string]$CacheV0 = "data_cache_v0",
  [string]$RunsV0  = "runs_v0",
  [string]$CacheV1 = "data_cache_v1",
  [string]$RunsV1  = "runs_v1"
)
$ErrorActionPreference = "Stop"

if (-not (Test-Path $Config)) { throw "Config not found: $Config" }

# Use Python+PyYAML to safely edit YAML
$py = @"
import sys, yaml, io
cfg_path = sys.argv[1]
mode     = sys.argv[2]
cache_v0, runs_v0, cache_v1, runs_v1 = sys.argv[3:7]

with open(cfg_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f) or {}

cfg.setdefault('paths', {})

if mode.lower() == 'v0':
    cfg['paths']['cache_root'] = cache_v0
    cfg['paths']['runs_root']  = runs_v0
else:
    cfg['paths']['cache_root'] = cache_v1
    cfg['paths']['runs_root']  = runs_v1

with open(cfg_path, 'w', encoding='utf-8') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print("OK:", cfg['paths'])
"@

python - <<PY $Config $Mode $CacheV0 $RunsV0 $CacheV1 $RunsV1
$py
PY
'@
New-Item -ItemType Directory -Path scripts -ErrorAction SilentlyContinue | Out-Null
Set-Content -Path scripts\set_paths.ps1 -Value $code -Encoding UTF8
Write-Host "Created scripts\set_paths.ps1" -ForegroundColor Green
