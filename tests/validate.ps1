[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repositoryRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repositoryRoot

Import-Module (Join-Path $PSScriptRoot 'TestPython.psm1') -Force
$testPython = Resolve-TestPython
$testPythonExecutable = $testPython[0]
$testPythonArguments = @($testPython | Select-Object -Skip 1)

$distributionScriptFiles = @()
$distributionScriptRoot = Join-Path $repositoryRoot 'ComfyUI-Windows'
if (Test-Path -LiteralPath $distributionScriptRoot -PathType Container) {
  $distributionScriptFiles = @(
    Get-ChildItem -LiteralPath $distributionScriptRoot -File -Filter '*.ps1'
  )
}

function Assert-NativeSuccess {
  param([Parameter(Mandatory = $true)] [string] $Description)

  if ($LASTEXITCODE -ne 0) {
    throw "$Description failed with exit code $LASTEXITCODE."
  }
}

$powerShellFiles = @(
  Get-ChildItem -LiteralPath $repositoryRoot -File -Filter '*.ps1' |
    Where-Object Name -NotIn @('install-broken.ps1', 'install.old.ps1')
  $distributionScriptFiles
  Get-ChildItem -LiteralPath (Join-Path $repositoryRoot 'scripts') -File -Filter '*.ps1'
  Get-ChildItem -LiteralPath (Join-Path $repositoryRoot 'scripts') -File -Filter '*.psm1'
  Get-ChildItem -LiteralPath $PSScriptRoot -File -Filter '*.ps1'
  Get-ChildItem -LiteralPath $PSScriptRoot -File -Filter '*.psm1'
)

foreach ($file in $powerShellFiles) {
  $tokens = $null
  $errors = $null
  [void] [Management.Automation.Language.Parser]::ParseFile(
    $file.FullName,
    [ref] $tokens,
    [ref] $errors
  )
  if ($errors.Count -gt 0) {
    $messages = ($errors | ForEach-Object Message) -join [Environment]::NewLine
    throw "PowerShell parse failed for $($file.FullName):$([Environment]::NewLine)$messages"
  }
}
Write-Output "PASS: parsed $($powerShellFiles.Count) PowerShell files"

$pythonFiles = @(
  Get-ChildItem -LiteralPath $repositoryRoot -File -Filter '*.py'
  Get-ChildItem -LiteralPath (Join-Path $repositoryRoot 'patches') -File -Filter '*.py'
  Get-ChildItem -LiteralPath (Join-Path $repositoryRoot 'scripts') -File -Filter '*.py'
)
$pythonPaths = @($pythonFiles | ForEach-Object FullName)
$compileCode = "import ast,pathlib,sys; [ast.parse(pathlib.Path(p).read_text(encoding='utf-8-sig'), filename=p) for p in sys.argv[1:]]"
& $testPythonExecutable @testPythonArguments '-c' $compileCode @pythonPaths
Assert-NativeSuccess 'Python source compilation'
Write-Output "PASS: compiled $($pythonFiles.Count) Python files without writing bytecode"

foreach ($workflow in Get-ChildItem -LiteralPath (Join-Path $repositoryRoot 'examples') -File -Filter '*.json') {
  Get-Content -Raw -LiteralPath $workflow.FullName | ConvertFrom-Json | Out-Null
}
Write-Output 'PASS: parsed example workflow JSON files'

$modelManifestPath = Join-Path $repositoryRoot 'config\models.json'
$modelManifest = Get-Content -Raw -LiteralPath $modelManifestPath | ConvertFrom-Json
if ($modelManifest.schema_version -ne 1 -or @($modelManifest.wan.artifacts).Count -eq 0) {
  throw 'Model manifest schema or artifacts are invalid.'
}
$allowedManifestDestinations = @('checkpoints', 'diffusion_models', 'text_encoders', 'vae', 'loras', 'latent_upscale_models')
foreach ($manifestFamily in @($modelManifest.PSObject.Properties | Where-Object Name -ne 'schema_version')) {
  if ([string] $manifestFamily.Value.repository -notmatch '^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$') {
    throw "Model family '$($manifestFamily.Name)' repository is invalid."
  }
  if ([string] $manifestFamily.Value.revision -notmatch '^[A-Za-z0-9._/-]+$') {
    throw "Model family '$($manifestFamily.Name)' revision is invalid."
  }
  if (@($manifestFamily.Value.artifacts).Count -eq 0) {
    throw "Model family '$($manifestFamily.Name)' declares no artifacts."
  }
  foreach ($manifestArtifact in $manifestFamily.Value.artifacts) {
    if (@($manifestArtifact.groups).Count -eq 0) {
      throw "Model family '$($manifestFamily.Name)' has an artifact without groups."
    }
    if ([string] $manifestArtifact.destination -notin $allowedManifestDestinations) {
      throw "Model family '$($manifestFamily.Name)' uses an unsupported destination: $($manifestArtifact.destination)"
    }
    $manifestArtifactPath = [string] $manifestArtifact.path
    if ($manifestArtifactPath -notmatch '^[A-Za-z0-9][A-Za-z0-9._/-]*$' -or $manifestArtifactPath.Contains('..')) {
      throw "Model family '$($manifestFamily.Name)' has an unsafe artifact path: $manifestArtifactPath"
    }
    if ([string] $manifestArtifact.sha256 -notmatch '^[0-9a-f]{64}$') {
      throw "Model family '$($manifestFamily.Name)' artifact lacks a valid sha256: $manifestArtifactPath"
    }
  }
}
Write-Output 'PASS: parsed the versioned model manifest across all families'

$nodeManifestPath = Join-Path $repositoryRoot 'config\nodes.json'
$nodeManifest = Get-Content -Raw -LiteralPath $nodeManifestPath | ConvertFrom-Json
if ($nodeManifest.schema_version -ne 1 -or @($nodeManifest.nodes).Count -eq 0) {
  throw 'Node manifest schema or node list is invalid.'
}
foreach ($manifestNode in $nodeManifest.nodes) {
  if ([string] $manifestNode.name -notmatch '^[A-Za-z0-9._-]+$') {
    throw "Node manifest contains an unsafe name: $($manifestNode.name)"
  }
  if ([string] $manifestNode.repository -notmatch '^https://github\.com/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+\.git$') {
    throw "Node manifest contains an unsupported repository: $($manifestNode.repository)"
  }
  if ([string] $manifestNode.revision -notmatch '^[0-9a-f]{7,40}$') {
    throw "Node manifest revision must be a pinned commit: $($manifestNode.revision)"
  }
}
Write-Output 'PASS: parsed the versioned node manifest with pinned commits'

$workflowSanitizer = Join-Path $repositoryRoot 'scripts\sanitize_workflows.py'
& $testPythonExecutable @testPythonArguments $workflowSanitizer
Assert-NativeSuccess 'Workflow privacy metadata validation'

$trackedFiles = @(& git ls-files)
Assert-NativeSuccess 'Tracked-file inventory'
$escapedBackslash = [Regex]::Escape([string] [char] 92)
$forbiddenPatterns = @(
  ('(?i)[a-z]:\\' + 'Users\\'),
  ('(?i)[a-z]:\\[^\r\n]*\\python-' + 'projects\\'),
  ('(?i)(?<![a-z])[a-z]:' + '\\'),
  ('(?i)(?<![a-z])[a-z]:' + '/'),
  ('(?i)/ho' + 'me/[^/\s]+/'),
  ('(?i)/Us' + 'ers/[^/\s]+/'),
  ('(?i)/(?:work' + 'space|data|dev_share|root)/(?:[^\s"''<>]+/)*[^\s"''<>]*'),
  ('(?i)https://liblibai-tmp-image\.' + 'liblib\.cloud'),
  ('(?i)https?://[^\s"''<>]+[?&](?:token|access_token|api_key|signature|x-amz-signature)='),
  ('(?i)\b[A-Z0-9._%+-]+@(?!(?:example\.(?:com|org|net)|github\.com)\b)[A-Z0-9.-]+\.[A-Z]{2,}\b')
)
$uncPatterns = @(
  "(?i)(?<![\p{L}\p{N}_.-])$escapedBackslash$escapedBackslash[^$escapedBackslash\s]+$escapedBackslash",
  "(?i)$escapedBackslash$escapedBackslash$escapedBackslash$escapedBackslash[^$escapedBackslash\s]+$escapedBackslash$escapedBackslash"
)
$forbiddenPatterns += $uncPatterns
$machineSpecificValues = @(
  $env:USERNAME,
  $env:COMPUTERNAME,
  $env:USERDOMAIN,
  $env:USERPROFILE
) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) -and $_.Length -ge 4 } |
  Sort-Object -Unique
$textExtensions = @(
  '.bat', '.cjs', '.cmd', '.css', '.html', '.ini', '.js', '.json', '.md',
  '.mjs', '.ps1', '.psm1', '.py', '.sh', '.toml', '.ts', '.tsx', '.txt',
  '.xml', '.yml', '.yaml'
)
foreach ($trackedFile in $trackedFiles) {
  $trackedExtension = [IO.Path]::GetExtension($trackedFile)
  if ($trackedExtension -ieq '.lnk') {
    throw "Portable source repositories cannot track binary Windows shortcuts: $trackedFile"
  }
  foreach ($pattern in $forbiddenPatterns) {
    if ($trackedFile -match $pattern) {
      throw "Tracked filename contains a personal or machine-specific value: $trackedFile"
    }
  }
  foreach ($value in $machineSpecificValues) {
    if ($trackedFile.IndexOf($value, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
      throw "Tracked filename identifies the current machine or account: $trackedFile"
    }
  }
  if ($textExtensions -notcontains $trackedExtension) {
    continue
  }
  $fullTrackedPath = Join-Path $repositoryRoot $trackedFile
  if (-not (Test-Path -LiteralPath $fullTrackedPath -PathType Leaf)) {
    continue
  }
  $content = Get-Content -Raw -LiteralPath $fullTrackedPath -ErrorAction Stop
  $searchableText = $content
  foreach ($pattern in $forbiddenPatterns) {
    if (
      $trackedFile -in @('tests/validate.ps1', 'scripts/sanitize_workflows.py') -and
      $uncPatterns -contains $pattern
    ) {
      continue
    }
    if ($searchableText -match $pattern) {
      throw "Tracked file contains a personal or machine-specific value: $trackedFile"
    }
  }
  foreach ($value in $machineSpecificValues) {
    if ($searchableText.IndexOf($value, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
      throw "Tracked file contains a value identifying the current machine or account: $trackedFile"
    }
  }
}
Write-Output 'PASS: tracked text paths and contents contain no personal or machine-specific values'

if (-not (Get-Command ruff -ErrorAction SilentlyContinue)) {
  throw 'ruff is required for source validation.'
}
& ruff check --exclude ComfyUI --exclude hf_cache --exclude Pixal3D-upstream-test .
Assert-NativeSuccess 'Ruff'

if ($IsWindows) {
  $gitExecutable = (Get-Command git -ErrorAction Stop).Source
  $gitRoot = Split-Path -Parent (Split-Path -Parent $gitExecutable)
  $bash = Join-Path $gitRoot 'bin\bash.exe'
  if (-not (Test-Path -LiteralPath $bash -PathType Leaf)) {
    $bash = (Get-Command bash -ErrorAction Stop).Source
  }
}
else {
  $bash = (Get-Command bash -ErrorAction Stop).Source
}
& $bash -n install.sh installme.sh
Assert-NativeSuccess 'Bash syntax validation'

$unixDryRun = @(
  & $bash install.sh --cuda=cu128 --models=5b --with-manager=false --start=false --listen-all=false --reuse-venv=false --dry-run=true
)
$unixDryRun | Write-Output
Assert-NativeSuccess 'Unix installer dry run'
$targetVenvPython = if ($IsWindows) {
  Join-Path $repositoryRoot 'ComfyUI\.venv\Scripts\python.exe'
}
else {
  Join-Path $repositoryRoot 'ComfyUI/.venv/bin/python'
}
$unsafeRecreationPattern = [Regex]::Escape($targetVenvPython) + '\s+-m\s+venv\s+'
if (($unixDryRun -join [Environment]::NewLine) -match $unsafeRecreationPattern) {
  throw 'Unix installer cannot recreate a venv using the interpreter inside that same venv.'
}

if ($IsWindows) {
  $npm = 'npm.cmd'
}
else {
  $npm = 'npm'
}
$metadataPaths = @('package.json', 'package-lock.json')
$metadataBefore = @{}
foreach ($metadataPath in $metadataPaths) {
  $metadataBefore[$metadataPath] = (Get-FileHash -LiteralPath $metadataPath -Algorithm SHA256).Hash
}
& $npm install --package-lock-only --ignore-scripts --offline
Assert-NativeSuccess 'Offline npm lockfile validation'
foreach ($metadataPath in $metadataPaths) {
  $metadataAfter = (Get-FileHash -LiteralPath $metadataPath -Algorithm SHA256).Hash
  if ($metadataBefore[$metadataPath] -cne $metadataAfter) {
    throw "Offline npm lockfile validation changed $metadataPath; commit synchronized metadata first."
  }
}
Write-Output 'PASS: npm metadata is internally consistent'

Write-Output 'PASS: repository validation suite'
