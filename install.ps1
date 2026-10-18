# Wan 2.2 + ComfyUI installer for Windows PowerShell.
# Compatible with the current wan2_cli.py, which exposes only the `start` command.

[CmdletBinding()]
param(
  # 'directml' targets AMD (and Intel) GPUs on Windows through torch-directml.
  [ValidateSet('cu128','cu121','cu118','directml','cpu')]
  [string] $Cuda = 'cu128',

  [ValidateSet('5b','14b','i2v','ltx','ltx2','all')]
  [string] $Models = '5b',

  [switch] $WithManager,
  [switch] $SkipNodes,
  [switch] $Start,
  [ValidateRange(1, 65535)]
  [int] $Port = 8188,
  [switch] $ListenAll,
  [string] $BasePath = $PSScriptRoot,
  [string] $PyVersion = '3.10',
  [string] $HfToken = $env:HF_TOKEN,
  [string] $ModelRepository = $env:CUSTOM_WAN_MODEL_REPOSITORY,
  [string] $ModelRevision = $env:CUSTOM_WAN_MODEL_REVISION,
  [switch] $ReuseVenv,

  [ValidateSet('Stop','Fail')]
  [string] $LockedVenvAction = 'Fail'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# -----------------------------------------------------------------------------
# Compute backend selection. An explicit -Cuda value always wins. Otherwise the
# installed display adapters decide: with both NVIDIA and AMD present the user
# chooses interactively, and an AMD-only machine defaults to DirectML.
# -----------------------------------------------------------------------------
if (-not $PSBoundParameters.ContainsKey('Cuda')) {
  $videoAdapterNames = @()
  try {
    $videoAdapterNames = @(
      Get-CimInstance -ClassName Win32_VideoController -ErrorAction Stop |
        ForEach-Object Name |
        Where-Object { $_ }
    )
  }
  catch {
    Write-Warning ("GPU detection failed; keeping the default backend '{0}'. {1}" -f $Cuda, $_.Exception.Message)
  }

  $nvidiaAdapters = @($videoAdapterNames | Where-Object { $_ -match 'NVIDIA|GeForce|Quadro' })
  $amdAdapters = @($videoAdapterNames | Where-Object { $_ -match 'AMD|Radeon' })

  if ($nvidiaAdapters.Count -gt 0 -and $amdAdapters.Count -gt 0) {
    Write-Host '[install.ps1] Both NVIDIA and AMD GPUs were detected:' -ForegroundColor Yellow
    $videoAdapterNames | ForEach-Object { Write-Host ("  - {0}" -f $_) }
    if ([Console]::IsInputRedirected) {
      Write-Host ("[install.ps1] Non-interactive session; keeping the default backend '{0}'. Pass -Cuda directml to use the AMD GPU." -f $Cuda) -ForegroundColor Yellow
    }
    else {
      $gpuAnswer = Read-Host 'Which GPU should ComfyUI use? [N] NVIDIA CUDA (default) / [A] AMD DirectML'
      if ($gpuAnswer -match '^\s*(a|amd)') {
        $Cuda = 'directml'
      }
      Write-Host ("[install.ps1] Selected backend: {0}" -f $Cuda) -ForegroundColor Cyan
    }
  }
  elseif ($amdAdapters.Count -gt 0 -and $nvidiaAdapters.Count -eq 0) {
    Write-Host '[install.ps1] An AMD GPU was detected without NVIDIA; using the DirectML backend.' -ForegroundColor Yellow
    $Cuda = 'directml'
  }
}

$modelManifestPath = Join-Path $PSScriptRoot 'config\models.json'
if (-not (Test-Path -LiteralPath $modelManifestPath -PathType Leaf)) {
  throw "Model manifest is missing: $modelManifestPath"
}
$modelManifest = Get-Content -LiteralPath $modelManifestPath -Raw | ConvertFrom-Json
if ($modelManifest.schema_version -ne 1) {
  throw 'Unsupported model manifest schema version.'
}

$nodeManifestPath = Join-Path $PSScriptRoot 'config\nodes.json'
if (-not (Test-Path -LiteralPath $nodeManifestPath -PathType Leaf)) {
  throw "Node manifest is missing: $nodeManifestPath"
}
$nodeManifest = Get-Content -LiteralPath $nodeManifestPath -Raw | ConvertFrom-Json
if ($nodeManifest.schema_version -ne 1) {
  throw 'Unsupported node manifest schema version.'
}
if ([string]::IsNullOrWhiteSpace($ModelRepository)) {
  $ModelRepository = [string] $modelManifest.wan.repository
}
if ([string]::IsNullOrWhiteSpace($ModelRevision)) {
  $ModelRevision = [string] $modelManifest.wan.revision
}

if ($ModelRepository -notmatch '^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$') {
  throw 'ModelRepository must use the owner/repository form.'
}
if ($ModelRevision -notmatch '^[A-Za-z0-9._/-]+$' -or $ModelRevision.Contains('..')) {
  throw 'ModelRevision must be a branch, tag, or commit without traversal segments.'
}

$installerVenvModule = Join-Path $PSScriptRoot 'scripts\Installer.Venv.psm1'
if (-not (Test-Path -LiteralPath $installerVenvModule)) {
  throw "Installer support module is missing: $installerVenvModule"
}
Import-Module $installerVenvModule -Force

function Invoke-Native {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory = $true)] [string] $FilePath,
    [Parameter()] [string[]] $ArgumentList = @()
  )

  Write-Host ("> {0} {1}" -f $FilePath, ($ArgumentList -join ' ')) -ForegroundColor DarkGray
  & $FilePath @ArgumentList
  $code = $LASTEXITCODE
  if ($code -ne 0) {
    throw ("Command failed with exit code {0}: {1} {2}" -f $code, $FilePath, ($ArgumentList -join ' '))
  }
}

function Ensure-Directory {
  param([Parameter(Mandatory = $true)] [string] $Path)
  if (-not (Test-Path -LiteralPath $Path)) {
    New-Item -ItemType Directory -Path $Path -Force | Out-Null
  }
}

function Test-FileSha256 {
  param(
    [Parameter(Mandatory = $true)] [string] $Path,
    [Parameter(Mandatory = $true)] [string] $Expected
  )
  $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
  return $actual -eq $Expected.ToLowerInvariant()
}

function Get-HuggingFaceFile {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory = $true)] [string] $Repository,
    [Parameter(Mandatory = $true)] [string] $Revision,
    [Parameter(Mandatory = $true)] [string] $RelativePath,
    [Parameter(Mandatory = $true)] [string] $Destination,
    [Parameter(Mandatory = $true)] [string] $Sha256
  )

  if (Test-Path -LiteralPath $Destination) {
    if (Test-FileSha256 -Path $Destination -Expected $Sha256) {
      Write-Host "[models] Existing file verified and kept: $Destination" -ForegroundColor Green
      return
    }
    Write-Warning "Existing file failed sha256 verification and will be re-downloaded: $Destination"
    Remove-Item -LiteralPath $Destination -Force
  }

  $destinationDir = Split-Path -Parent $Destination
  Ensure-Directory $destinationDir

  $encodedRepository = (($Repository -split '/') | ForEach-Object { [Uri]::EscapeDataString($_) }) -join '/'
  $encodedRevision = (($Revision -split '/') | ForEach-Object { [Uri]::EscapeDataString($_) }) -join '/'
  $repo = "https://huggingface.co/${encodedRepository}/resolve/${encodedRevision}"
  $encodedPath = (($RelativePath -split '/') | ForEach-Object { [Uri]::EscapeDataString($_) }) -join '/'
  $url = "${repo}/${encodedPath}?download=true"
  $partial = "$Destination.part"

  Write-Host "[models] Downloading: $(Split-Path -Leaf $Destination)" -ForegroundColor Cyan

  $curlArgs = @(
    '-L',
    '--fail',
    '--retry', '8',
    '--retry-delay', '5',
    '--retry-all-errors',
    '--continue-at', '-',
    '--output', $partial
  )

  if ($env:HF_TOKEN) {
    if ($env:HF_TOKEN -match '[\r\n]') {
      throw 'HF_TOKEN contains an invalid newline character.'
    }

    # curl accepts headers from stdin via @-. This keeps the bearer token out
    # of the process command line and out of Invoke-Native's command log.
    Write-Host '> curl.exe [Authorization header supplied via stdin]' -ForegroundColor DarkGray
    "Authorization: Bearer $($env:HF_TOKEN)" | & curl.exe @curlArgs '-H' '@-' $url
    $curlExitCode = $LASTEXITCODE
    if ($curlExitCode -ne 0) {
      throw "Model download failed with curl exit code $curlExitCode."
    }
  }
  else {
    $curlArgs += $url
    Invoke-Native -FilePath 'curl.exe' -ArgumentList $curlArgs
  }

  if (-not (Test-Path -LiteralPath $partial)) {
    throw "Model download completed without creating the expected file: $partial"
  }

  $downloaded = Get-Item -LiteralPath $partial
  if ($downloaded.Length -lt 1MB) {
    throw "Downloaded model file is unexpectedly small: $partial"
  }

  if (-not (Test-FileSha256 -Path $partial -Expected $Sha256)) {
    Remove-Item -LiteralPath $partial -Force
    throw "Downloaded model file failed sha256 verification: $Destination"
  }

  Move-Item -LiteralPath $partial -Destination $Destination -Force
  Write-Host "[models] Saved and sha256 verified: $Destination" -ForegroundColor Green
}

Write-Host ("[install.ps1] Base: {0}  CUDA: {1}  MODELS: {2}  Manager: {3}  SkipNodes: {4}  Start: {5}  Port: {6}  ListenAll: {7}  ReuseVenv: {8}  LockedVenvAction: {9}" `
  -f $BasePath,$Cuda,$Models,$WithManager.IsPresent,$SkipNodes.IsPresent,$Start.IsPresent,$Port,$ListenAll.IsPresent,$ReuseVenv.IsPresent,$LockedVenvAction)

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
  throw "Python launcher 'py' was not found. Install 64-bit Python $PyVersion first."
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
  throw "Git was not found. Install Git for Windows and reopen PowerShell."
}

if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
  throw "curl.exe was not found. A current Windows 10/11 installation normally includes it."
}

if ($HfToken) {
  $env:HF_TOKEN = $HfToken
}

$BasePath = [IO.Path]::GetFullPath($BasePath)
Ensure-Directory $BasePath

$comfyPath   = Join-Path $BasePath 'ComfyUI'
$mainPy      = Join-Path $comfyPath 'main.py'
$venvPath    = Join-Path $comfyPath '.venv'
$venvPython  = Join-Path $venvPath 'Scripts\python.exe'
$cliPath     = Join-Path $PSScriptRoot 'wan2_cli.py'
$runWanPath  = Join-Path $PSScriptRoot 'ComfyUI-Windows\run_wan.ps1'

# Refuse or stop scoped users of the environment before Git, pip, or filesystem
# mutations. This also prevents the run_wan.ps1 watchdog from racing deletion.
if (Test-Path -LiteralPath $venvPath) {
  Resolve-VirtualEnvironmentLock `
    -Path $venvPath `
    -SupervisorScriptPath $runWanPath `
    -LockedVenvAction $LockedVenvAction
}

function Invoke-NativeWithRetry {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory = $true)] [string] $FilePath,
    [Parameter()] [string[]] $ArgumentList = @(),
    [Parameter()] [ValidateRange(1, 10)] [int] $MaxAttempts = 3,
    [Parameter()] [ValidateRange(1, 300)] [int] $InitialDelaySeconds = 5
  )

  for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
    try {
      Invoke-Native -FilePath $FilePath -ArgumentList $ArgumentList
      return
    }
    catch {
      if ($attempt -eq $MaxAttempts) {
        throw
      }

      $delay = $InitialDelaySeconds * [Math]::Pow(2, $attempt - 1)
      Write-Warning ("Command attempt {0}/{1} failed. Retrying in {2} seconds: {3}" -f `
        $attempt, $MaxAttempts, $delay, $_.Exception.Message)
      Start-Sleep -Seconds $delay
    }
  }
}

# -----------------------------------------------------------------------------
# Install or update ComfyUI directly. Do not call `wan2_cli.py install` because
# the current CLI only implements the `start` command.
# -----------------------------------------------------------------------------
if (-not (Test-Path -LiteralPath $mainPy)) {
  if (Test-Path -LiteralPath $comfyPath) {
    $items = @(Get-ChildItem -LiteralPath $comfyPath -Force -ErrorAction SilentlyContinue)
    if ($items.Count -gt 0) {
      throw "The ComfyUI directory exists but main.py is missing: $comfyPath. Rename/remove that incomplete directory and run again."
    }
  }

  Write-Host '[install.ps1] Cloning ComfyUI...' -ForegroundColor Cyan
  Invoke-Native -FilePath 'git' -ArgumentList @(
    'clone',
    '--depth', '1',
    'https://github.com/Comfy-Org/ComfyUI.git',
    $comfyPath
  )
}
elseif (Test-Path -LiteralPath (Join-Path $comfyPath '.git')) {
  Write-Host '[install.ps1] Updating ComfyUI...' -ForegroundColor Cyan

  # A normal git pull refuses to overwrite locally modified ComfyUI core files.
  # Preserve tracked local edits in a named stash, update ComfyUI, and leave the
  # stash intact for manual review. Untracked/ignored custom nodes and models are
  # not stashed or removed.
  $trackedStatus = @(
    & git -C $comfyPath status --porcelain --untracked-files=no
  )
  if ($LASTEXITCODE -ne 0) {
    throw 'Unable to inspect the ComfyUI Git working tree.'
  }

  $autoStashMessage = $null
  $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
  if ($trackedStatus.Count -gt 0) {
    $autoStashMessage = "custom-wan-installer-before-update-$stamp"

    Write-Warning 'Tracked local ComfyUI changes were found. They will be preserved in Git stash before updating.'
    $trackedStatus | ForEach-Object { Write-Host ("  {0}" -f $_) -ForegroundColor Yellow }

    Invoke-Native -FilePath 'git' -ArgumentList @(
      '-C', $comfyPath,
      'stash', 'push',
      '--message', $autoStashMessage
    )
  }

  try {
    # `git pull` without an explicit upstream fails when ComfyUI is checked out
    # at a release tag (detached HEAD), which is how this installer has
    # historically provisioned some installations. Fetch and fast-forward to
    # the remote's default branch explicitly so both attached and detached
    # checkouts update deterministically.
    Invoke-Native -FilePath 'git' -ArgumentList @('-C', $comfyPath, 'fetch', 'origin')

    $remoteHead = @(
      & git -C $comfyPath symbolic-ref --quiet --short refs/remotes/origin/HEAD
    )
    if ($LASTEXITCODE -ne 0 -or $remoteHead.Count -eq 0 -or [string]::IsNullOrWhiteSpace($remoteHead[0])) {
      Invoke-Native -FilePath 'git' -ArgumentList @('-C', $comfyPath, 'remote', 'set-head', 'origin', '--auto')
      $remoteHead = @(
        & git -C $comfyPath symbolic-ref --quiet --short refs/remotes/origin/HEAD
      )
    }
    if ($LASTEXITCODE -ne 0 -or $remoteHead.Count -eq 0 -or $remoteHead[0] -notmatch '^origin/.+') {
      throw 'Unable to determine the ComfyUI origin default branch after fetching.'
    }

    $targetRef = $remoteHead[0].Trim()
    $currentBranch = (& git -C $comfyPath branch --show-current | Out-String).Trim()
    if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($currentBranch)) {
      $upstream = @(
        & git -C $comfyPath rev-parse --abbrev-ref --symbolic-full-name '@{upstream}'
      )
      if ($LASTEXITCODE -eq 0 -and $upstream.Count -gt 0 -and -not [string]::IsNullOrWhiteSpace($upstream[0])) {
        $targetRef = $upstream[0].Trim()
      }
    }

    $isDetached = [string]::IsNullOrWhiteSpace($currentBranch)
    & git -C $comfyPath merge-base --is-ancestor HEAD $targetRef
    $canFastForward = ($LASTEXITCODE -eq 0)

    if ($canFastForward) {
      Invoke-Native -FilePath 'git' -ArgumentList @('-C', $comfyPath, 'merge', '--ff-only', $targetRef)
    }
    elseif ($isDetached) {
      # Old installs may be detached at a release tag from a history that has
      # since diverged from the remote default branch. Keep the original HEAD
      # on a backup branch, move the working checkout to the new default tip,
      # and replay self-contained custom-node commits (the in-tree Pixal3D
      # installer is one such local addition). Core-history-only commits remain
      # recoverable from the backup ref and are not replayed onto modern core.
      $oldHead = (& git -C $comfyPath rev-parse HEAD | Out-String).Trim()
      if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($oldHead)) {
        throw 'Unable to record the current detached ComfyUI revision.'
      }
      $shortHead = $oldHead.Substring(0, [Math]::Min(8, $oldHead.Length))
      $backupBranch = "custom-wan-backup-comfyui-$stamp-$shortHead"
      Invoke-Native -FilePath 'git' -ArgumentList @('-C', $comfyPath, 'branch', $backupBranch, $oldHead)

      $replayCommits = @(& git -C $comfyPath rev-list --reverse HEAD --not $targetRef)
      if ($LASTEXITCODE -ne 0) {
        throw 'Unable to inspect detached ComfyUI commits before updating.'
      }
      $customNodeCommits = @()
      foreach ($commit in $replayCommits) {
        $changedPaths = @(& git -C $comfyPath diff-tree --no-commit-id --name-only -r $commit)
        if ($LASTEXITCODE -ne 0) {
          throw "Unable to inspect files in detached ComfyUI commit $commit."
        }
        if ($changedPaths.Count -gt 0 -and @($changedPaths | Where-Object { $_ -notlike 'custom_nodes/*' }).Count -eq 0) {
          $customNodeCommits += $commit
        }
      }

      Invoke-Native -FilePath 'git' -ArgumentList @('-C', $comfyPath, 'checkout', '--detach', $targetRef)
      try {
        foreach ($commit in $customNodeCommits) {
          Invoke-Native -FilePath 'git' -ArgumentList @('-C', $comfyPath, 'cherry-pick', '--no-edit', $commit)
        }
      }
      catch {
        & git -C $comfyPath cherry-pick --abort *> $null
        & git -C $comfyPath checkout --detach $backupBranch *> $null
        throw
      }

      Write-Warning "ComfyUI history had diverged. The previous checkout is preserved as branch '$backupBranch'."
      if ($customNodeCommits.Count -gt 0) {
        Write-Host ("Reapplied {0} local custom-node commit(s) on the updated ComfyUI checkout." -f $customNodeCommits.Count) -ForegroundColor Cyan
      }
    }
    else {
      throw "The current ComfyUI branch '$currentBranch' cannot fast-forward to '$targetRef'. It was left unchanged."
    }
  }
  catch {
    # If updating itself fails, restore the just-created stash when possible so
    # the working installation is returned to its previous state.
    if ($autoStashMessage) {
      Write-Warning 'ComfyUI update failed. Attempting to restore the preserved local changes...'
      & git -C $comfyPath stash pop
      if ($LASTEXITCODE -ne 0) {
        Write-Warning "Automatic stash restoration also failed. The changes remain available in git stash: $autoStashMessage"
      }
    }
    throw
  }

  if ($autoStashMessage) {
    $stashRecord = @(
      & git -C $comfyPath stash list --max-count=1 '--format=%gd %s'
    )
    if ($LASTEXITCODE -eq 0 -and $stashRecord.Count -gt 0) {
      Write-Warning ("ComfyUI was updated. Your prior core-file edits remain preserved as: {0}" -f $stashRecord[0])
      Write-Host "Review later: git -C `"$comfyPath`" stash show -p 'stash@{0}'" -ForegroundColor Yellow
      Write-Host "Restore later: git -C `"$comfyPath`" stash apply 'stash@{0}'" -ForegroundColor Yellow
    }
  }
}
else {
  Write-Host '[install.ps1] Existing non-Git ComfyUI installation detected; update skipped.' -ForegroundColor Yellow
}

# -----------------------------------------------------------------------------
# Create/reuse the virtual environment.
# -----------------------------------------------------------------------------
if ((Test-Path -LiteralPath $venvPath) -and -not $ReuseVenv) {
  Write-Host '[install.ps1] Removing the existing virtual environment...' -ForegroundColor Yellow
  Remove-VirtualEnvironment `
    -Path $venvPath `
    -AllowedParentPath $comfyPath `
    -SupervisorScriptPath $runWanPath `
    -LockedVenvAction $LockedVenvAction
}

if (-not (Test-Path -LiteralPath $venvPython)) {
  Write-Host "[install.ps1] Creating Python $PyVersion virtual environment..." -ForegroundColor Cyan
  Invoke-Native -FilePath 'py' -ArgumentList @("-$PyVersion", '-m', 'venv', $venvPath)
}

# Recheck immediately before health checks and package mutation. Editors may
# start a language server while the Git update or venv creation is in progress.
Resolve-VirtualEnvironmentLock `
  -Path $venvPath `
  -SupervisorScriptPath $runWanPath `
  -LockedVenvAction $LockedVenvAction

$venvCheckCode = 'import pathlib, sys; expected = pathlib.Path(sys.argv[1]).resolve(); actual = pathlib.Path(sys.prefix).resolve(); assert actual == expected, f"Unexpected sys.prefix: {actual} != {expected}"'
Invoke-Native -FilePath $venvPython -ArgumentList @('-c', $venvCheckCode, $venvPath)
Invoke-Native -FilePath $venvPython -ArgumentList @('-m', 'pip', '--version')

Invoke-Native -FilePath $venvPython -ArgumentList @('-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools<82', 'wheel', 'build')
$pipNetworkOptions = @('--retries', '10', '--resume-retries', '10', '--timeout', '120')

# -----------------------------------------------------------------------------
# Install PyTorch for the requested compute backend.
# -----------------------------------------------------------------------------
if ($Cuda -eq 'directml') {
  Write-Host '[install.ps1] Installing PyTorch with the DirectML backend (AMD/Intel GPUs on Windows)...' -ForegroundColor Cyan
  # torch-directml pins the torch build it supports; pip resolves matching
  # torchvision/torchaudio versions against that pin.
  Invoke-NativeWithRetry -FilePath $venvPython -ArgumentList (
    @('-m', 'pip', 'install', '--upgrade') +
    $pipNetworkOptions +
    @('torch-directml', 'torchvision', 'torchaudio')
  )
}
else {
  $torchIndex = switch ($Cuda) {
    'cu128' { 'https://download.pytorch.org/whl/cu128' }
    'cu121' { 'https://download.pytorch.org/whl/cu121' }
    'cu118' { 'https://download.pytorch.org/whl/cu118' }
    'cpu'   { 'https://download.pytorch.org/whl/cpu' }
  }

  Write-Host "[install.ps1] Installing PyTorch build: $Cuda" -ForegroundColor Cyan
  Invoke-NativeWithRetry -FilePath $venvPython -ArgumentList (
    @('-m', 'pip', 'install', '--upgrade') +
    $pipNetworkOptions +
    @('torch', 'torchvision', 'torchaudio', '--index-url', $torchIndex)
  )
}

$requirements = Join-Path $comfyPath 'requirements.txt'
if (-not (Test-Path -LiteralPath $requirements)) {
  throw "ComfyUI requirements.txt is missing: $requirements"
}
Invoke-NativeWithRetry -FilePath $venvPython -ArgumentList (@('-m', 'pip', 'install') + $pipNetworkOptions + @('-r', $requirements))

# -----------------------------------------------------------------------------
# Optional ComfyUI Manager.
# -----------------------------------------------------------------------------
if ($WithManager) {
  $customNodes = Join-Path $comfyPath 'custom_nodes'
  $managerPath = Join-Path $customNodes 'ComfyUI-Manager'
  Ensure-Directory $customNodes

  if (-not (Test-Path -LiteralPath (Join-Path $managerPath '.git'))) {
    if (Test-Path -LiteralPath $managerPath) {
      Remove-Item -LiteralPath $managerPath -Recurse -Force
    }
    Write-Host '[install.ps1] Installing ComfyUI-Manager...' -ForegroundColor Cyan
    Invoke-Native -FilePath 'git' -ArgumentList @(
      'clone',
      '--depth', '1',
      'https://github.com/Comfy-Org/ComfyUI-Manager.git',
      $managerPath
    )
  }
  else {
    Invoke-Native -FilePath 'git' -ArgumentList @('-C', $managerPath, 'pull', '--ff-only')
  }

  $managerRequirements = Join-Path $managerPath 'requirements.txt'
  if (Test-Path -LiteralPath $managerRequirements) {
    Invoke-NativeWithRetry -FilePath $venvPython -ArgumentList (@('-m', 'pip', 'install') + $pipNetworkOptions + @('-r', $managerRequirements))
  }
}

# -----------------------------------------------------------------------------
# Curated required custom nodes (config/nodes.json), each at a pinned commit.
# -----------------------------------------------------------------------------
if (-not $SkipNodes) {
  $customNodesRoot = Join-Path $comfyPath 'custom_nodes'
  Ensure-Directory $customNodesRoot

  foreach ($node in $nodeManifest.nodes) {
    $nodeName = [string] $node.name
    $nodeRepository = [string] $node.repository
    $nodeRevision = [string] $node.revision
    if ($nodeName -notmatch '^[A-Za-z0-9._-]+$') {
      throw "Unsafe node name in manifest: $nodeName"
    }
    if ($nodeRepository -notmatch '^https://github\.com/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+\.git$') {
      throw "Unsupported node repository in manifest: $nodeRepository"
    }
    if ($nodeRevision -notmatch '^[0-9a-f]{7,40}$') {
      throw "Node revision must be a pinned commit: $nodeRevision"
    }

    $nodePath = Join-Path $customNodesRoot $nodeName
    if (Test-Path -LiteralPath (Join-Path $nodePath '.git')) {
      Write-Host "[nodes] $nodeName present - fetching pinned revision." -ForegroundColor Cyan
      & git -C $nodePath fetch origin *> $null
    }
    else {
      Write-Host "[nodes] Installing $nodeName..." -ForegroundColor Cyan
      Invoke-Native -FilePath 'git' -ArgumentList @('clone', $nodeRepository, $nodePath)
    }
    Invoke-Native -FilePath 'git' -ArgumentList @('-C', $nodePath, 'checkout', '--detach', $nodeRevision)

    $nodeRequirements = Join-Path $nodePath 'requirements.txt'
    if (Test-Path -LiteralPath $nodeRequirements -PathType Leaf) {
      Invoke-NativeWithRetry -FilePath $venvPython -ArgumentList (@('-m', 'pip', 'install') + $pipNetworkOptions + @('-r', $nodeRequirements))
    }
  }
}

# -----------------------------------------------------------------------------
# Environment patches for ONNX, face-swap, and audio custom nodes.
# Use the venv interpreter explicitly; do not depend on activate.ps1 or bare pip.
# -----------------------------------------------------------------------------
$env:GIT_CLONE_PROTECTION_ACTIVE = 'false'

$corePackages = @(
  'protobuf>=4.25,<5',
  'onnx',
  'huggingface_hub',
  'regex',
  'librosa',
  'ffmpeg-python',
  'soundfile'
)

if ($Cuda -eq 'cpu') {
  $corePackages += 'onnxruntime'
}
elseif ($Cuda -eq 'directml') {
  $corePackages += 'onnxruntime-directml'
}
else {
  $corePackages += 'onnxruntime-gpu'
}

Invoke-NativeWithRetry -FilePath $venvPython -ArgumentList (@('-m', 'pip', 'install', '--upgrade') + $pipNetworkOptions + $corePackages)

# Older revisions installed the unrelated PyPI package named `dac`, which
# downgrades Click/Typer and conflicts with current Hugging Face Hub. Remove
# only that known legacy package before installing the intended audio codec.
& $venvPython -m pip show dac *> $null
if ($LASTEXITCODE -eq 0) {
  Write-Warning "Removing legacy package 'dac'; the intended package is 'descript-audio-codec'."
  Invoke-Native -FilePath $venvPython -ArgumentList @('-m', 'pip', 'uninstall', '-y', 'dac')
}

$optionalPackages = @(
  'insightface',
  'audiotoolbox',
  'descript-audio-codec>=1.0.0',
  'git+https://github.com/descriptinc/audiotools@348ebf2034ce24e2a91a553e3171cb00c0c71678'
)

foreach ($package in $optionalPackages) {
  try {
    Invoke-NativeWithRetry -FilePath $venvPython -ArgumentList (@('-m', 'pip', 'install', '--upgrade') + $pipNetworkOptions + @($package))
  }
  catch {
    Write-Warning "Optional package failed and was skipped: $package`n$($_.Exception.Message)"
  }
}

# Optional audio packages have historically imposed older CLI dependency
# bounds. Restore the core-compatible versions, then require a clean graph.
Invoke-NativeWithRetry -FilePath $venvPython -ArgumentList (
  @('-m', 'pip', 'install', '--upgrade') +
  $pipNetworkOptions +
  @('click>=8.4.2,<9', 'typer>=0.27,<1')
)
Invoke-Native -FilePath $venvPython -ArgumentList @('-m', 'pip', 'check')

# -----------------------------------------------------------------------------
# Download the selected model files across every manifest family (Wan, LTX,
# text encoders). `all` includes 5B TI2V, 14B T2V, 14B I2V, and LTX-Video.
# The -ModelRepository/-ModelRevision overrides apply to the Wan family only.
# -----------------------------------------------------------------------------
$selectedModelGroups = if ($Models -eq 'all') { @('5b', '14b', 'i2v', 'ltx', 'ltx2') } else { @($Models) }
$allowedDestinations = @('checkpoints', 'diffusion_models', 'text_encoders', 'vae', 'loras', 'latent_upscale_models')
$modelFamilies = @($modelManifest.PSObject.Properties | Where-Object Name -ne 'schema_version')
foreach ($modelFamily in $modelFamilies) {
  $familyRepository = [string] $modelFamily.Value.repository
  $familyRevision = [string] $modelFamily.Value.revision
  if ($modelFamily.Name -eq 'wan') {
    $familyRepository = $ModelRepository
    $familyRevision = $ModelRevision
  }
  if ($familyRepository -notmatch '^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$') {
    throw "Model family '$($modelFamily.Name)' repository must use the owner/repository form."
  }
  if ($familyRevision -notmatch '^[A-Za-z0-9._/-]+$' -or $familyRevision.Contains('..')) {
    throw "Model family '$($modelFamily.Name)' revision must be a branch, tag, or commit without traversal segments."
  }

  foreach ($artifact in $modelFamily.Value.artifacts) {
    $artifactGroups = @($artifact.groups | ForEach-Object { [string] $_ })
    if (-not @($artifactGroups | Where-Object { $_ -in $selectedModelGroups })) {
      continue
    }
    $destinationGroup = [string] $artifact.destination
    if ($destinationGroup -notin $allowedDestinations) {
      throw "Unsupported model destination in manifest: $destinationGroup"
    }
    $relativePath = [string] $artifact.path
    if ($relativePath -notmatch '^[A-Za-z0-9][A-Za-z0-9._/-]*$' -or $relativePath.Contains('..')) {
      throw "Unsafe model path in manifest: $relativePath"
    }
    $artifactSha256 = [string] $artifact.sha256
    if ($artifactSha256 -notmatch '^[0-9a-f]{64}$') {
      throw "Model artifact is missing a valid sha256 in the manifest: $relativePath"
    }
    $destinationDirectory = Join-Path $comfyPath "models\$destinationGroup"
    Ensure-Directory $destinationDirectory
    Get-HuggingFaceFile `
      -Repository $familyRepository `
      -Revision $familyRevision `
      -RelativePath $relativePath `
      -Destination (Join-Path $destinationDirectory (Split-Path -Leaf $relativePath)) `
      -Sha256 $artifactSha256
  }
}

# -----------------------------------------------------------------------------
# Verify Python/PyTorch. CUDA verification is reported clearly but does not hide
# the installed torch build information.
# -----------------------------------------------------------------------------
$verifyCode = if ($Cuda -eq 'directml') {
@'
import sys
import torch
print('Python:', sys.version.split()[0])
print('Torch:', torch.__version__)
try:
    import torch_directml
    print('DirectML available:', torch_directml.is_available())
    if torch_directml.is_available():
        print('GPU:', torch_directml.device_name(0))
except Exception as exc:
    print('DirectML unavailable:', exc)
'@
}
else {
@'
import sys
import torch
print('Python:', sys.version.split()[0])
print('Torch:', torch.__version__)
print('Torch CUDA build:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
'@
}

Invoke-Native -FilePath $venvPython -ArgumentList @('-c', $verifyCode)

# Repair optional custom nodes after their repositories and dependencies are
# present. The launcher repeats this idempotent preflight for reused installs.
$nodeRepair = Join-Path $PSScriptRoot 'patches\repair_custom_nodes.py'
if (Test-Path -LiteralPath $nodeRepair -PathType Leaf) {
  Invoke-Native -FilePath $venvPython -ArgumentList @(
    $nodeRepair,
    '--comfyui', $comfyPath,
    '--install-dependencies'
  )
}

Write-Host ''
Write-Host '[install.ps1] Installation completed.' -ForegroundColor Green
Write-Host ("Root:       {0}" -f $BasePath)
Write-Host ("ComfyUI:    {0}" -f $comfyPath)
Write-Host ("Venv:       {0}" -f $venvPath)
Write-Host ("Models:     {0}" -f $Models)
Write-Host ("Port:       {0}" -f $Port)
Write-Host ''

# -----------------------------------------------------------------------------
# Start only after installation is complete. Global CLI flags are placed before
# the positional `start` command to match the current argparse usage.
# -----------------------------------------------------------------------------
$device = switch ($Cuda) {
  'cpu'      { 'cpu' }
  'directml' { 'directml' }
  default    { 'gpu' }
}

if ($Start) {
  if (Test-Path -LiteralPath $cliPath) {
    # Run the launcher inside the same venv where CUDA-enabled PyTorch was installed.
    # Using the global `py` launcher can load a CPU-only torch installation.
    $startArgs = @(
      $cliPath,
      '--path', $BasePath,
      '--port', "$Port",
      '--device', $device
    )
    if ($ListenAll) {
      $startArgs += '--listen-all'
    }
    $startArgs += 'start'

    Invoke-Native -FilePath $venvPython -ArgumentList $startArgs
  }
  else {
    $mainArgs = @($mainPy, '--port', "$Port")
    if ($ListenAll) {
      $mainArgs += @('--listen', '0.0.0.0')
    }
    if ($Cuda -eq 'cpu') {
      $mainArgs += '--cpu'
    }
    elseif ($Cuda -eq 'directml') {
      $mainArgs += '--directml'
    }

    Invoke-Native -FilePath $venvPython -ArgumentList $mainArgs
  }
}
else {
  Write-Host 'Start later with:' -ForegroundColor Cyan
  Write-Host ("  & `"{0}`" `"{1}`" start --path `"{2}`" --port {3} --device {4}" -f $venvPython, $cliPath, $BasePath, $Port, $device)
}
