[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$installerPath = Join-Path $repositoryRoot 'install.ps1'

$installer = Get-Command $installerPath
$portParameter = $installer.Parameters['Port']
$rangeAttributes = @(
  $portParameter.Attributes |
    Where-Object { $_ -is [Management.Automation.ValidateRangeAttribute] }
)

if ($rangeAttributes.Count -ne 1) {
  throw 'install.ps1 Port must declare exactly one ValidateRange attribute.'
}

$range = $rangeAttributes[0]
if ($range.MinRange -ne 1 -or $range.MaxRange -ne 65535) {
  throw "install.ps1 Port range must be inclusive 1..65535; found $($range.MinRange)..$($range.MaxRange)."
}

foreach ($invalidPort in @(0, 65536)) {
  $rejected = $false
  try {
    & $installerPath -Port $invalidPort -ErrorAction Stop
  }
  catch {
    $rejected = $_.FullyQualifiedErrorId -like 'ParameterArgumentValidationError*'
  }

  if (-not $rejected) {
    throw "install.ps1 accepted invalid port $invalidPort."
  }
}

Write-Output 'PASS: install.ps1 enforces the inclusive port range 1..65535'

Import-Module (Join-Path $PSScriptRoot 'TestPython.psm1') -Force
$python = Resolve-TestPython

$pythonArguments = if ($python.Count -gt 1) {
  @($python[1..($python.Count - 1)])
}
else {
  @()
}
$pythonCommand = $python[0]
$pythonArguments = @($pythonArguments) + @(
  '-m',
  'unittest',
  'discover',
  '-s', 'tests',
  '-p', 'test_*.py'
)
& $pythonCommand @pythonArguments
if ($LASTEXITCODE -ne 0) {
  throw "Python port validation tests failed with exit code $LASTEXITCODE."
}

Write-Output 'PASS: Python launchers enforce the inclusive port range 1..65535'
