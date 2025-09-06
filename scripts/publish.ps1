Param(
  [string]$RemoteUrl = "https://github.com/kashif2112/kairox-ai-web-search-agent.git",
  [string]$Branch = "main",
  [string]$UserName = $null,
  [string]$UserEmail = $null
)

$ErrorActionPreference = "Stop"

function Require-Cmd($cmd) {
  if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
    Write-Error "'$cmd' is not installed or not on PATH. Please install it and re-run."
  }
}

Require-Cmd git

# Configure identity if provided
if ($UserName) { git config user.name $UserName }
if ($UserEmail) { git config user.email $UserEmail }

# Ensure .env and UI history are ignored
".env`n.kairox_ui_history.json`n" | Out-File -FilePath .gitignore -Append -Encoding utf8 -ErrorAction SilentlyContinue

# Init repo if needed
if (-not (Test-Path .git)) {
  git init
}

# Create default branch
git checkout -B $Branch | Out-Null

# Add remote if missing or update
$existingRemote = git remote get-url origin 2>$null
if (-not $existingRemote) {
  git remote add origin $RemoteUrl
} elseif ($existingRemote -ne $RemoteUrl) {
  git remote set-url origin $RemoteUrl
}

# Stage and commit
git add .

# Skip commit if nothing changed
try {
  $status = git status --porcelain
  if (-not [string]::IsNullOrWhiteSpace($status)) {
    git commit -m "Initial public release"
  } else {
    Write-Host "No changes to commit."
  }
} catch {
  Write-Host "Nothing to commit or already committed."
}

# Push
git push -u origin $Branch
Write-Host "Pushed to $RemoteUrl ($Branch)."

