# Push credit_agent v1.0.0 to GitHub
# This script overwrites remote main with local v1.0.0 history.

$ErrorActionPreference = "Stop"
$env:GIT_TERMINAL_PROMPT = "1"

Set-Location -LiteralPath "D:\vibe coding\2026-09-08-16-37-15\credit_agent"

Write-Host "============================================================"
Write-Host "Current state"
Write-Host "============================================================"
git status --short
Write-Host ""
Write-Host "Local main:"
git log --oneline -3
Write-Host ""
Write-Host "Remote main:"
git ls-remote --heads origin
Write-Host ""

Write-Host "============================================================"
Write-Host "Step 1/3: Push backup branch"
Write-Host "============================================================"
try {
    git push origin backup-connector-test
} catch {
    Write-Host "[WARN] Backup branch push failed or already exists, continuing..."
}
Write-Host ""

Write-Host "============================================================"
Write-Host "Step 2/3: Force push local v1.0.0 to main"
Write-Host "============================================================"
Write-Host "When prompted, enter:"
Write-Host "  Username: xzxgsjgs"
Write-Host "  Password: your GitHub Personal Access Token (PAT)"
Write-Host ""
Write-Host "If you do not have a PAT yet, generate one at:"
Write-Host "  https://github.com/settings/tokens?type=beta"
Write-Host ""
git push --force-with-lease origin main
Write-Host ""

Write-Host "============================================================"
Write-Host "Step 3/3: Push v1.0.0 tag"
Write-Host "============================================================"
try {
    git push origin v1.0.0
} catch {
    Write-Host "[WARN] Tag push failed, possibly already exists."
}
Write-Host ""

Write-Host "============================================================"
Write-Host "Done!"
Write-Host "============================================================"
Write-Host "Local log:"
git log --oneline -3
Write-Host ""
Write-Host "Tags:"
git tag -l
Write-Host ""
Write-Host "Verify at: https://github.com/xzxgsjgs/Enterprise-Credit-Scoring-Agent"
Read-Host -Prompt "Press Enter to exit"