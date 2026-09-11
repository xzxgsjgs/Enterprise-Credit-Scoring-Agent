@echo off
REM ============================================================
REM Push credit_agent v1.0.0 to GitHub
REM This script overwrites remote main with local v1.0.0 history.
REM ============================================================

chcp 437 >nul
set GIT_TERMINAL_PROMPT=1

cd /d "D:\vibe coding\2026-09-08-16-37-15\credit_agent"

echo ============================================================
echo Current state
echo ============================================================
git status --short
echo.
echo Local main:
git log --oneline -3
echo.
echo Remote main:
git ls-remote --heads origin
echo.

echo ============================================================
echo Step 1/3: Push backup branch
echo ============================================================
git push origin backup-connector-test
if errorlevel 1 (
    echo [WARN] Backup branch push failed or already exists, continuing...
)
echo.

echo ============================================================
echo Step 2/3: Force push local v1.0.0 to main
echo ============================================================
echo When prompted, enter:
echo   Username: xzxgsjgs
echo   Password: your GitHub Personal Access Token (PAT)
echo.
echo If you do not have a PAT yet, generate one at:
echo   https://github.com/settings/tokens?type=beta
echo.
git push --force-with-lease origin main
if errorlevel 1 (
    echo [ERROR] Main push failed. Check your credentials or network.
    echo If you need help creating a PAT, see README.md or ask me.
    pause
    exit /b 1
)
echo.

echo ============================================================
echo Step 3/3: Push v1.0.0 tag
echo ============================================================
git push origin v1.0.0
if errorlevel 1 (
    echo [WARN] Tag push failed, possibly already exists.
)
echo.

echo ============================================================
echo Done!
echo ============================================================
echo Local log:
git log --oneline -3
echo.
echo Tags:
git tag -l
echo.
echo Verify at:
echo   https://github.com/xzxgsjgs/Enterprise-Credit-Scoring-Agent
pause