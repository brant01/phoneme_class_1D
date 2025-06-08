# Upgrade Notes for phoneme_class_1D

## Upgrade Date: 2025-06-08
## Upgrade Path: v1.0 → 2.1.0

This project has been upgraded to the latest CLAUDE template version.

## IMPORTANT: Manual Migration Required

Your previous configuration has been backed up:
- claude_config → claude_config_old (if upgrading from 2.x)
- CLAUDE.md → CLAUDE_old.md (if upgrading from 1.x)
- No backup needed (if creating from scratch)

Please use Claude Code to complete the upgrade:

## Quick Upgrade Steps:

1. Open this project in Claude Code
2. Ask: "Please help me upgrade my CLAUDE configuration from v1.0 to 2.1.0"
3. Claude will:
   - Read your backed up configuration
   - Migrate content to the new structure
   - Preserve all project-specific information
4. Once complete, delete the backup files and this UPGRADE_NOTES.md

## What's New in 2.1.0:

- Cross-platform .env support with @REMOTE_DRIVE@ placeholder
- Enhanced commit guidelines (NO Claude/AI mentions)
- Improved data handling documentation
- Platform-specific path resolution

## Files to Review and Update:
- [ ] claude_config/project/details.md - Project description, goals
- [ ] claude_config/project/architecture.md - Technical design
- [ ] claude_config/project/log.md - Session history
- [ ] claude_config/guidelines/* - Project guidelines
- [ ] .env files - Add @REMOTE_DRIVE@ for cross-platform paths

## Backup Locations:
- Previous config: CLAUDE_old.md
- Delete backups after successful migration
