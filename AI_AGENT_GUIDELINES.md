# PCLN - AI Agent Guidelines

**Purpose**: Ensure all AI agents follow consistent folder structure and documentation practices for this project.

---

## 📁 Folder Structure Rules

### Main Folder (Root)
**Only user-facing files go here:**
- `README.md` - Quick start guide, features overview, project status
- `plan.md` - Original project roadmap and vision
- `FEATURES_GUIDE.md` → **MOVED TO AI_NOTES** (was here, now in AI_NOTES only)
- All `.py` source code (`src/`, `scripts/`)
- Config files (`requirements.txt`, etc.)

### AI_NOTES Folder (Development Documentation)
**All internal development/analysis docs go here:**
- `NEXT_TASKS.md` - Tasks for next session (start here!)
- `PHASE4_STATUS.md` - Status of Phase 4 implementation
- `BENCHMARKING_PROGRESS.md` - Current benchmark status
- `FEATURES_IMPLEMENTED.md` - Technical implementation details
- `FEATURES_GUIDE.md` - User guide for using new features
- `PROGRESS_STEP3.md` - Historical progress (Sparse MoE)
- `TRAINING_COMPLETE.md` - Historical training results
- `REAL_DATA_GUIDE.md` - Historical data loading notes
- `PLAN.md` - Historical original plan

---

## 📝 Documentation Rules

### When Creating New Markdown Files

**✓ DO:**
- Put development docs in `AI_NOTES/`
- Use clear filenames: `PHASE5_PLAN.md`, `BENCHMARK_RESULTS.md`, etc.
- Keep files focused (one topic per file, max ~500 lines)
- Link between related docs with relative paths: `[See next tasks](NEXT_TASKS.md)`
- Start with one-line summary at the top
- Include timestamps for time-sensitive content (like benchmarking)

**✗ DON'T:**
- Create markdown files in root folder (except README.md + plan.md)
- Create huge monolithic docs (>1000 lines)
- Duplicate information across multiple files
- Forget to update README.md with major changes
- Leave large markdown files with chaotic structure

### File Size Guidelines
- **Small** (< 200 lines): Code-focused docs
- **Medium** (200-500 lines): Implementation summaries, status updates
- **Large** (500+ lines): ONLY if absolutely necessary, then split into smaller files

### Example Good Structure
```
AI_NOTES/
├── NEXT_TASKS.md              (50 lines) - What to do next
├── PHASE4_STATUS.md           (200 lines) - Phase 4 summary
├── FEATURES_GUIDE.md          (150 lines) - How to use features
├── BENCHMARKING_PROGRESS.md   (100 lines) - Current benchmark status
└── TECHNICAL/                 (Optional subdirectory)
    ├── DYNAMIC_NEURONS.md     (150 lines) - Implementation details
    └── CHAR_TOKENIZATION.md   (100 lines) - Character encoding
```

---

## 🎯 When Starting a Session

**Always start here:**

Activate the project's Python virtual environment (`venv`) before running any project scripts. On Windows PowerShell: `.\venv\Scripts\Activate.ps1`; on Windows CMD: `.\venv\Scripts\activate.bat`; on Unix/macOS: `source venv/bin/activate`.

1. Read `AI_NOTES/NEXT_TASKS.md` - knows exactly what to do
2. Check current progress in status files
3. Review any error logs if continuing broken work
4. Update NEXT_TASKS.md when you COMPLETE items

**Before ending a session:**
1. Update status files with completed work
2. Create/update `NEXT_TASKS.md` with next steps
3. Document any blockers or issues
4. Keep README.md in sync with project status

---

## 🔍 Current Project Layout (CORRECT)

```
PCLN/
├── README.md                              # User-facing
├── plan.md                                # Original roadmap
├── requirements.txt
├── src/
│   ├── model/
│   │   ├── pcn_model.py
│   │   ├── dynamic_neurons.py            # Phase 4 new
│   │   └── ...
│   └── data/
│       └── __init__.py
├── scripts/
│   ├── train_full.py
│   ├── run_benchmarks.py                  # Phase 4 benchmarking
│   ├── analyze_benchmarks.py              # Phase 4 analysis
│   └── ...
├── results/
│   ├── exp1_baseline/
│   ├── exp2_char_level/
│   ├── exp3_dynamic_neurons/
│   ├── exp4_all_features/
│   └── benchmark_log_*.txt
│
└── AI_NOTES/                              # ← ALL DEVELOPMENT DOCS HERE
    ├── NEXT_TASKS.md                      # ← START HERE!
    ├── PHASE4_STATUS.md
    ├── BENCHMARKING_PROGRESS.md
    ├── FEATURES_GUIDE.md
    ├── FEATURES_IMPLEMENTED.md
    ├── PROGRESS_REPORT.md
    ├── PROGRESS_STEP3.md
    ├── TRAINING_COMPLETE.md
    ├── REAL_DATA_GUIDE.md
    └── PLAN.md
```

---

## 📋 Doc Update Checklist

When making significant changes:

- [ ] Update `AI_NOTES/PHASE4_STATUS.md` or status file for current phase
- [ ] Update `AI_NOTES/NEXT_TASKS.md` with new tasks
- [ ] Update `README.md` if user-visible changes
- [ ] Archive old status docs (keep as historical reference)
- [ ] Remove redundant/outdated docs
- [ ] Ensure all docs are in AI_NOTES (except README.md + plan.md)

---

## 🤖 For AI Agents

Use this checklist when working on PCLN:

```python
# At session START:
status_files = [
    "AI_NOTES/NEXT_TASKS.md",        # What to do
    "AI_NOTES/PHASE4_STATUS.md",      # Current phase status
    "AI_NOTES/BENCHMARKING_PROGRESS.md"  # If running experiments
]
# Read these first!

# Before CREATING new markdown:
if new_doc.importance == "development":
    new_doc.location = "AI_NOTES/"
elif new_doc.is_user_facing:
    new_doc.location = "root"  # Only README.md
else:
    new_doc.location = "AI_NOTES/"

# Before ENDING session:
update("AI_NOTES/NEXT_TASKS.md", completed_items, new_tasks)
update("README.md", user_visible_changes)
archive_old_docs()  # Keep historical, but not cluttering
```

---

## ❌ Anti-Patterns to Avoid

1. **✗ Markdown Chaos**: Multiple large files with overlapping content
   - ✓ Solution: Consolidate, move to AI_NOTES, split if >500 lines

2. **✗ Root Folder Clutter**: Development docs in main folder
   - ✓ Solution: Keep only README.md and plan.md in root

3. **✗ No Status Tracking**: Can't tell what's done vs. what's next
   - ✓ Solution: Maintain clear NEXT_TASKS.md and status files

4. **✗ Outdated Docs**: Old docs never updated, leave confusion
   - ✓ Solution: Archive historical docs, keep current status files

5. **✗ No Clear Path**: Read 10 docs to understand current state
   - ✓ Solution: NEXT_TASKS.md is the entry point, everything else supports it

---

## 📍 Quick Reference

**Starting new work?**
→ Read `AI_NOTES/NEXT_TASKS.md`

**Want to understand Phase 4?**
→ Read `AI_NOTES/PHASE4_STATUS.md`

**How to use new features?**
→ Read `AI_NOTES/FEATURES_GUIDE.md`

**Looking for technical implementation?**
→ Read `AI_NOTES/FEATURES_IMPLEMENTED.md`

**What's the original vision?**
→ Read `plan.md`

---

**Last Updated**: February 23, 2026  
**Enforced By**: This guidelines file (check before creating any markdown!)
