<!-- CLAUDE.md Template Version: 1.0 -->
# Claude.md - Project Configuration v1.0

## NAVIGATION GUIDE FOR CLAUDE

- **First time**: Read entire file
- **Session start**: Read [PROJECT DETAILS] + [SESSION START CHECKLIST]
- **Before coding**: Read [BEFORE WRITING CODE CHECKLIST]
- **Before commit**: Read [COMMIT CHECKLIST] + [GIT REMINDERS]
- **Session end**: Update [PROJECT LOG] + read [SESSION END CHECKLIST]
- **If unsure**: Check [CORE PRINCIPLES] or [WHEN TO ASK]

---

## [PROJECT DETAILS] - EDITABLE BY CLAUDE

<!-- Claude updates this section based on project needs -->

Project Name: phoneme-class-1D (Phoneme Classification 1D)  
Last Updated: 2025-06-08  
Description: A phoneme classification system using 1D convolutional neural networks and contrastive learning for audio analysis in medical/speech research

### Project Context

- **Primary Goal**: Develop robust phoneme classification using contrastive learning on audio signals
- **Project Type**: Machine Learning Research Tool / Audio Analysis Library
- **Data Types**: WAV audio files, spectrograms, MFCC features, wavelet transforms
- **Key Libraries**: PyTorch, TorchAudio, scikit-learn, pydantic, pywavelets, soundfile
- **Performance Requirements**: GPU optimization, memory-efficient batch processing, real-time augmentation
- **Domain Context**: Medical/Speech Research - Audio signal processing and phoneme recognition

### Core Functionality

1. **Audio Data Pipeline**
   - Parse WAV files from structured CV/VCV phoneme datasets
   - Extract phoneme, speaker, and gender metadata
   - Support variable-length audio processing

2. **Feature Extraction**
   - MFCC (Mel-frequency cepstral coefficients)
   - Log-Mel spectrograms
   - Wavelet transforms
   - Configurable transform pipelines

3. **Data Augmentation**
   - Time and frequency masking
   - Additive noise injection
   - Composable augmentation chains

4. **Contrastive Learning**
   - Multi-view batch sampling
   - Contrastive loss optimization
   - Latent space representation learning

5. **Experiment Framework**
   - Cross-validation support
   - Configurable hyperparameters via YAML
   - Comprehensive logging and diagnostics

### Design Decisions

1. **Contrastive Learning Approach**: Chosen for robust phoneme representations without extensive labeled data
2. **Modular Transform Pipeline**: Allows flexible feature engineering and augmentation strategies
3. **Experiment-Based Architecture**: Facilitates reproducible research with parameter tracking
4. **Multi-View Sampling**: Enables effective contrastive learning by creating positive/negative pairs
5. **Memory-Aware Processing**: Handles varying audio lengths efficiently with proper batching

### Architecture

```
phoneme_class_1D/
├── src/
│   ├── data/           # Dataset handling and feature extraction
│   │   ├── dataset.py      # PhonemeDataset class
│   │   ├── parser.py       # Data parsing utilities
│   │   └── features.py     # Feature extraction logic
│   ├── experiment/     # Experiment framework
│   │   ├── experiment.py   # Main experiment runner
│   │   └── exp_params.py   # Parameter definitions
│   ├── transforms/     # Audio transforms
│   │   ├── build_transforms.py  # Transform factory
│   │   ├── compose.py          # Transform composition
│   │   ├── log_mel.py          # Log-Mel spectrogram
│   │   ├── mfcc.py             # MFCC features
│   │   ├── wavelet.py          # Wavelet transforms
│   │   ├── masking.py          # Time/freq masking
│   │   └── noise.py            # Noise augmentation
│   └── utils/          # Utilities
│       ├── device.py           # GPU/CPU device management
│       ├── logging.py          # Logging configuration
│       ├── samplers.py         # Multi-view sampling
│       └── evaluate_latent_classification.py  # Evaluation
├── tests/              # pytest unit tests
├── data/               # Raw audio data
│   └── raw/
│       └── New Stimuli 9-8-2024/
│           ├── CV/     # Consonant-Vowel phonemes
│           └── VCV/    # Vowel-Consonant-Vowel phonemes
├── runs/               # Experiment outputs and logs
├── configs/            # YAML configuration files
└── main.py            # Entry point
```

---

## [CORE PRINCIPLES] - PROTECTED

### Critical Mindset

1. **THINK FIRST** - 80% planning, 20% coding
2. **CHALLENGE EVERYTHING** - If you see a better way, say so
3. **VERIFY ASSUMPTIONS** - Question my suggestions
4. **BE CRITICAL** - Present honest tradeoffs
5. **TEACH CLEARLY** - I'm self-taught, explain thoroughly

### Communication Rules

- NO emojis, icons, or symbols anywhere in the project
- NO "co-authored by" or "with Claude" in commits or comments
- Professional, direct communication
- Challenge suboptimal suggestions immediately

### Technical Standards

- **Language**: Python 3.12+ with type hints where appropriate
- **Dependencies**: UV package manager, requirements in pyproject.toml
- **Error Handling**: Comprehensive try-except blocks with informative error messages, proper logging
- **Testing**: pytest-based unit testing, fixtures for audio data mocking
- **Structure**: Modular architecture with clear separation: data/, experiment/, transforms/, utils/

### Security Requirements (NON-NEGOTIABLE)

1. Never store credentials in code
2. Use environment variables for secrets
3. Validate all user input
4. Handle file paths safely
5. Follow security best practices for the domain
6. No PHI/PII in code, logs, or commits (medical research compliance)
7. Ensure proper data anonymization practices
8. Secure handling of audio file paths and metadata

---

## [WHEN TO ASK] - PROTECTED

### Always Ask Before

- Adding new dependencies
- Changing core architecture
- Modifying data structures
- Adding complex features
- Changing default behaviors

### How to Present Options

"I see X approaches:
- Option A: [approach] - Pros: [list] Cons: [list]
- Option B: [approach] - Pros: [list] Cons: [list]
Which aligns with your needs?"

---

## [SESSION START CHECKLIST]

- [ ] Read PROJECT DETAILS section
- [ ] Check recent changes in PROJECT LOG
- [ ] Review any pending TODOs
- [ ] Verify development environment

---

## [BEFORE WRITING CODE CHECKLIST]

- [ ] Is this the simplest solution? (KISS)
- [ ] Are we building only what's needed? (YAGNI)
- [ ] Does this follow project conventions?
- [ ] Error handling comprehensive?
- [ ] Have I presented alternatives?
- [ ] Audio data handling follows medical research protocols?
- [ ] Memory usage considered for batch processing?
- [ ] GPU/CPU device handling appropriate?

---

## [COMMIT CHECKLIST]

- [ ] Code follows project style
- [ ] Functions/methods documented
- [ ] Error messages are helpful
- [ ] No hardcoded values
- [ ] Tests pass (if applicable)
- [ ] Updated PROJECT LOG
- [ ] Commit message clear (no co-author references)

---

## [SESSION END CHECKLIST]

- [ ] Update PROJECT LOG with decisions
- [ ] Note any unresolved questions
- [ ] Document next steps
- [ ] Add TODOs for incomplete work

---

## [GIT REMINDERS]

### Before EVERY Push

1. **Test core functions**: `pytest tests/`
2. **Run linter**: `ruff check src/` (if available)
3. **Push**: `git push origin main`

### Commit Format

```
type: brief description

Detailed explanation if needed
```

NO "Co-authored-by" tags ever.

### Required .gitignore Entries

```
# Environment
.env
*.local

# Dependencies
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/

# OS files
.DS_Store
Thumbs.db
```

---

## [TODO/FIXME GUIDELINES]

When adding TODOs that should be tracked by pm tool:

```python
# TODO: Add error handling for API calls
# TODO Add validation for user input
# FIXME: This breaks when input is None
# TODO - implement caching mechanism
```

Requirements:
- Must be uppercase: `TODO` or `FIXME`
- Must be in .py files within src/ directory
- Common formats: `# TODO:`, `# TODO`, `# FIXME:`, inline comments
- Also works in docstrings: `"""TODO: implement this method"""`

---

## [PROJECT LOG] - EDITABLE BY CLAUDE

### Template Version: v1.0 (2025-06-03)

### Project Sessions

#### 2025-06-08 - Session 1: Claude.md Update and Project Configuration

**Decisions**:
- Migrated from old Claude configuration to new template v1.0
- Integrated project-specific standards from claude_config directory
- Established medical research data handling protocols

**Architecture Choices**:
- Maintained contrastive learning approach for phoneme classification
- Preserved modular transform pipeline architecture
- Kept experiment-based framework for reproducible research

**Next Steps**:
- [ ] Review and test updated configuration
- [ ] Ensure all team members aware of new standards
- [ ] Validate data handling protocols meet requirements

#### Active Development

<!-- Current work - clean up when complete -->
- Working on: Classification performance optimization and memory management
- Questions: 
  - Best practices for handling variable-length audio in batches?
  - Optimal contrastive learning parameters for phoneme data?

---

## [QUICK REFERENCE]

### Common Commands

```bash
# Development
python main.py                    # Run experiment with default config
python main.py --config configs/base.yaml  # Run with specific config

# Data Processing  
python -m src.data.parser data/raw  # Parse audio dataset

# Testing
pytest tests/                     # Run all tests
pytest tests/test_transforms.py   # Test specific module

# Environment
uv sync                          # Install dependencies
uv pip install -e .              # Install package in dev mode
```

### Development Philosophy

**KISS** > YAGNI > DRY  
Simple > Clever  
Explicit > Implicit  
Working > Perfect