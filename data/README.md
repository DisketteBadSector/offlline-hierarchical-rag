
# مستندات پوشه `data/`

این پوشه شامل تمام داده‌های ورودی و ثابت (fixture) پروژه است که یا مستقیماً توسط کد بارگذاری می‌شوند یا به‌عنوان نمونه برای تست و توسعه استفاده می‌شوند.

---

## ساختار کلی

```
data/
├── fixtures/                  # رجیستری‌های ساختاریافته (YAML)
│   ├── layers.yaml
│   ├── models.yaml
│   └── nodes.yaml
│
├── prompts/                   # ۱۲ قالب پرامپت نسخه‌بندی‌شده (YAML)
│   ├── qa.yaml
│   ├── technical_summary.yaml
│   ├── managerial_summary.yaml
│   ├── issue_decomposition.yaml
│   ├── sub_issue_comparison.yaml
│   ├── executor_analysis.yaml
│   ├── research_gap.yaml
│   ├── knowledge_transfer.yaml
│   ├── rfp_generation.yaml
│   ├── proposal_evaluation.yaml
│   ├── feedback_generation.yaml
│   └── judge.yaml
│
└── sample-documents/          # اسناد نمونه برای تست استخراج
    ├── doc/
    ├── docx/
    └── pdf/
```

---

## ۱. `fixtures/` — رجیستری‌های ساختاریافته

فایل‌های این پوشه **منبع حقیقت (single source of truth)** برای پیکربندی دامنه‌اند و توسط `FixtureConfigLoader` در `src/offline_rag/infrastructure/config/__init__.py` بارگذاری می‌شوند.

### ۱.۱. `layers.yaml`

**هدف:** تعریف قواعد انتقال دانش بین لایه‌های سازمانی (target layer rules).

**ساختار کلیدی:**

```yaml
version: "0.1.2-phase1"
last_updated: "2026-10-29"
maxLengthUnit: "characters"

layer_transfer_rules:
  - targetLayerIndex: 0
    usageScope: "upward_transfer"        # یا intra_layer_or_audit
    detailLevel: "executive_high"
    requiredFields:
      - "key_achievements"
      - "critical_risks"
      - "unresolved_issues"
      - "strategic_recommendations"
    maxLength: 2000
    minJudgeScore: 18
  # ...
```

**فیلدها:**

| فیلد | توضیح |
|------|-------|
| `targetLayerIndex` | لایه‌ی مقصد (۰ تا ۳). لایه ۴ منبع دانش است و قاعده ندارد |
| `usageScope` | `upward_transfer` (انتقال به بالا) یا `intra_layer_or_audit` (درون‌لایه/ممیزی) |
| `detailLevel` | `executive_high` / `managerial` / `technical_summary` / `technical_detailed` |
| `requiredFields` | فیلدهای اجباری که بسته‌ی انتقال باید داشته باشد |
| `maxLength` | حداکثر طول محتوا (بر حسب کاراکتر) |
| `minJudgeScore` | حداقل امتیاز داور برای پذیرش (پیش‌فرض ۱۸ از ۲۰) |

**نکته:** قاعده‌ی `targetLayerIndex: 3` با `usageScope: "intra_layer_or_audit"` تعریف شده تا با قاعده R-08 در `nodes.yaml` (که لایه ۴ را منبع دانش می‌داند) تعارض نداشته باشد.

**بارگذاری با:** `FixtureConfigLoader.load_transfer_rules()` و `FixtureConfigLoader.load_hierarchy()`

---

### ۱.۲. `nodes.yaml`

**هدف:** تعریف لایه‌های سازمانی (۰ تا ۴) و گره‌های سازمانی سلسله‌مراتبی به همراه قواعد اعتبارسنجی R-01 تا R-08.

**ساختار کلیدی:**

```yaml
version: "0.1.1-phase1"

layers:
  - layerIndex: 0
    name: "لایه راهبردی"
    required: true
  # ... (۰ تا ۴)

nodes:
  - nodeId: "N-000"
    name: "مرجع راهبردی نمونه"
    layerIndex: 0
    parentNodeId: null
    activityDomain: "راهبری کلان"
    boundaryType: "ROOT"
    confidentiality: "internal"
    transfer: null

  - nodeId: "N-300"
    name: "تیم پروژه پ"
    layerIndex: 3
    parentNodeId: "N-200"
    activityDomain: "پیاده‌سازی RAG"
    boundaryType: "INTERMEDIATE"
    confidentiality: "restricted"
    transfer:
      ruleLayerIndex: 2
      extraRequiredFields:
        - "test_notes"
```

**فیلدهای هر گره:**

| فیلد | توضیح |
|------|-------|
| `nodeId` | شناسه یکتا با الگوی `N-xxx` |
| `layerIndex` | لایه سازمانی (۰ تا ۴) |
| `parentNodeId` | شناسه والد (`null` فقط برای ROOT) |
| `activityDomain` | حوزه فعالیت (اجباری، غیرخالی) |
| `boundaryType` | `ROOT` / `INTERMEDIATE` / `LEAF` / `KNOWLEDGE_SOURCE` |
| `confidentiality` | `public` / `internal` / `restricted` / `confidential` |
| `transfer` | پیکربندی انتقال (`null` برای ROOT و KNOWLEDGE_SOURCE) |

**گر ه‌های نمونه:**

| Node ID | نام | لایه | والد | نوع |
|---------|-----|------|------|-----|
| `N-000` | مرجع راهبردی | ۰ | — | ROOT |
| `N-100` | سازمان مادر الف | ۱ | `N-000` | INTERMEDIATE |
| `N-200` | مرکز پژوهشی ب | ۲ | `N-100` | INTERMEDIATE |
| `N-300` | تیم پروژه پ | ۳ | `N-200` | INTERMEDIATE |
| `N-301` | تیم پژوهشی ت | ۳ | `N-200` | LEAF |
| `N-400` | آرشیو اسناد محلی | ۴ | `N-200` | KNOWLEDGE_SOURCE |

**قواعد اعتبارسنجی (R-01 تا R-08):** در `validation_rules` همین فایل تعریف شده‌اند و توسط `OrganizationHierarchyValidator` در دامنه اجرا می‌شوند.

**بارگذاری با:** `FixtureConfigLoader.load_hierarchy()` → خروجی `(layers, nodes, transfer_rules)`

---

### ۱.۳. `models.yaml`

**هدف:** رجیستری استاندارد مدل‌های کوچک آفلاین در سه نقش `embedding`، `llm` و `lvm`.

**ساختار کلیدی:**

```yaml
version: "0.2.0-phase2"

runtime_policy:
  offlineOnly: true
  allowCloudFallback: false
  defaultLoadingStrategy: "sequential_single_model"
  referenceHardwareProfile: "system_a_rtx3050_6gb"

models:
  - modelId: "bge-m3"
    role: "embedding"
    sourceName: "BAAI/BGE-M3"
    format: "safetensors-fp16"
    localWeightsPath: "models/weights/bge-m3"
    providerEndpoint: "http://localhost:1235/v1"
    contextWindow: 8192
    maxOutputTokens: 1024
    embeddingDimension: 1024
    temperatureDefault: 0.0
    status: "selected"
    supportedTasks: [...]
    hardwareConstraints:
      minVramGb: 2.0
      minRamGb: 4.0
      preferredDevice: "gpu"
      cpuFallbackSupported: true
      executionMode: "sequential_load_unload"
```

**مدل‌های ثبت‌شده:**

| نقش | مدل | فرمت | VRAM حداقل |
|-----|------|------|-----------|
| `embedding` | BGE-M3 (`bge-m3`) | safetensors-fp16 | ۲.۰ گیگ |
| `llm` | Diba-Base (`diba-base`) | gguf-q4_k_m | ۴.۰ گیگ |
| `lvm` | GLM-OCR (`glm-ocr`) | safetensors-bf16 | ۳.۵ گیگ |

**سیاست اجرایی:** هر سه مدل روی ۶ گیگابایت VRAM همزمان بارگذاری نمی‌شوند (`concurrentWithLlmOn6GbVram: false`) و به‌صورت ترتیبی بارگذاری/تخلیه می‌شوند.

**بارگذاری با:**
- `FixtureConfigLoader.load_models_registry()` → خروجی `OfflineModelSpec[]`
- `load_and_validate_models_registry()` در `scripts/test/benchmark_offline_models.py` → اعتبارسنجی و ایندکس بر اساس نقش

---

## ۲. `prompts/` — قالب‌های پرامپت نسخه‌بندی‌شده

این پوشه شامل **۱۲ فایل YAML مستقل** است که هر کدام یک دسته پرامپت را تعریف می‌کند. هر فایل شامل `promptId`، `taskType`، `activeVersion` و لیست `versions` است. **نسخه‌های قبلی هرگز حذف نمی‌شوند** و در `history` نگهداری می‌شوند.

### نگاشت فایل به دسته پرامپت

| فایل | `promptId` | `taskType` |
|------|------------|-----------|
| `qa.yaml` | `PRM-QA` | `qa` |
| `technical_summary.yaml` | `PRM-TECH-SUM` | `technical_summary` |
| `managerial_summary.yaml` | `PRM-EXEC-SUM` | `executive_managerial_summary` |
| `issue_decomposition.yaml` | `PRM-ISSUE-DEC` | `issue_decomposition` |
| `sub_issue_comparison.yaml` | `PRM-SUBISSUE-CMP` | `sub_issue_comparison` |
| `executor_analysis.yaml` | `PRM-EXEC-ANL` | `executor_expertise_analysis` |
| `research_gap.yaml` | `PRM-GAP-ID` | `research_gap_identification` |
| `knowledge_transfer.yaml` | `PRM-TRANSFER-PREP` | `knowledge_transfer_prep` |
| `rfp_generation.yaml` | `PRM-RFP-GEN` | `rfp_generation` |
| `proposal_evaluation.yaml` | `PRM-PROP-EVAL` | `proposal_evaluation` |
| `feedback_generation.yaml` | `PRM-FEEDBACK-GEN` | `feedback_generation` |
| `judge.yaml` | `PRM-JUDGE` | `llm_as_judge` |

### ساختار هر فایل

```yaml
promptId: "PRM-QA"
taskType: "qa"
targetModelId: "diba-base"
activeVersion: "1.0.0"

versions:
  - version: "1.0.0"
    lastUpdated: "2026-10-29"
    systemTemplate: |
      تو دستیار پرسش‌وپاسخ مبتنی بر منبع در هسته RAG سلسله‌مراتبی آفلاین هستی...
    userTemplate: |
      محدوده گره: {{node_scope}}
      زمینه بازیابی‌شده:
      {{context}}
      پرسش کاربر:
      {{query}}
    requiredVariables:
      - "node_scope"
      - "context"
      - "query"
    expectedOutputFormat:
      format: "json"
      requiredKeys:
        - "answer"
        - "source_references"
        - "uncertainty_declared"
        - "uncertainty_reason"
    citationPolicy: "strict_source_id"
    judgeRubricId: "RUBRIC-DEFAULT-20"

  - version: "0.9.0"
    # نسخه قدیمی...
```

### فیلدهای کلیدی

| فیلد | توضیح |
|------|-------|
| `version` | نسخه با فرمت semver (مثلاً `1.0.0`، `0.9.0`) |
| `systemTemplate` | دستور سیستمی |
| `userTemplate` | قالب کاربر با متغیرهای `{{variable}}` |
| `requiredVariables` | متغیرهایی که باید پیش از assembly تأمین شوند |
| `expectedOutputFormat` | `json` یا `markdown` به همراه `requiredKeys` |
| `citationPolicy` | `strict_source_id` / `optional` / `none` |
| `judgeRubricId` | شناسه Rubric داور (مثلاً `RUBRIC-DEFAULT-20`) |

### بارگذاری با:
- `YamlPromptTemplateRepository(prompts_path=data/prompts)` → رجیستری نسخه‌بندی‌شده
- `FixtureConfigLoader.load_prompt_registry()` و `FixtureConfigLoader.bootstrap_in_memory_uow()`
- `PromptAssembler.assemble(prompt_id=..., variables=...)`

---

## ۳. `sample-documents/` — اسناد نمونه برای تست استخراج

این پوشه شامل اسناد واقعی در سه فرمت مختلف است که برای تست‌های ingestion (فاز ۴) و نمایش قابلیت Docling استفاده می‌شوند.