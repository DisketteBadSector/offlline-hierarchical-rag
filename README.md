# Offline Hierarchical RAG

هسته RAG سلسله‌مراتبی آفلاین برای مدیریت دانش سازمانی، مدل‌سازی مسئله و انتقال دانش مدیریتی.

---

## درباره پروژه

این مخزن یک نمونه نرم‌افزاری آکادمیک است که یک هسته RAG آفلاین، دامنه‌محور و سلسله‌مراتبی را پیاده‌سازی می‌کند.

سامانه دانش سازمانی را در یک ساختار گره‌ای سلسله‌مراتبی مدیریت می‌کند، مسائل را مدل‌سازی و تجزیه می‌کند، مجریان را به دانش و مسئله نگاشت می‌کند، بسته‌های انتقال دانش را برای لایه‌های بالاتر آماده می‌کند و پرسش‌وپاسخ مبتنی بر منبع ارائه می‌دهد.

### ویژگی‌های کلیدی

- **آفلاین:** سعی بر این است که روند اجرای هسته را به صورت کاملا آفلاین حفظ کنیم.
- **دامنه‌محور (DDD):** معماری بر اساس Contextهای مرزبندی‌شده، Aggregateها و Value Objectها.
- **سلسله‌مراتبی:** مدل‌سازی لایه‌های سازمانی (۰ تا ۴) و انتقال دانش بین لایه‌ها.
- **چرخه عمر دانش:** فراداده، نسخه‌بندی، خردسازی، تشخیص تکرار و هم‌پوشانی.
- **مدل Issue-Knowledge-Executor:** نگاشت ساخت‌یافته مسئله، دانش و مجری.
- **داوری LLM:** ارزیابی کیفیت خروجی با Rubric ۲۰ امتیازی.
- **پشتیبانی از DOC، DOCX و PDF:** شامل پردازش کمکی LVM برای اسناد بصری.
- **آزمون واحد:** تنها سطح آزمون پروژه؛ بدون integration/e2e/load.

---

## وضعیت فازها

پروژه در **۱۶ فاز اجرایی** سازمان‌دهی شده است. هر فاز یک فایل مستند مستقل در `docs/` دارد.

**هم‌اکنون فاز ۴ به پایان رسیده است.**

### فازهای تکمیل‌شده

| فاز | عنوان | خروجی اصلی |
|------|-------|-------------|
| ۱ | راه‌اندازی پروژه و کشف دامنه | ساختار DDD، Value Objectها، فیکسچرهای `layers.yaml` و `nodes.yaml`، قواعد R-01 تا R-08 |
| ۲ | خط پایه مدل‌های آفلاین | `models.yaml`، اسکریپت بنچمارک `scripts/test/benchmark_offline_models.py`، پشتیبانی از نقش‌های embedding / llm / lvm |
| ۳ | مدیریت پرامپت و رجیستری مدل | ۱۲ فایل پرامپت نسخه‌بندی‌شده در `data/prompts/`، `PromptAssembler`، `ModelRegistryRepository` |
| ۴ | پایه استخراج و خردسازی اسناد | `DocumentIngestionService`، آداپتورهای Docling برای DOC/DOCX/PDF، `KnowledgeChunker`، `SourceReference`، `MetadataRecord`، LVM Layout Assistant (Mock) |

### فازهای پیش‌رو

فازهای ۵ تا ۱۶ شامل پیاده‌سازی نمایه برداری واقعی، اتصال به سرور مدل محلی، RAG End-to-End، انتقال دانش چرخه‌ای، داوری LLM واقعی، RFP و ارزیابی پیشنهاد، و رابط‌های کاربری است.

---

## وضعیت اجرایی فعلی
 هنوز هیچ entry point اجرایی (CLI، سرور، `main.py`) در پروژه وجود ندارد. روند اجرایی صرفا بر اساس اسکریپت‌‌های فاز به فاز است.

---

## مدل‌های آفلاین ثبت‌شده

| نقش | مدل | فرمت | کاربرد |
|-----|------|------|--------|
| `embedding` | BGE-M3 (`bge-m3`) | safetensors-fp16 | بردارسازی، بازیابی، تشخیص هم‌پوشانی |
| `llm` | Diba-Base (`diba-base`) | gguf-q4_k_m | پرسش‌وپاسخ، خلاصه‌سازی، داوری، RFP و ۱۲ task |
| `lvm` | GLM-OCR (`glm-ocr`) | safetensors-bf16 | تحلیل چیدمان، OCR، استخراج جدول و شکل |

هر سه مدل برای پروفایل سخت‌افزاری مرجع `system_a_rtx3050_6gb` (۶ گیگابایت VRAM) تنظیم شده‌اند و به‌صورت ترتیبی (`sequential_load_unload`) بارگذاری می‌شوند.


```
offline-hierarchical-rag/
│
├── docs/                              # مستندات فازها
│
├── data/
│   ├── fixtures/
│   │   ├── layers.yaml                # قواعد انتقال بین لایه‌ها
│   │   ├── models.yaml                # رجیستری مدل‌های آفلاین
│   │   └── nodes.yaml                 # گره‌های سازمانی
│   ├── prompts/                       # فایل‌های پرامپت
│   │   ├── qa.yaml
│   │   ├── technical_summary.yaml
│   │   ├── managerial_summary.yaml
│   │   ├── issue_decomposition.yaml
│   │   ├── sub_issue_comparison.yaml
│   │   ├── executor_analysis.yaml
│   │   ├── research_gap.yaml
│   │   ├── knowledge_transfer.yaml
│   │   ├── rfp_generation.yaml
│   │   ├── proposal_evaluation.yaml
│   │   ├── feedback_generation.yaml
│   │   └── judge.yaml
│   └── sample-documents/              # نمونه اسناد
│       ├── doc/                       
│       ├── docx/                     
│       └── pdf/                      
│
├── scripts/
│   └── test‍/
│       ├── benchmark_offline_models.py
│       ├── extract_with_docling.py.py #اسکریپت استخراج متن از اسناد
│       └── _common.py
│
├── src/offline_rag/
│   ├── application/                   # لایه کاربرد
│   │   ├── ports/                     # پورت‌های انتزاعی
│   │   └── services/                  # سرویس‌های کاربردی
│   │       ├── document_ingestion_service.py
│   │       └── __init__.py            # Hierarchy/Knowledge/Transfer Services
│   ├── domain/                        # ۱۰ Bounded Context دامنه
│   │   ├── evaluation/                # JudgeScore، JudgeRubric
│   │   ├── executor/                  # Executor
│   │   ├── ike_model/                 # ۱۲ نوع رابطه IKE
│   │   ├── issue/                     # Issue، SubIssue
│   │   ├── knowledge/                 # KnowledgeItem، KnowledgeChunk، Chunker
│   │   ├── model_prompt/              # PromptTemplate، PromptAssembler
│   │   ├── organization/              # Layer، Node، R-01..R-08
│   │   ├── rag/                       # RetrievalQuery، RAGResponse
│   │   ├── rfp/                       # RFP، Proposal، FeedbackReport
│   │   ├── transfer/                  # TransferPackage
│   │   └── shared/                    # Value Objectها، خطاها، base
│   └── infrastructure/                # لایه زیرساخت
│       ├── config/                    # FixtureConfigLoader، YamlPromptTemplateRepository
│       ├── ingestion/                 # Docling adapters، Mock LVM
│       └── persistence/in_memory/     # In-Memory Repositories و Test Doubles
│
├── tests/
│   └── unit/                          # تست پروژه
│
├── pyproject.toml
└── README.md
```