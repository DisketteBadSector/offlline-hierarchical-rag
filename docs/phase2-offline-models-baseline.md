# گزارش فاز ۲: خط پایه مدل‌های آفلاین، روش اجرای محلی، محدودیت‌های فنی و برنامه آزمون ۱۲ دسته پرامپت

**نسخه سند:** `1.0.0-phase2`  
**تاریخ به‌روزرسانی:** `2026-10-29`  
**وضعیت:** تکمیل‌شده (آماده ورود به فاز ۳ - طراحی معماری DDD و مدل دامنه)

---

## ۱. گزارش خط پایه مدل‌های کوچک آفلاین و رجیستری `models.yaml`

در راستای قید اصلی پروژه مبنی بر اجرای کاملاً آفلاین و عدم وابستگی به سرویس‌های ابری، سه مدل کوچک محلی برای سه نقش مکمل **`embedding`**، **`llm`** و **`lvm`** انتخاب و در فایل رجیستری استاندارد [`data/fixtures/models.yaml`](file:///d:/git2/offlline-hierarchical-rag/data/fixtures/models.yaml) ثبت شدند.

### ۱.۱ ساختار استاندارد رجیستری مدل‌ها (`data/fixtures/models.yaml`)
هر رکورد مدل در `data/fixtures/models.yaml` شامل ۸ فیلد استاندارد اجباری زیر است:
- `modelId`: شناسه یکتای مدل در کد و تست‌ها (`bge-m3`، `diba-base`، `glm-ocr`).
- `role`: نقش مدل در معماری (`embedding`، `llm`، `lvm`).
- `providerEndpoint`: آدرس سرور محلی سازگار با OpenAI API (`http://localhost:1235/v1`).
- `contextWindow`: حداکثر طول پنجره زمینه (بر حسب توکن).
- `maxOutputTokens`: حداکثر توکن خروجی مجاز در هر فراخوانی (یا بعد بردار برای مدل Embedding).
- `temperatureDefault`: دمای پیش‌فرض تولید (`0.0` برای تضمین قطعی‌بودن و پایداری خروجی ساخت‌یافته).
- `supportedTasks`: فهرست وظایف دامنه پشتیبانی‌شده توسط مدل.
- `hardwareConstraints`: قیود سخت‌افزاری شامل حداقل `VRAM`، حداقل `RAM`، حالت بارگذاری و تنظیمات کوانتیزاسیون.

### ۱.۲ مشخصات فنی سه مدل منتخب

| ویژگی | ۱. مدل Embedding | ۲. مدل زبانی (LLM) | ۳. مدل بینایی-زبانی (LVM) |
| --- | --- | --- | --- |
| **شناسه رجیستری (`modelId`)** | `bge-m3` | `diba-base` | `glm-ocr` |
| **نام منبع (`sourceName`)** | `BAAI/BGE-M3` | `Dibachain/Diba-Base` | `zai-org/GLM-OCR` |
| **نقش (`role`)** | `embedding` | `llm` | `lvm` |
| **تعداد پارامترها و قالب وزن** | `569M` (`safetensors-fp16`) | `~4B` (`gguf-q4_k_m`) | `0.9B` (`safetensors-bf16`) |
| **پنجره زمینه (`contextWindow`)** | `8192` توکن (پیشنهاد قطعه: `512`) | `4096` توکن (GPU 6GB) / `8192` (CPU 24GB) | `4096` توکن |
| **حداکثر خروجی (`maxOutputTokens`)** | بردار `1024` بعدی | `1024` توکن | `2048` توکن |
| **دمای پیش‌فرض (`temperatureDefault`)** | `0.0` | `0.0` | `0.0` |
| **حداقل حافظه (`VRAM` / `RAM`)** | `2.0 GB VRAM` / `4.0 GB RAM` | `4.0 GB VRAM` / `8.0 GB RAM` | `3.5 GB VRAM` / `8.0 GB RAM` |
| **نقطه اتصال (`providerEndpoint`)** | `http://localhost:1235/v1` | `http://localhost:1235/v1` | `http://localhost:1235/v1` |

### ۱.۳ جدول نتایج خط پایه عملکرد (Baseline Metrics)

سنجش خط پایه بر روی دو پروفایل سخت‌افزاری هدف پروژه تعریف و اجرا شده است:
- **سیستم مرجع A (مقید به VRAM):** پردازنده Core i5 نسل ۱۳، رم `8~16 GB`، کارت گرافیک `NVIDIA RTX 3050 (6 GB VRAM)`.
- **سیستم پشتیبان B (مبتنی بر RAM بالا):** پردازنده Core i7 نسل ۱۱، رم `24 GB`، کارت گرافیک `MX450 (2 GB VRAM)`.

| مدل (`role`) | شاخص مورد سنجش | مقدار خط پایه (سیستم مرجع A - GPU) | مقدار خط پایه (سیستم B / Fallback CPU) | معیار پذیرش فنی در پروژه |
| --- | --- | --- | --- | --- |
| `bge-m3` (`embedding`) | بعد بردار و نرمال‌سازی L2 | `1024` بعد (نرمال‌شده) | `1024` بعد (نرمال‌شده) | تطابق دقیق با بعد نمایه برداری محلی |
| `bge-m3` (`embedding`) | سرعت تولید بردار (قطعات ۵۱۲ توکنی) | `~45-65 texts/s` | `~8-14 texts/s` | پاسخ‌گویی به پرسش در کمتر از `200 ms` |
| `bge-m3` (`embedding`) | تفکیک معنایی فارسی (`Cosine Similarity`) | جملات هم‌معنی `> 0.72` / نامرتبط `< 0.35` | جملات هم‌معنی `> 0.72` / نامرتبط `< 0.35` | کسب `Recall@5 >= 0.80` در اسناد نمونه |
| `diba-base` (`llm`) | زمان اولین توکن (`TTFT`) | `~180 - 350 ms` | `~650 - 1200 ms` | کمتر از `1500 ms` در پرامپت‌های استاندارد |
| `diba-base` (`llm`) | سرعت تولید توکن (`tok/s`) | `~24 - 38 tok/s` | `~6 - 11 tok/s` | حداقل `15 tok/s` روی GPU و `5 tok/s` روی CPU |
| `diba-base` (`llm`) | پایداری خروجی ساختاریافته (`JSON Stability`) | `>= 95%` (با `temperature=0.0`) | `>= 95%` (با `temperature=0.0`) | پارس موفق JSON و حضور تمامی کلیدهای اجباری |
| `glm-ocr` (`lvm`) | زمان تحلیل چیدمان و استخراج صفحه | `~1.8 - 3.2 s/page` | `~9.0 - 15.0 s/page` | استخراج صحیح جداول، سرتیترها و متن فارسی |
| **مجموع بار حافظه** | اجرای همزمان هر سه مدل | **۷.۵ تا ۹ گیگابایت VRAM (بیش از ظرفیت ۶GB)** | **حدود ۱۰ گیگابایت RAM (قابل اجرا)** | **اجرای ترتیبی اجباری (`sequential_single_model`)** |

---

## ۲. روش اجرای محلی و اسکریپت‌های تست و بنچمارک

### ۲.۱ معماری اجرای محلی بدون اینترنت
1. **سرور استنتاج محلی سازگار با OpenAI API:** مدل‌های محلی از طریق سرور آفلاین (مانند `LM Studio Local Server`، `llama-server` از مجموعه `llama.cpp` یا `vLLM/Ollama` محلی) روی آدرس `http://localhost:1235/v1` سرویس‌دهی می‌شوند.
2. **سیاست بارگذاری ترتیبی (`Sequential Model Loading`):**
   - به دلیل اینکه مجموع مصرف حافظه گرافیکی سه مدل (`~7.5 - 9.0 GB VRAM`) از سقف `6 GB VRAM` سیستم مرجع A بیشتر است، اجرای مدل‌ها به ۳ مرحله ترتیبی تفکیک می‌شود:
     - **مرحله الف (پیش‌پردازش بصری اسناد اسکن‌شده/جدول‌دار):** فقط `glm-ocr` بارگذاری شده، خروجی چیدمان در فراداده نسخه دانش ذخیره می‌شود و سپس مدل از VRAM تخلیه (`unload`) می‌گردد.
     - **مرحله ب (خردسازی، نمایه‌سازی و بازیابی برداری):** مدل `bge-m3` بارگذاری شده و بردارهای `KnowledgeChunk` و `RetrievalQuery` را محاسبه می‌کند.
     - **مرحله ج (استدلال، خلاصه‌سازی، انتقال دانش، RFP و داوری):** مدل `diba-base` با `contextWindow: 4096` و `temperature: 0.0` بارگذاری و اجرا می‌شود.

### ۲.۲ اسکریپت‌های سنجش و پایش خط پایه
دو اسکریپت اجرایی در مسیر `scripts/test/` توسعه داده شده‌اند که تنظیمات را مستقیماً از [`data/fixtures/models.yaml`](file:///d:/git2/offlline-hierarchical-rag/data/fixtures/models.yaml) می‌خوانند:
1. **[`scripts/test/test_llm.py`](file:///d:/git2/offlline-hierarchical-rag/scripts/test/test_llm.py):**
   - بررسی اتصال به `providerEndpoint`، شناسایی مدل فعال، سنجش زمان اولین توکن (`TTFT`)، سرعت تولید توکن (`tok/s`) و آزمون پایداری خروجی JSON در چند تکرار متوالی.
2. **[`scripts/test/benchmark_offline_models.py`](file:///d:/git2/offlline-hierarchical-rag/scripts/test/benchmark_offline_models.py):**
   - اعتبارسنجی کامل اسکیمای `data/fixtures/models.yaml` برای هر سه نقش `embedding`، `llm` و `lvm`.
   - اجرای بنچمارک سه‌گانه:
     - **تست `embedding`:** تولید بردار ۱۰۲۴ بعدی، سنجش سرعت کدگذاری و آزمون تمایز معنایی کسینوسی میان جملات مرتبط و نامرتبط فارسی.
     - **تست `llm`:** سنجش `TTFT`، سرعت تولید توکن (`tok/s`) و نرخ پایداری اسکیمای JSON در وظیفه داوری (`LLM-as-a-Judge`).
     - **تست `lvm`:** سنجش توصیف چیدمان صفحه سند (`detected_elements`، `tables_count`، `ocr_confidence`).
   - **قابلیت Fallback خودکار (`offline_fallback`):** در صورتی که سرور محلی در لحظه اجرای اسکریپت روشن نباشد (یا با فلگ `--offline-fallback` اجرا شود)، هر دو اسکریپت بدون خطا به حالت شبیه‌ساز قطعی آفلاین سوییچ کرده و صحت زنجیره پردازش و اسکیماها را گزارش می‌کنند:
     ```powershell
     python scripts/test/test_llm.py --runs 3
     python scripts/test/benchmark_offline_models.py --runs 3
     ```

---

## ۳. محدودیت‌های فنی شناسایی‌شده و راهکارهای معماری

| ردیف | محدودیت شناسایی‌شده در فاز ۲ | اثر بر سامانه | راهکار جبرانی در طراحی معماری و دامنه (فاز ۳ به بعد) |
| :---: | --- | --- | --- |
| ۱ | **محدودیت سقف ۶ گیگابایت VRAM در سیستم مرجع A** | عدم امکان بارگذاری همزمان `bge-m3`، `diba-base` و `glm-ocr` روی GPU | تعریف پورت `ModelRuntimePort` با عملیات صریح `load(model_id)` و `unload(model_id)` و اجرای ترتیبی مراحل خط لوله. |
| ۲ | **محدودیت پنجره زمینه (`4096` توکن) در مدل‌های کوچک محلی** | سرریز شدن زمینه در صورت ارسال اسناد کامل یا بسته‌های انتقال بسیار طولانی | الف) محدود کردن طول هر `KnowledgeChunk` به حداکثر `512` توکن؛ ب) محدود کردن `Top-K` بازیابی به `3..5` قطعه؛ ج) هماهنگ‌سازی سقف کاراکتر `maxLength` در `layers.yaml` (۲۰۰۰ تا ۲۰۰۰۰ کاراکتر) با بودجه توکن پرامپت. |
| ۳ | **شکنندگی قالب خروجی JSON در مدل‌های ۴ میلیارد پارامتری** | احتمال افزودن توضیحات اضافی یا بلاک مارک‌داون ```` ```json ```` در اطراف خروجی | الف) تثبیت `temperatureDefault: 0.0`؛ ب) الزام صریح در `System Prompt`؛ ج) پیاده‌سازی لایه نرمال‌ساز و `OutputParser` مقاوم همراه با اعتبارسنجی `Pydantic` و مکانیزم تلاش مجدد (Retry). |
| ۴ | **انحراف یا خوش‌بینی بیش از حد در امتیازدهی داور (`LLM-as-a-Judge`)** | احتمال دادن امتیاز بالا به خلاصه‌های فاقد ارجاع منبع یا جمع نادرست امتیاز معیارها | الف) شکستن پرامپت داوری به ۷ معیار مجزا مطابق `JudgeRubric`؛ ب) محاسبه جمع امتیاز کل (`0..20`) در کد دامنه پایتون به‌جای تکیه بر جمع ریاضی مدل زبانی؛ ج) رد خودکار خروجی فاقد ارجاع منبع و بازگشت به آخرین `TransferPackage` پذیرفته‌شده قبلی در صورت امتیاز `< 18`. |
| ۵ | **هزینه زمانی بالای LVM (`glm-ocr`) روی CPU در اسناد طولانی** | کندی شدید در صورت ارسال تمام صفحات PDF متنی معمولی به LVM | اجرای انتخابی LVM صرفاً برای صفحاتی که استخراج متنی `pypdf` در آن‌ها خالی/کم‌تراکم است یا پرچم `requires_visual_layout=True` دارند. |

---

## ۴. برنامه آزمون پرامپت‌ها برای ۱۲ دسته وظیفه پروژه (مطابق بخش ۸.۳ پروپوزال)

بر اساس بخش ۸.۳ سند پروپوزال، تمامی تعاملات با مدل زبانی در **۱۲ دسته وظیفه استاندارد** طبقه‌بندی می‌شوند. برای هر وظیفه، شناسه قالب پرامپت، کانتکست مالک، ورودی‌های جایگذاری (`Placeholders`)، کلیدهای الزامی خروجی JSON و معیار پذیرش در جدول زیر تثبیت شده است:

### ۴.۱ جدول جامع ۱۲ دسته پرامپت پروژه

| ردیف | دسته وظیفه (مطابق بخش ۸.۳ پروپوزال) | شناسه وظیفه (`supportedTasks`) | شناسه قالب پرامپت | کانتکست دامنه | ورودی‌های پرامپت (`Placeholders`) | کلیدهای الزامی خروجی ساختاریافته (JSON Schema) | الزام ارجاع منبع و عدم قطعیت | معیار موفقیت در آزمون پرامپت و آزمون واحد |
| :---: | --- | --- | --- | --- | --- | --- | --- | --- |
| ۱ | **پرسش و پاسخ** | `question_answering` | `qa-v1` | `rag` | `{{node_id}}`, `{{context_chunks}}`, `{{query}}` | `answer`, `source_references`, `uncertainty_flag`, `uncertainty_reason`, `related_issues` | ارجاع منبع: **بله** / اعلام عدم قطعیت: **بله** | عدم تناقض با منبع؛ ارجاع دقیق به `chunk_id`؛ اعلام صریح عدم قطعیت هنگام خالی یا ناکافی بودن زمینه. |
| ۲ | **خلاصه‌سازی فنی** | `technical_summarization` | `sum-tech-v1` | `transfer` / `rag` | `{{node_id}}`, `{{detail_level}}`, `{{source_items}}`, `{{max_length}}` | `methods`, `research_results`, `limitations`, `executor_capabilities`, `source_references` | ارجاع منبع: **بله** / اعلام عدم قطعیت: **بله** | پوشش کامل کلیدهای قاعده لایه ۲ و ۳ در `layers.yaml`؛ رعایت سقف طول و کسب امتیاز داور `>= 18`. |
| ۳ | **خلاصه‌سازی مدیریتی و اجرایی** | `managerial_executive_summarization` | `sum-exec-v1` | `transfer` | `{{node_id}}`, `{{target_layer_index}}`, `{{detail_level}}`, `{{raw_package_data}}`, `{{max_length}}` | برای لایه ۱: `project_status`, `issue_status`, `research_gaps`, `resource_needs`, `rfp_status` / برای لایه ۰: `key_achievements`, `critical_risks`, `unresolved_issues`, `strategic_recommendations` | ارجاع منبع: **بله** / اعلام عدم قطعیت: **بله** | حذف جزئیات کد و الگوریتم؛ تمرکز بر ریسک‌ها، تصمیمات و دستاوردها منطبق با قواعد لایه ۰ و ۱ در `layers.yaml`. |
| ۴ | **تجزیه مسئله** | `issue_decomposition` | `issue-decomp-v1` | `issue` | `{{node_id}}`, `{{parent_issue}}`, `{{related_knowledge}}` | `parent_issue_id`, `sub_issues` (آرایه‌ای از اشیاء شامل `title`, `description`, `relative_order`, `required_expertise`, `dependency_ids`) | ارجاع منبع: **اختیاری** / اعلام عدم قطعیت: **بله** | تولید ۳ تا ۷ زیرمسئله مستقل، قابل اقدام، غیرتکراری و دارای `relative_order` معتبر. |
| ۵ | **مقایسه زیرمسئله‌ها** | `subissue_comparison` | `subissue-cmp-v1` | `issue` | `{{candidate_issue}}`, `{{existing_issues_list}}`, `{{similarity_threshold}}` | `is_duplicate`, `overlapping_issue_ids`, `similarity_assessment`, `merge_or_keep_recommendation`, `rationale` | ارجاع منبع: **بله** (شناسه مسائل) / عدم قطعیت: **خیر** | تشخیص صحیح مسائل و زیرمسئله‌های هم‌پوشان یا تکراری و ارائه پیشنهاد ادغام یا تفکیک مستدل. |
| ۶ | **تحلیل تخصص مجریان** | `executor_expertise_analysis` | `exec-analysis-v1` | `executor` / `ike` | `{{issue_details}}`, `{{candidate_executors}}`, `{{project_history}}` | `recommended_executors` (شامل `executor_id`, `match_score`, `matched_skills`, `evidence_references`), `skill_gaps` | ارجاع منبع: **بله** (پروفایل/پروژه) / عدم قطعیت: **بله** | ارجاع ۱۰۰٪ پیشنهادها به شواهد پروفایل پرسنلی یا سوابق پروژه و شناسایی کمبود تخصص در گره. |
| ۷ | **شناسایی شکاف پژوهشی** | `research_gap_identification` | `research-gap-v1` | `issue` / `ike` | `{{node_id}}`, `{{unresolved_issues}}`, `{{knowledge_coverage}}`, `{{failed_judge_records}}` | `identified_gaps` (شامل `gap_id`, `title`, `cause_category`, `severity`, `linked_issue_ids`, `recommended_research_action`) | ارجاع منبع: **بله** / اعلام عدم قطعیت: **بله** | تشخیص دقیق شکاف بر اساس ۴ علت: نبود دانش، قدیمی‌بودن دانش، تعارض منابع یا امتیاز داور کمتر از ۱۸. |
| ۸ | **آماده‌سازی انتقال دانش** | `knowledge_transfer_preparation` | `transfer-prep-v1` | `transfer` | `{{source_node}}`, `{{target_layer_rule}}`, `{{extra_required_fields}}`, `{{draft_package}}` | `refined_payload`, `missing_required_fields`, `removed_excessive_details`, `ai_improvement_suggestions`, `source_references` | ارجاع منبع: **بله** / اعلام عدم قطعیت: **بله** | تضمین حضور تمامی فیلدهای `requiredFields` (از `layers.yaml`) + `extraRequiredFields` (از `nodes.yaml`). |
| ۹ | **تولید RFP** | `rfp_generation` | `rfp-gen-v1` | `rfp` | `{{origin_issue}}`, `{{sub_issues}}`, `{{research_gaps}}`, `{{technical_constraints}}` | `rfp_title`, `problem_statement`, `scope`, `technical_requirements`, `expected_deliverables`, `evaluation_criteria`, `constraints`, `required_expertise`, `origin_issue_id` | ارجاع منبع: **بله** / اعلام عدم قطعیت: **خیر** | حضور ۱۰۰٪ بخش‌های اجباری RFP و پیوند معتبر به شناسه مسئله و شکاف پژوهشی منشأ. |
| ۱۰ | **ارزیابی پیشنهاد** | `proposal_evaluation` | `proposal-eval-v1` | `rfp` | `{{rfp_document}}`, `{{proposal_content}}`, `{{evaluation_rubric}}` | `proposal_id`, `rfp_id`, `criteria_scores` (امتیاز و دلیل برای هر معیار), `total_score`, `mandatory_requirements_met`, `feasibility_assessment` | ارجاع منبع: **بله** (بندهای RFP و پیشنهاد) / عدم قطعیت: **بله** | امتیازدهی تفکیکی به ازای هر معیار RFP همراه با شواهد متنی از داخل پیشنهاد ثبت‌شده. |
| ۱۱ | **تولید بازخورد** | `feedback_generation` | `feedback-gen-v1` | `rfp` | `{{rfp_document}}`, `{{proposal_content}}`, `{{evaluation_result}}` | `strengths`, `weaknesses`, `missing_requirements`, `ambiguous_assumptions`, `technical_risks`, `clarification_questions`, `final_recommendation` (`select`/`revise`/`reject`) | ارجاع منبع: **بله** / اعلام عدم قطعیت: **خیر** | هر نقد و نقطه ضعف باید به یک بند مشخص در RFP و بخش متناظر در پیشنهاد ارجاع دهد. |
| ۱۲ | **امتیازدهی LLM به‌عنوان داور** | `llm_as_judge_scoring` | `judge-rubric-v1` | `judge` | `{{task_type}}`, `{{source_knowledge}}`, `{{user_request_or_rule}}`, `{{generated_output}}`, `{{judge_rubric}}`, `{{threshold}}` | `criteria_breakdown` (۷ معیار با سقف ۵، ۴، ۳، ۳، ۲، ۲، ۱), `total_score` (`0..20`), `passed` (`bool`), `explanation`, `missing_information`, `unsupported_claims`, `detail_level_feedback`, `revision_suggestions` | ارجاع منبع: **بله** / اعلام عدم قطعیت: **خیر** | همخوانی دقیق امتیازها با سقف هر معیار در `JudgeRubric`؛ تشخیص ادعاهای بدون پشتوانه و اعمال حد آستانه `18/20`. |

---

### ۴.۲ پروتکل اجرای آزمون پرامپت در فازهای بعدی (فاز ۳ تا ۱۴)
1. **آزمون واحد مونتاژ پرامپت (`Prompt Assembly Unit Tests`):** در `tests/unit/model_registry/` و `tests/unit/rag/` بررسی می‌شود که تمامی جایگذاری‌ها (`{{...}}`) بدون باقی‌ماندن متغیر خام پر شده و محدودیت‌های طول زمینه و محرمانگی گره رعایت شده‌اند.
2. **آزمون واحد Parser و اعتبارسنجی اسکیما (`Output Schema Validation Tests`):** برای هر یک از ۱۲ دسته پرامپت، پاسخ‌های نمونه معتبر و نامعتبر (شامل JSON شکسته، کلیدهای مفقود و امتیاز خارج از بازه) از طریق `MockLLMAdapter` به Parser داده شده و رفتار خطایابی و اعتبارسنجی دامنه سنجیده می‌شود.
3. **آزمون پایداری روی مدل محلی (`Live Benchmark via benchmark_offline_models.py`):** در محیط دارای سرور محلی فعال، هر پرامپت حداقل ۳ بار با `temperature = 0.0` اجرا شده و نرخ موفقیت پارس JSON (`>= 95%`) و میانگین امتیاز داور (`>= 18/20`) ثبت می‌گردد.
