# گزارش فاز ۴: زیربنای ورود سند (Document Ingestion Foundation)، مشخصات فراداده، ارجاع به منبع و الگوریتم خردسازی

**نسخه سند:** `1.0.0-phase4`  
**تاریخ به‌روزرسانی:** `2026-10-29`  
**وضعیت:** تکمیل‌شده و پوشش‌داده‌شده با آزمون‌های واحد (`100% Passing`)

---

## ۱. مشخصات Schema فراداده (`MetadataRecord`) و فیلترگذاری در بازیابی

شیء مقداری تغییرناپذیر [`MetadataRecord`](file:///d:/git2/offlline-hierarchical-rag/src/offline_rag/domain/shared/value_objects.py#L406-L561) در کانتکست مدیریت دانش و هسته مشترک، مسئول ثبت، اعتبارسنجی و فیلترگذاری فراداده اسناد، مسائل و خروجی‌های تولیدشده سامانه است (مطابق بخش ۷.۵ پروپوزال).

### ۱.۱ فیلدهای الزامی و اختیاری Schema فراداده

| ردیف | نام فیلد | نوع داده | وضعیت | قاعده اعتبارسنجی |
| :---: | --- | --- | :---: | --- |
| ۱ | `domain` | `str` | **الزامی** | رشته غیرخالی پس از حذف فاصله‌های اضافی. |
| ۲ | `node_id` | `NodeIdentifier` | **الزامی** | شناسه معتبر گره سازمانی مالک دانش (مانند `N-200`، `N-300`). |
| ۳ | `document_type` | `str` | **الزامی** | نوع سند (مانند `technical_report`، `managerial_report`، `policy`، `research_notes`). |
| ۴ | `confidentiality` | `str` | **الزامی** | یکی از سطوح مجاز: `public`، `internal`، `restricted`، `confidential`. |
| ۵ | `initial_version_id` / `initial_version_tag` | `str` / `VersionTag` | **الزامی** | شناسه نسخه اولیه سند (مانند `v1` با `sequence_number >= 1`). |
| ۶ | `active_version_id` / `active_version_tag` | `str` / `VersionTag` | **الزامی** | شناسه نسخه فعال که شماره توالی آن باید `>= initial_version_tag` باشد. |
| ۷ | `author_or_owner` | `str` | الزامی (پیش‌فرض `"system"`) | نویسنده یا واحد مالک سند (رشته غیرخالی). |
| ۸ | `sub_domain` | `str \| None` | اختیاری | زیرحوزه تخصصی دانش یا مسئله. |
| ۹ | `project_id` | `str \| None` | اختیاری | شناسه پروژه مرتبط. |
| ۱۰ | `related_issue_id` | `str \| None` | اختیاری | شناسه مسئله مرتبط (`Issue`). |
| ۱۱ | `related_executor_id` | `str \| None` | اختیاری | شناسه مجری مرتبط (`Executor`). |
| ۱۲ | `related_rfp_id` | `str \| None` | اختیاری | شناسه `RFP` مرتبط. |
| ۱۳ | `status` | `str` | پیش‌فرض `"active"` | وضعیت قلم دانش در چرخه عمر. |
| ۱۴ | `priority` | `str` | پیش‌فرض `"medium"` | یکی از مقادیر `low`، `medium`، `high`، `critical`. |
| ۱۵ | `tags` | `tuple[str, ...]` | اختیاری | برچسب‌های موضوعی نرمال‌شده. |
| ۱۶ | `source_reliability` | `float` | پیش‌فرض `1.0` | عدد اعشاری در بازه بسته `0.0 <= value <= 1.0`. |
| ۱۷ | `target_transfer_layer` | `LayerIndex \| None` | اختیاری | لایه هدف انتقال (`0..4`). |
| ۱۸ | `suggested_detail_level` | `TransferDetailLevel \| None` | اختیاری | سطح جزئیات پیشنهادی (`executive_high`، `managerial`، `technical_summary`، `technical_detailed`). |

### ۱.۲ پشتیبانی از فیلترگذاری در سناریوهای بازیابی
متد [`MetadataRecord.matches_filters()`](file:///d:/git2/offlline-hierarchical-rag/src/offline_rag/domain/shared/value_objects.py#L533-L561) امکان فیلترگذاری ترکیبی بر اساس هر زیرمجموعه‌ای از فیلدهای فراداده (از جمله تطبیق زیرمجموعه برچسب‌ها `tags`، `domain`، `document_type`، `project_id` و `active_version_id`) را در `InMemoryVectorIndex.search()` و خط لوله RAG فراهم می‌کند.

---

## ۲. مدل ارجاع به منبع (`SourceReference`)

شیء مقداری [`SourceReference`](file:///d:/git2/offlline-hierarchical-rag/src/offline_rag/domain/knowledge/entities.py#L31-L82) در کانتکست `knowledge` پیوند تغییرناپذیر و دقیق میان هر قطعه دانش (`KnowledgeChunk`) و سند و نسخه منشأ را برقرار می‌سازد:

| فیلد | نوع | شرح |
| --- | --- | --- |
| `document_id` | `str` | شناسه پایدار سند منبع (`KnowledgeItem.item_id`). |
| `version_id` | `str` | شناسه نسخه سند (مانند `v1`، `v2`). |
| `file_format` | `DocumentFormat` | قالب سند منشأ (`DOC`، `DOCX`، `PDF`، `STRUCTURED_MODEL`). |
| `page_number` | `int \| None` | شماره صفحه (`>= 1`) ویژه اسناد PDF یا اسناد دارای صفحه‌بندی. |
| `section_reference` | `str \| None` | ارجاع بخش، سرفصل جاری یا شناسه جدول/بند. |
| `heading_path` | `tuple[str, ...]` | مسیر سلسله‌مراتبی عنوان‌ها از ریشه سند تا بخش جاری (مانند `("گزارش مدیریتی", "وضعیت پروژه‌ها")`). |

متد `to_citation_string(sequence_index)` رشته استاندارد ارجاع (مانند `DOC-300@v1#p2§روش‌ها:c0`) را برای الصاق در خروجی‌های RAG و بسته‌های انتقال تولید می‌کند.

---

## ۳. معماری آداپتورهای استخراج سند (`DOC`، `DOCX`، `PDF` و `LVM Hook`)

در مسیر [`src/offline_rag/infrastructure/ingestion/`](file:///d:/git2/offlline-hierarchical-rag/src/offline_rag/infrastructure/ingestion/__init__.py) خط لوله استخراج سند بر اساس الگوهای **Adapter** و **Strategy** پیاده‌سازی شده است:

```
                      +-----------------------------------+
                      |     DocumentIngestionService      |
                      +-----------------+-----------------+
                                        |
                                        v
                      +-----------------------------------+
                      |     DocumentExtractorRegistry     |
                      +-----------------+-----------------+
                                        | selects by format (.doc/.docx/.pdf)
              +-------------------------+-------------------------+
              |                         |                         |
              v                         v                         v
  +-----------------------+ +-----------------------+ +-----------------------+
  |  DocExtractorAdapter  | | DocxExtractorAdapter  | |  PdfExtractorAdapter  |
  | (OLE2 Binary UTF-16LE | | (OOXML document.xml,  | | (Page-indexed CMap /  |
  |  + Text Stream Multi) | |  Headings & Tables)   | |  ActualText + LVM Flag|
  +-----------+-----------+ +-----------+-----------+ +-----------+-----------+
              |                         |                         |
              +-------------------------+-------------------------+
                                        | returns ExtractedDocumentContent
                                        v
                      +-----------------------------------+
                      |   requires_lvm_assistance == True |
                      |  ---> LvmLayoutAssistantPort      |
                      |      (MockLvmLayoutAdapter)       |
                      +-----------------------------------+
```

1. **[`build_offline_docling_converter`](file:///d:/git2/offlline-hierarchical-rag/src/offline_rag/infrastructure/ingestion/__init__.py#L38-L77) و پایه مشترک [`_DoclingExtractorBase`](file:///d:/git2/offlline-hierarchical-rag/src/offline_rag/infrastructure/ingestion/__init__.py#L220-L289):**
   - استفاده از یک نمونه کش‌شده (`@lru_cache`) از `DocumentConverter` کتابخانه `docling` با تنظیمات `PdfPipelineOptions(do_ocr=False, do_table_structure=True, generate_page_images=False)` و مسیر وزن‌های آفلاین (`artifacts_path`).
   - تبدیل جریان بایت در حافظه از طریق `DocumentStream` بدون نیاز به فایل موقت و پیمایش `doc.iterate_items()` برای استخراج سرفصل‌ها (`title`/`section_header`)، پاراگراف‌ها، جداول (`table.data.grid`) و شماره صفحات (`prov[0].page_no`).
2. **[`DocxExtractorAdapter`](file:///d:/git2/offlline-hierarchical-rag/src/offline_rag/infrastructure/ingestion/__init__.py#L292-L297) و [`DocExtractorAdapter`](file:///d:/git2/offlline-hierarchical-rag/src/offline_rag/infrastructure/ingestion/__init__.py#L308-L313):**
   - استخراج ساختاریافته پاراگراف‌ها، سلسله‌مراتب `heading_path` و جداول از اسناد Word از طریق `WordFormatOption` در Docling.
3. **[`PdfExtractorAdapter`](file:///d:/git2/offlline-hierarchical-rag/src/offline_rag/infrastructure/ingestion/__init__.py#L300-L305):**
   - استخراج بلوک‌ها و جداول تفکیک‌شده به ازای هر صفحه با `PdfFormatOption` در Docling و علامت‌گذاری خودکار صفحات تصویری (`picture`/`figure`) یا کم‌متن در `flagged_lvm_pages` (`requires_lvm_assistance = True`).
4. **[`LvmLayoutAssistantPort`](file:///d:/git2/offlline-hierarchical-rag/src/offline_rag/application/ports/__init__.py#L75-L89) و [`MockLvmLayoutAdapter`](file:///d:/git2/offlline-hierarchical-rag/src/offline_rag/infrastructure/ingestion/__init__.py#L316-L364):**
   - دریافت اسناد و صفحات علامت‌گذاری‌شده (`requires_lvm_assistance=True`)، تولید توصیف ساختاریافته بصری/چیدمان برای هر صفحه و پیوست کردن آن هم به بلوک‌های استخراج‌شده (`block_type="lvm_layout"`) و هم به فیلد `lvm_extracted_notes` در موجودیت `KnowledgeItem`.

---

## ۴. الگوریتم و قواعد مرزبندی خردسازی دانش (`KnowledgeChunker`)

در فایل [`src/offline_rag/domain/knowledge/chunking.py`](file:///d:/git2/offlline-hierarchical-rag/src/offline_rag/domain/knowledge/chunking.py) سرویس دامنه `KnowledgeChunker` بر اساس سیاست `ChunkingPolicy` (`max_chunk_chars`، `chunk_overlap_chars`، `respect_heading_boundaries` و `respect_page_boundaries`) عمل می‌کند:

1. **حفظ مرز سرفصل‌ها و صفحات:**
   - در صورت فعال بودن `respect_heading_boundaries`، با هر تغییر در `heading_path` قطعه قبلی بسته شده و قطعه جدیدی با مسیر سرفصل جدید آغاز می‌شود.
   - در صورت فعال بودن `respect_page_boundaries`، قطعات از مرز صفحات عبور نمی‌کنند تا `page_number` در `SourceReference` کاملاً دقیق باقی بماند.
2. **عدم شکستن جملات و کلمات از وسط:**
   - متد `_split_preserving_sentences_and_words` ابتدا متن را بر اساس علائم پایانی جمله فارسی و انگلیسی (`.`، `؟`، `?`، `!`، `؛`، `;`، `\n`) تفکیک می‌کند.
   - اگر جمله‌ای طولانی‌تر از `max_chunk_chars` باشد، شکستن صرفاً روی مرز فاصله بین کلمات انجام می‌شود و هیچ کلمه فارسی یا انگلیسی از وسط شکسته نمی‌شود.
   - هم‌پوشانی (`chunk_overlap_chars`) نیز از طریق `_compute_word_safe_overlap` صرفاً روی کلمات کامل انتهایی قطعه قبلی محاسبه می‌گردد.
3. **حفظ کامل فراداده و ارجاع منبع در هر قطعه:**
   - هر `KnowledgeChunk` دارای `chunk_id` یکتا، `sequence_index` متوالی (`0..n-1`)، آفست‌های کاراکتری `context_boundary_start` و `context_boundary_end`، شیء کامل `SourceReference` و `inherited_metadata` ارث‌بری‌شده از `KnowledgeItem` است.
