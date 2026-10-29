# گزارش فاز ۳: معماری طراحی دامنه‌محور (DDD)، مدل دامنه اصلی و انتزاع‌های ذخیره‌سازی

**نسخه سند:** `1.0.0-phase3`  
**تاریخ به‌روزرسانی:** `2026-10-29`  
**وضعیت:** تکمیل‌شده و پوشش‌داده‌شده با آزمون‌های واحد (`100% Passing`)

---

## ۱. نمای کلی معماری لایه‌ای (Clean / Hexagonal DDD)

هسته نرم‌افزاری «RAG سلسله‌مراتبی آفلاین» در پکیج [`src/offline_rag`](file:///d:/git2/offlline-hierarchical-rag/src/offline_rag/__init__.py) بر اساس معماری لایه‌ای دامنه‌محور (DDD) و بدون وابستگی به فریم‌ورک‌های خارجی پیاده‌سازی شده است. جهت وابستگی‌ها همواره **به سمت داخل (لایه دامنه)** است:

```
+-----------------------------------------------------------------------------------+
|                        Infrastructure Layer (infrastructure/)                     |
|  - persistence/in_memory/: In-Memory Repositories, VectorIndex, CSV, Mock Models  |
|  - config/: FixtureConfigLoader (layers.yaml, nodes.yaml, models.yaml)            |
+-----------------------------------------+-----------------------------------------+
                                          | implements Ports & Repositories
                                          v
+-----------------------------------------------------------------------------------+
|                         Application Layer (application/)                          |
|  - ports/: UnitOfWork, VectorIndexPort, DocumentStoragePort, CSVTransferExporter, |
|            EmbeddingModelPort, LLMModelPort, LVMModelPort, ModelRuntimePort       |
|  - services/: HierarchyApplicationService, KnowledgeApplicationService,           |
|               TransferApplicationService                                          |
+-----------------------------------------+-----------------------------------------+
                                          | orchestrates
                                          v
+-----------------------------------------------------------------------------------+
|                            Domain Layer (domain/)                                 |
|  - shared/: Entity, AggregateRoot, DomainEvent, DomainError, 11 Value Objects     |
|  - 10 Bounded Contexts: organization, knowledge, rag, issue, executor,            |
|                         ike_model, transfer, evaluation, rfp, model_prompt        |
+-----------------------------------------------------------------------------------+
```

---

## ۲. نگاشت کامل ۱۰ کانتکست مرزبندی‌شده (Bounded Contexts) به کد

| ردیف | کانتکست مرزبندی‌شده | مسیر پکیج در کد | Aggregate Rootها و موجودیت‌ها | سرویس‌های دامنه و قواعد کلیدی | ریپازیتوری دامنه |
| :---: | --- | --- | --- | --- | --- |
| ۱ | سلسله‌مراتب سازمانی | [`domain/organization`](file:///d:/git2/offlline-hierarchical-rag/src/offline_rag/domain/organization/__init__.py) | `Node` (AR)، `Layer` (Entity)، `LayerTransferRule`، `NodeTransferConfig` | `OrganizationHierarchyValidator` (کنترل قواعد `R-01` تا `R-08`) | `NodeRepository`، `LayerRepository` |
| ۲ | مدیریت دانش | [`domain/knowledge`](file:///d:/git2/offlline-hierarchical-rag/src/offline_rag/domain/knowledge/__init__.py) | `KnowledgeItem` (AR)، `KnowledgeVersion` (Entity)، `KnowledgeChunk` (Entity) | نسخه‌بندی افزایشی بدون حذف نسخه‌های قدیم؛ حفظ یک نسخه فعال (`is_active`) | `KnowledgeRepository` |
| ۳ | هسته RAG | [`domain/rag`](file:///d:/git2/offlline-hierarchical-rag/src/offline_rag/domain/rag/__init__.py) | `RAGContextPackage`، `RAGResponse`، `RetrievedChunk` | کنترل مرز دانشی گره (`requesting_node_id`) و الزام ارجاع منبع یا اعلام عدم قطعیت | از طریق `VectorIndexPort` و `KnowledgeRepository` |
| ۴ | مدیریت مسئله | [`domain/issue`](file:///d:/git2/offlline-hierarchical-rag/src/offline_rag/domain/issue/__init__.py) | `Issue` (AR)، `SubIssue` (Entity) | تجزیه مسئله، یکتایی `relative_order` و محاسبه قطعی `ProgressPercentage` | `IssueRepository` |
| ۵ | مدیریت مجری | [`domain/executor`](file:///d:/git2/offlline-hierarchical-rag/src/offline_rag/domain/executor/__init__.py) | `Executor` (AR) | نرمال‌سازی حوزه‌های تخصص فارسی و تطبیق تخصص (`matches_expertise`) | `ExecutorRepository` |
| ۶ | مدل Issue-Knowledge-Executor | [`domain/ike_model`](file:///d:/git2/offlline-hierarchical-rag/src/offline_rag/domain/ike_model/__init__.py) | `IKERelationship` (AR)، `IKERelationshipType` (۱۲ نوع رابطه) | ثبت گراف روابط سه‌جانبه با `SimilarityScore` (اعتماد) و `VersionTag` | `IKERepository` |
| ۷ | انتقال دانش | [`domain/transfer`](file:///d:/git2/offlline-hierarchical-rag/src/offline_rag/domain/transfer/__init__.py) | `TransferPackage` (AR)، `TransferDeliveryResolution` | کنترل فیلدهای الزامی لایه مقصد، اعمال داوری و بازگشت خودکار (Fallback) به آخرین بسته مقبول قبلی | `TransferPackageRepository` |
| ۸ | ارزیابی و داوری | [`domain/evaluation`](file:///d:/git2/offlline-hierarchical-rag/src/offline_rag/domain/evaluation/__init__.py) | `JudgeScore` (AR) | اعتبارسنجی ۷ معیار در برابر سقف‌های `JudgeRubric` (۲۰ امتیاز) و سنجش `JudgeThreshold` (۱۸) | متصل به موجودیت‌های هدف (`TransferPackage` و غیره) |
| ۹ | مدیریت RFP و پیشنهاد | [`domain/rfp`](file:///d:/git2/offlline-hierarchical-rag/src/offline_rag/domain/rfp/__init__.py) | `RFP` (AR)، `Proposal` (AR)، `FeedbackReport` (AR) | الزام پیوند `RFP` به `origin_issue_id` و الزام ارجاع به معیارها در `FeedbackReport` | `RFPRepository` |
| ۱۰ | مدیریت مدل و پرامپت | [`domain/model_prompt`](file:///d:/git2/offlline-hierarchical-rag/src/offline_rag/domain/model_prompt/__init__.py) | `OfflineModelSpec` (AR)، `PromptTemplate` (AR)، `PromptTaskCategory` (۱۲ دسته) | استخراج خودکار متغیرهای `{{...}}` و مونتاژ ایمن پرامپت (`assemble_messages`) | `ModelRegistryRepository`، `PromptTemplateRepository` |

---

## ۳. مشخصات فنی ۱۱ شیء مقداری تغییرناپذیر (Value Objects)

تمامی ۱۱ شیء مقداری در [`src/offline_rag/domain/shared/value_objects.py`](file:///d:/git2/offlline-hierarchical-rag/src/offline_rag/domain/shared/value_objects.py) با `@dataclass(frozen=True)` پیاده‌سازی شده‌اند و در صورت نقض ناوردایی، استثنای `InvalidValueObjectError` پرتاب می‌کنند:

1. **`NodeIdentifier`:** رشته غیرخالی منطبق با الگوی استاندارد `^N-[0-9A-Za-z]{2,12}(-[0-9A-Za-z]{1,12})*$` (مانند `N-000`، `N-301`).
2. **`LayerIndex`:** عدد صحیح (با رد مقادیر بولین) در بازه بسته `0 <= value <= 4`.
3. **`KnowledgeScope`:** مالکیت گره (`NodeIdentifier`)، سطح محرمانگی در `("public", "internal", "restricted", "confidential")`، حوزه کاربرد و لایه‌های مجاز دسترسی.
4. **`VersionTag`:** شماره توالی نسخه (`sequence_number >= 1`) و برچسب نسخه تغییرناپذیر (مانند `v1`، `v2`) همراه با متد `next_version()`.
5. **`ProgressPercentage`:** عدد در بازه `0.0 <= value <= 100.0` با متدهای کمکی `is_completed` و `is_not_started`.
6. **`JudgeRubric`:** چارچوب امتیازدهی شامل ۷ معیار پیش‌فرض با مجموع دقیق `20.0` امتیاز (`source_faithfulness: 5`، `completeness: 4`، `detail_level_fit: 3`، `uncertainty_handling: 3`، `clarity_and_structure: 2`، `source_citability: 2`، `no_unsupported_claims: 1`) و متد `validate_and_sum_scores()`.
7. **`JudgeThreshold`:** حد آستانه قبولی داور (پیش‌فرض `18.0` از `20.0`) و متد `is_satisfied_by(score)`.
8. **`RetrievalQuery`:** پرسش کاربر یا سامانه همراه با نرمال‌سازی نویسه‌های فارسی (`normalize_persian_text`)، شناسه گره درخواست‌کننده، `top_k` (`1..100`)، سقف محرمانگی و حداقل شباهت.
9. **`SimilarityScore`:** امتیاز شباهت نرمال‌شده در بازه `0.0 <= value <= 1.0` با متد `meets_threshold()`.
10. **`TransferDetailLevel`:** سطح جزئیات انتقال در مجموعه مجاز `("executive_high", "managerial", "technical_summary", "technical_detailed")` و نگاشت مستقیم از `LayerIndex` هدف.
11. **`MetadataRecord`:** رکورد فراداده ۱۸ فیلدی شامل `domain`، `sub_domain`، `node_id`، `project_id`، `author_or_owner`، `document_type`، `confidentiality`، `initial_version_tag`، `active_version_tag`، `related_issue_id`، `related_executor_id`، `related_rfp_id`، `status`، `priority`، `tags`، `source_reliability` (`0.0..1.0`)، `target_transfer_layer` و `suggested_detail_level`.

---

## ۴. رویدادهای دامنه (Domain Events)

Aggregate Rootها از کلاس پایه [`AggregateRoot`](file:///d:/git2/offlline-hierarchical-rag/src/offline_rag/domain/shared/base.py) ارث‌بری کرده و تغییرات وضعیت مهم را در قالب رویدادهای تغییرناپذیر ثبت می‌کنند:
- **`NodeRegisteredEvent`:** هنگام ثبت و اعتبارسنجی یک گره سازمانی جدید در `Node`.
- **`KnowledgeVersionCreatedEvent`:** هنگام ثبت اولیه یا افزودن نسخه جدید (`add_version`) در `KnowledgeItem`.
- **`IssueDecomposedEvent`:** هنگام الحاق یک `SubIssue` جدید به `Issue` والد.
- **`OutputJudgedEvent`:** هنگام محاسبه و ثبت امتیاز داور در `JudgeScore`.
- **`TransferPackageEvaluatedEvent`:** هنگام اعمال نتیجه داوری روی `TransferPackage` و تغییر وضعیت به `ACCEPTED` یا `REJECTED_BY_JUDGE`.

---

## ۵. پورت‌های کاربردی، انتزاع‌های ذخیره‌سازی و Test Doubleها

در [`src/offline_rag/application/ports/__init__.py`](file:///d:/git2/offlline-hierarchical-rag/src/offline_rag/application/ports/__init__.py) و [`src/offline_rag/infrastructure/persistence/in_memory/__init__.py`](file:///d:/git2/offlline-hierarchical-rag/src/offline_rag/infrastructure/persistence/in_memory/__init__.py) تمامی رابط‌های انتزاعی و پیاده‌سازی‌های `In-Memory` برای آزمون واحد فراهم شده‌اند:

| پورت / ریپازیتوری انتزاعی | پیاده‌سازی Test Double / In-Memory | کاربرد در معماری و آزمون واحد |
| --- | --- | --- |
| `UnitOfWork` | `InMemoryUnitOfWork` | مدیریت تراکنشی هر ۱۰ ریپازیتوری دامنه (`commit` / `rollback`) |
| `LayerRepository` / `NodeRepository` | `InMemoryLayerRepository` / `InMemoryNodeRepository` | نگهداری لایه‌ها، قواعد انتقال و گره‌های سازمانی |
| `KnowledgeRepository` | `InMemoryKnowledgeRepository` | نگهداری `KnowledgeItem`، تشخیص تکرار با هش محتوا و بازیابی قطعات نسخه فعال |
| `IssueRepository` / `ExecutorRepository` | `InMemoryIssueRepository` / `InMemoryExecutorRepository` | نگهداری مسائل، زیرمسائل و جست‌وجوی مجریان بر حسب تخصص |
| `IKERepository` | `InMemoryIKERepository` | نگهداری و پیمایش گراف روابط `Issue-Knowledge-Executor` |
| `TransferPackageRepository` | `InMemoryTransferPackageRepository` | نگهداری بسته‌های انتقال و بازیابی آخرین بسته یا آخرین بسته پذیرفته‌شده (`ACCEPTED`) |
| `RFPRepository` | `InMemoryRFPRepository` | نگهداری `RFP`، `Proposal` و `FeedbackReport` |
| `ModelRegistryRepository` / `PromptTemplateRepository` | `InMemoryModelRegistryRepository` / `InMemoryPromptTemplateRepository` | نگهداری مشخصات مدل‌های آفلاین و نسخه‌های قالب پرامپت |
| `VectorIndexPort` | `InMemoryVectorIndex` | نمایه‌سازی برداری در حافظه و جست‌وجوی کسینوسی با رعایت مرز گره و سطح محرمانگی |
| `DocumentStoragePort` | `InMemoryDocumentStorage` | ذخیره‌سازی بایت‌های سند در حافظه بدون وابستگی به دیسک |
| `CSVTransferExporterPort` | `InMemoryCSVTransferExporter` | تولید خروجی استاندارد CSV از `TransferPackage` |
| `EmbeddingModelPort` / `LLMModelPort` / `LVMModelPort` / `ModelRuntimePort` | `DeterministicMockEmbeddingAdapter` / `DeterministicMockLLMAdapter` / `DeterministicMockLVMAdapter` / `InMemoryModelRuntime` | اجرای کاملاً قطعی آزمون‌های واحد بدون نیاز به سرور مدل یا GPU |
